"""A graph's resident operand (autotune/graph_prefetch.py), by XLA's cross-program prefetch rule: a graph
input every use of which is a matmul operand or a reduction input, in XLA's order, the first that fits; the
kernel form holding it states it by tensor index, every other kernel (); a refusal is an error."""
import os
import sys
import tempfile
from unittest import mock

sys.path.insert(0, os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"))

import torch  # noqa: E402  the npu backend loads pytorchsim whole, before its parts are imported
from pytorchsim.backend import codecache, compiler_bridge, compiler_input  # noqa: E402
from pytorchsim.frontend.inductor.autotune import graph_prefetch  # noqa: E402

_A = torch.ops.aten


class _Graph:
    """What the election reads of Inductor's GraphLowering: the FX module, its nodes' meta vals."""

    def __init__(self):
        self.graph = torch.fx.Graph()
        self.module = torch.fx.GraphModule(torch.nn.Module(), self.graph)

    def input(self, name, *shape):
        node = self.graph.placeholder(name)
        node.meta["val"] = torch.empty(*shape, dtype=torch.bfloat16, device="meta")
        return node

    def op(self, target, *args):
        """A call of `target` on `args`, its meta val what it makes of theirs."""
        node = self.graph.call_function(target, args)
        node.meta["val"] = target(*[a.meta["val"] if isinstance(a, torch.fx.Node) else a for a in args])
        return node


def _held(g):
    """The placeholder targets `g` holds."""
    return sorted(graph_prefetch.elected(g))


def _attention(g, q, k, v):
    """softmax(q @ k^T) @ v as XLA lowers attention: two batched dots, the softmax between them."""
    s = g.op(_A.bmm.default, q, g.op(_A.transpose.int, k, 1, 2))
    return g.op(_A.bmm.default, g.op(_A._softmax.default, s, -1, False), v)


def test_two_mms_elect_the_larger_weight():
    """Breaks if the smaller input is held, or an input reached through a view is not seen as one."""
    g = _Graph()
    x, w1, w2 = g.input("x", 1024, 4096), g.input("w1", 4096, 1024), g.input("w2", 4096, 4096)
    g.op(_A.mm.default, x, w1)
    g.op(_A.mm.default, x, g.op(_A.t.default, w2))
    assert _held(g) == ["w2"]


def test_an_elementwise_co_use_disqualifies():
    """Breaks if an input a matmul reads is held though an elementwise op reads it too (XLA: an activation),
    as a scale fused into the matmul would."""
    g = _Graph()
    x, w = g.input("x", 4096, 1024), g.input("w", 1024, 512)
    g.op(_A.mm.default, x, w)
    g.op(_A.mul.Tensor, x, x)
    assert _held(g) == ["w"]


def test_an_addmm_full_bias_is_disqualified_and_its_weight_elected():
    """Breaks if the bias (an elementwise add's operand, however large) is held, or the weight is not, or the
    weight's statement index counts the bias wrong or counts a scalar (the index is a tensor's)."""
    g = _Graph()
    bias, h, w = g.input("bias", 1024, 4096), g.input("h", 1024, 256), g.input("w", 256, 4096)
    g.op(_A.addmm.default, bias, h, w)
    assert _held(g) == ["w"]
    args = [{"name": n, "role": "in"} for n in ("in_ptr0", "arg_A", "arg_B")]
    args.append({"name": "out", "role": "out"})
    meta = {"kernel_name": "addmm", "args": args, "graph_prefetch": ["arg_B"], "constants": {},
            "signature": {"in_ptr0": "*bf16", "ks0": "i32", "arg_A": "*bf16", "arg_B": "*bf16",
                          "out": "*bf16"}}
    assert compiler_input.graph_prefetch(meta) == (2,)


def test_a_bmm_input_is_elected_when_largest():
    g = _Graph()
    x, w, a, b = (g.input("x", 1024, 1024), g.input("w", 1024, 1024), g.input("a", 8, 2048, 1024),
                  g.input("b", 8, 1024, 512))
    g.op(_A.mm.default, x, w)
    g.op(_A.bmm.default, a, b)
    assert _held(g) == ["a"]


def test_attention_qkv_tie_by_their_consumers_output():
    """Breaks if an XLA-form attention's q, k, v are not candidates, or a tie in size is not broken by the
    consumers' output (q, k feed the scores, v the output), then the buffer id; a scaled q is no candidate."""
    g = _Graph()
    q, k, v = (g.input(n, 8, 1024, 128) for n in "qkv")
    _attention(g, q, k, v)
    assert _held(g) == ["q"]
    g = _Graph()
    q, k, v = (g.input(n, 8, 1024, 128) for n in "qkv")
    _attention(g, g.op(_A.mul.Tensor, q, 0.125), k, v)
    assert _held(g) == ["k"]
    g = _Graph()
    q, k, v = g.input("q", 1, 8, 1024, 128), g.input("k", 1, 8, 1024, 128), g.input("v", 1, 8, 1024, 256)
    g.op(_A.scaled_dot_product_attention.default, q, k, v)
    assert _held(g) == ["v"]


def test_a_largest_input_that_does_not_fit_is_skipped():
    """Breaks if an input over the compiler's resident budget (contract/copy_node.resident_fits) is held, or
    the next candidate that fits is not."""
    g = _Graph()
    x, w = g.input("x", 256, 4096), g.input("w", 4096, 20480)
    g.op(_A.mm.default, x, w)
    assert _held(g) == ["x"]


def test_a_reduction_only_input_is_elected_and_held_by_no_kernel():
    """Breaks if an input only reductions read is no candidate (XLA prefetches it), or a reduction kernel,
    which is no resident form, states it."""
    g = _Graph()
    x = g.input("x", 4096, 1024)
    g.op(_A.sum.dim_IntList, x, [1])
    assert _held(g) == ["x"]
    from torch._inductor.virtualized import V
    with V.set_graph_handler(g):
        assert graph_prefetch.statement(object(), [{"name": "in_ptr0", "role": "in", "buffer": "x"}]) == []


def test_a_returned_input_is_no_candidate():
    g = _Graph()
    x, w = g.input("x", 1024, 1024), g.input("w", 1024, 4096)
    g.op(_A.mm.default, x, w)
    g.graph.output((w,))
    assert _held(g) == ["x"]


def test_a_kernel_of_no_resident_form_states_nothing():
    """Breaks if a kernel that is no resident form (here a pointwise one reading the held input) states
    anything but (), which overrides the compiler's in-kernel default."""
    g = _Graph()
    x, w = g.input("x", 1024, 4096), g.input("w", 4096, 4096)
    g.op(_A.mm.default, x, w)
    from torch._inductor.virtualized import V
    with V.set_graph_handler(g):
        assert graph_prefetch.statement(object(), [{"name": "in_ptr0", "role": "in", "buffer": "w"}]) == []


def test_a_refusal_is_an_error_naming_the_kernel_and_arg():
    """Breaks if the compiler's refusal to hold the elected operand is retried, retiled or built holding
    nothing: the two sides disagree, and torchsim_compile raises."""
    kind = compiler_bridge.compiler_contract("failure").SPAD_OVERFLOW
    calls = []

    def refuse(*a, **kw):
        calls.append(a)
        raise compiler_bridge.CompilerError("x", failure={"kind": kind, "usage": 8, "budget": 4,
                                                          "message": "over"})
    meta = {"kernel_name": "mm", "signature": {"arg_A": "*bf16", "arg_B": "*bf16", "XBLOCK": "constexpr"},
            "constants": {}, "numels": {"xnumel": 4096}, "fixed_config": {"XBLOCK": 1024},
            "args": [{"name": n, "role": "in", "buffer": b, "dtype": "bfloat16", "numel": 64}
                     for n, b in (("arg_A", "x"), ("arg_B", "w"))], "graph_prefetch": ["arg_B"]}
    with tempfile.TemporaryDirectory() as dump, mock.patch.dict(os.environ, {"TORCHSIM_DUMP_PATH": dump}), \
            mock.patch.object(codecache.compiler_bridge, "run_pipeline", refuse):
        try:
            codecache.torchsim_compile("def kernel(): pass\n", meta, "mm")
        except compiler_bridge.CompilerError as e:
            assert "mm" in str(e) and "arg_B" in str(e) and kind in str(e), e
        else:
            raise AssertionError("a refused hold compiled")
    assert len(calls) == 1, calls


def test_the_build_key_holds_the_statement():
    """Breaks if an elected and a non-elected build of one source, constants and args share a workdir."""
    meta = {"signature": {}, "constants": {}, "args": [], "graph_prefetch": ["arg_B"]}
    with tempfile.TemporaryDirectory() as dump, mock.patch.dict(os.environ, {"TORCHSIM_DUMP_PATH": dump}):
        streamed = dict(meta, graph_prefetch=[])
        assert codecache._write_path("src", meta) != codecache._write_path("src", streamed)


def _compiled_metas(fn, *xs):
    """The meta of each kernel torch.compile of `fn` defines, with the compiler faked away."""
    from torch._inductor import config as inductor_config
    got = []

    def compile_(src, meta, kernel_name):
        got.append(meta)
        return lambda *args: None
    torch._dynamo.reset()
    with mock.patch.object(codecache, "torchsim_compile", compile_), \
            inductor_config.patch(fx_graph_cache=False):
        torch.compile(fn, dynamic=False)(*[x.to("npu:0") for x in xs])
    return got


def test_a_compiled_graph_states_one_holder():
    """Breaks if a compiled graph of two mms holding graph inputs states a held operand for both, for the
    smaller, or for the one holding a computed operand."""
    g = torch.Generator().manual_seed(0)
    x, w1, w2 = (torch.randn(*s, generator=g).to(torch.bfloat16)
                 for s in ((256, 1024), (1024, 1024), (1024, 2048)))
    got = _compiled_metas(lambda x, w1, w2: torch.relu(x @ w1) @ w2, x, w1, w2)
    assert sorted(m["graph_prefetch"] for m in got) == [[], ["arg_B"]], got


def test_a_compiled_addmm_states_its_weight():
    g = torch.Generator().manual_seed(2)
    bias, x, w = (torch.randn(*s, generator=g).to(torch.bfloat16)
                  for s in ((256, 768), (256, 1024), (1024, 768)))
    (meta,) = _compiled_metas(torch.addmm, bias, x, w)
    assert meta["graph_prefetch"] == ["arg_B"], meta
    assert next(a for a in meta["args"] if a["name"] == "arg_B")["buffer"] == "arg2_1", meta


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
