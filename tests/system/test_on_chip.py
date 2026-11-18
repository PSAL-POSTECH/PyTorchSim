"""A graph's intermediates kept in the scratchpad (autotune/on_chip.py): a tensor a kernel makes, every use of
which is a matmul operand, kept where it fits beside the others; its maker and its readers state it by tensor
index, and a kept and a moved build of one source do not share a Triton cache key."""
import json
import os
import sys

sys.path.insert(0, os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"))
sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests", "system"))

import torch  # noqa: E402  the npu backend loads pytorchsim whole, before its parts are imported
from pytorchsim.backend import compiler_input  # noqa: E402
from pytorchsim.frontend.inductor.autotune import on_chip  # noqa: E402
from test_graph_prefetch import _Graph, _compiled_metas  # noqa: E402

_A = torch.ops.aten


def _kept(g):
    """The names of the nodes `g` keeps on chip."""
    return sorted(n.name for n in on_chip.elected(g))


def test_a_matmul_operand_a_kernel_makes_is_kept():
    """Breaks if an int8 activation an _int_mm alone reads, through a view, is not kept, or its input is."""
    g = _Graph()
    x, w = g.input("x", 1024, 1024), g.input("w", 1024, 1024)
    q = g.op(_A.mul.Tensor, x, x)
    y = g.op(_A.mm.default, g.op(_A.view.default, q, [1024, 1024]), w)
    g.graph.output(y)
    assert _kept(g) == [q.name], _kept(g)


def test_an_elementwise_co_use_is_not_kept():
    """Breaks if an intermediate an elementwise op reads beside a matmul is kept."""
    g = _Graph()
    x, w = g.input("x", 1024, 1024), g.input("w", 1024, 1024)
    q = g.op(_A.mul.Tensor, x, x)
    g.graph.output((g.op(_A.mm.default, q, w), g.op(_A.add.Tensor, q, x)))
    assert _kept(g) == [], _kept(g)


def test_a_returned_intermediate_is_not_kept():
    """Breaks if an intermediate the graph returns is kept: it has to reach DRAM."""
    g = _Graph()
    x, w = g.input("x", 1024, 1024), g.input("w", 1024, 1024)
    q = g.op(_A.mul.Tensor, x, x)
    g.graph.output((g.op(_A.mm.default, q, w), q))
    assert _kept(g) == [], _kept(g)


def test_an_intermediate_over_the_budget_is_not_kept():
    """Breaks if an intermediate larger than the resident budget a lane is kept."""
    g = _Graph()
    x, w = g.input("x", 8192, 16384), g.input("w", 16384, 128)
    q = g.op(_A.mul.Tensor, x, x)
    g.graph.output(g.op(_A.mm.default, q, w))
    assert _kept(g) == [], _kept(g)


def test_the_spec_and_the_compile_key_hold_the_statement():
    """Breaks if on_chip's names are not the tensor indices of the JSON spec, or a kept and a moved build of one
    source share a Triton cache key: the spec rides the compile's options, which the key hashes."""
    from triton.backends.pytorchsim.compiler import CPUOptions
    args = [{"name": n, "role": r} for n, r in (("arg_A", "in"), ("arg_B", "in"), ("out_ptr0", "out"))]
    meta = {"kernel_name": "mm", "args": args, "graph_prefetch": [], "on_chip": ["arg_A", "out_ptr0"],
            "signature": {}, "constants": {}}
    assert compiler_input.on_chip(meta) == (0, 2)
    spec = lambda chip: json.dumps(compiler_input.spec_dict(  # noqa: E731
        [("in", 64), ("in", 64), ("out", 64)], (1, 1, 1), on_chip=chip))
    assert json.loads(spec((0, 2)))["on_chip"] == [0, 2] and "on_chip" not in json.loads(spec(()))
    assert CPUOptions(pytorchsim_spec=spec((0, 2))).hash() != CPUOptions(pytorchsim_spec=spec(())).hash()


def test_a_compiled_chain_states_the_kept_activation_at_both_ends():
    """Breaks if relu(x @ w1), which only the second mm reads, is not stated on chip by the kernel making it
    and by the mm reading it, or a graph input or the returned output is."""
    g = torch.Generator().manual_seed(0)
    x, w1, w2 = (torch.randn(*s, generator=g).to(torch.bfloat16)
                 for s in ((256, 1024), (1024, 1024), (1024, 2048)))
    got = _compiled_metas(lambda x, w1, w2: torch.relu(x @ w1) @ w2, x, w1, w2)
    roles = sorted(tuple(a["role"] for a in m["args"] if a["name"] in m["on_chip"]) for m in got)
    assert roles == [("in",), ("out",)], [(m["kernel_name"], m["on_chip"]) for m in got]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
