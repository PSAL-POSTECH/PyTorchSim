import os
import sys
import torch
import torch._dynamo
from torch.fx.experimental.proxy_tensor import make_fx
from torch._inductor.decomposition import select_decomp_table
sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from _pytorchsim_utils import test_result

from pytorchsim.frontend.post_grad_pass import pytorchsim_post_grad_pass

aten = torch.ops.aten


def _rope(x, a):
    # DeepSeek-V2's apply_rotary_emb, which is the only complex path in the
    # model suite: polar for the angles, the query read as complex pairs, one
    # complex multiply, back to real.
    freqs = a.transpose(1, 2)
    fc = torch.polar(torch.ones_like(freqs), freqs) * 0.7
    xc = torch.view_as_complex(x.reshape(1, 32, 32, 2))
    return torch.view_as_real(xc * fc).flatten(-2)


def _trace(fn, *args):
    """The graph Inductor hands its post-grad passes: make_fx with Inductor's decomposition table, which
    keeps aten.polar, so the device pass rewrites it before the complex values."""
    return make_fx(fn, decomposition_table=select_decomp_table(),
                   tracing_mode="fake")(*args)


def _unlowered(gm):
    """The ops of gm Inductor has no lowering for; the device pass runs after decomposition, so any it
    inserts must lower as they are."""
    from torch._inductor.lowering import lowerings
    return sorted({str(m.target) for m in gm.graph.nodes if m.op == "call_function"
                   and m.target not in lowerings and m.target.overloadpacket not in lowerings})


def _complex_nodes(graph):
    out = []
    for n in graph.nodes:
        v = n.meta.get("val", None)
        if isinstance(v, torch.Tensor) and v.is_complex():
            out.append(n)
    return out


def test_pass_leaves_no_complex():
    # The pass on its own, on a traced graph and no device. Inductor does not
    # lower complex arithmetic -- every complex op becomes an extern call, and
    # on this backend that means the CPU, so the work leaves the simulator.
    # After the pass the graph must hold no complex value at all.
    x, a = torch.randn(1, 32, 64), torch.randn(1, 32, 32)
    gm = _trace(_rope, x, a)

    before = _complex_nodes(gm.graph)
    assert before, "the traced graph should have complex nodes to rewrite"

    pytorchsim_post_grad_pass(gm)
    gm.recompile()

    after = _complex_nodes(gm.graph)
    assert not after, f"complex nodes survived the pass: {[str(n.target) for n in after]}"

    test_result("complex-to-real (rewritten graph)", gm(x, a), _rope(x, a))


def test_rope_on_device(device):
    # And end to end. Without the pass this passes too -- through the CPU
    # fallback -- so the value check is not what pins the rewrite; the graph
    # check above is. This one pins that the rewrite did not break the answer
    # once real kernels compute it.
    x, a = torch.randn(1, 32, 64), torch.randn(1, 32, 32)
    cpu_out = _rope(x, a)
    npu_out = torch.compile(_rope, dynamic=False)(x.to(device=device), a.to(device=device))
    test_result("complex-to-real (RoPE on device)", npu_out, cpu_out)


def test_complex_scalar_multiply():
    """A complex Python scalar on either side is multiplied as a complex number, not
    broadcast over the real pair."""
    def f(x):
        z = torch.view_as_complex(x.reshape(1, 32, 32, 2))
        return torch.cat([torch.view_as_real(z * (0.6 - 0.8j) * 0.5),
                          torch.view_as_real(2j * z)], -1).flatten(-2)

    x = torch.randn(1, 32, 64)
    gm = _trace(f, x)
    pytorchsim_post_grad_pass(gm)
    gm.recompile()
    after = _complex_nodes(gm.graph)
    assert not after, f"complex nodes survived the pass: {[str(n.target) for n in after]}"
    assert not _unlowered(gm), f"no Inductor lowering for {_unlowered(gm)}"
    test_result("complex-to-real (complex scalar)", gm(x), f(x))


def test_components_are_judged_apart():
    """A component the pass cannot say in real arithmetic leaves a separate one it can rewritten; every node it
    makes keeps the stack trace of the node it replaces, and its from_node names that node."""
    def f(x, y):
        a = torch.view_as_real(torch.view_as_complex(x) * (0.6 - 0.8j))
        b = torch.view_as_real(torch.exp(torch.view_as_complex(y)))
        return a, b

    x, y = torch.randn(4, 8, 2), torch.randn(4, 8, 2)
    gm = _trace(f, x, y)
    for n in gm.graph.nodes:
        n.meta["stack_trace"] = f"user.py:{n.name}"
        n.meta["from_node"] = [n.name]
    before = {n.name for n in gm.graph.nodes}
    pytorchsim_post_grad_pass(gm)
    for n in gm.graph.nodes:
        if n.name in before:
            continue
        src = n.meta.get("from_node", [])
        last = src[-1] if src else None
        assert getattr(last, "name", None) in before, f"{n.name}: from_node {src}"
    left = sorted(str(n.target) for n in _complex_nodes(gm.graph))
    assert left == ["aten.exp.default", "aten.view_as_complex.default"], f"complex nodes left: {left}"
    bare = [n.name for n in gm.graph.nodes if n.op == "call_function" and "stack_trace" not in n.meta]
    assert not bare, f"nodes without provenance: {bare}"
    out = gm(x, y)
    test_result("complex-to-real (components apart)", out[0], f(x, y)[0])


def test_row_mm_stays_real():
    """The GEMV rewrite takes real floating operands only: a complex 1 x K mm stays one mm."""
    def f(a, b):
        return torch.view_as_real(torch.view_as_complex(a) @ torch.view_as_complex(b))

    a, b = torch.randn(1, 8, 2), torch.randn(8, 4, 2)
    gm = _trace(f, a, b)
    pytorchsim_post_grad_pass(gm)
    mms = [n for n in gm.graph.nodes if n.target is aten.mm.default]
    assert len(mms) == 1, f"complex mm rewritten: {[str(n.target) for n in gm.graph.nodes]}"
    test_result("complex 1 x K mm", gm(a, b), f(a, b))


def test_unhandled_op_is_left_alone():
    # WHOLE OR NOTHING. A complex op the pass cannot say in real arithmetic
    # makes the whole component ineligible -- half a rewrite is a graph with
    # complex values whose producers are gone.
    def f(x):
        z = torch.view_as_complex(x.reshape(1, 32, 32, 2))
        return torch.view_as_real(torch.exp(z))          # exp is not on the list

    x = torch.randn(1, 32, 64)
    gm = _trace(f, x)
    before = [str(n.target) for n in _complex_nodes(gm.graph)]
    pytorchsim_post_grad_pass(gm)
    after = [str(n.target) for n in _complex_nodes(gm.graph)]
    assert before == after, f"an ineligible component was rewritten: {before} -> {after}"
    print("unhandled complex op left alone:", after)


if __name__ == "__main__":
    device = torch.device("npu:0")
    test_pass_leaves_no_complex()
    test_complex_scalar_multiply()
    test_components_are_judged_apart()
    test_row_mm_stays_real()
    test_unhandled_op_is_left_alone()
    test_rope_on_device(device)
