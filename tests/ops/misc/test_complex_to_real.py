import os
import sys
import torch
import torch._dynamo
from torch.fx.experimental.proxy_tensor import make_fx
from torch._inductor.decomposition import select_decomp_table
sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from _pytorchsim_utils import test_result

from PyTorchSimFrontend.rewrite_fx_graph import ComplexToRealPairs

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
    # WITH INDUCTOR'S DECOMPOSITION TABLE, because that is the graph the pass
    # actually runs on. A bare make_fx keeps aten.polar.default, which this
    # backend decomposes one stage earlier (PyTorchSimFrontend/rewrite_fx_graph.py) and
    # which the pass therefore does not handle -- tracing without the table
    # measures a graph the pipeline never produces.
    return make_fx(fn, decomposition_table=select_decomp_table(),
                   tracing_mode="fake")(*args)


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

    ComplexToRealPairs()(gm.graph)
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
    ComplexToRealPairs()(gm.graph)
    after = [str(n.target) for n in _complex_nodes(gm.graph)]
    assert before == after, f"an ineligible component was rewritten: {before} -> {after}"
    print("unhandled complex op left alone:", after)


if __name__ == "__main__":
    device = torch.device("npu:0")
    test_pass_leaves_no_complex()
    test_unhandled_op_is_left_alone()
    test_rope_on_device(device)
