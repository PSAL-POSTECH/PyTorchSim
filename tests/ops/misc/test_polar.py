import os
import sys
import torch
import torch._dynamo
from torch.fx.experimental.proxy_tensor import make_fx
from torch._inductor.decomposition import select_decomp_table
sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from result_check import check_result

from pytorchsim.frontend.inductor.postgrad import pytorchsim_post_grad_pass


def _rope_like(a):
    """DeepSeek-V2's RoPE: polar on a transposed view out of a bmm; the transpose is what the test is about."""
    freqs = a.transpose(1, 2)
    return torch.view_as_real(torch.polar(torch.ones_like(freqs), freqs))


def test_polar_on_transposed(device, n=32):
    """polar on a transposed view: as a composite it would allocate its own contiguous result while Inductor's meta
    keeps the permutation (a stride assert); postgrad's real-pair rule computes it with ops this backend compiles."""
    x = torch.randn(1, n, n)
    cpu_out = _rope_like(x)

    opt_fn = torch.compile(dynamic=False)(_rope_like)
    npu_out = opt_fn(x.to(device=device))

    check_result("polar on a transposed view", npu_out, cpu_out)


def test_polar_contiguous(device, n=32):
    """The ordinary case, so a rewrite that only handles the transposed shape is caught."""
    def f(a):
        return torch.view_as_real(torch.polar(torch.ones_like(a), a))

    x = torch.randn(1, n, n)
    cpu_out = f(x)
    npu_out = torch.compile(dynamic=False)(f)(x.to(device=device))
    check_result("polar on a contiguous tile", npu_out, cpu_out)


def _unlowered(gm):
    """The ops of gm Inductor has no lowering for; the device pass runs after decomposition, so any it
    inserts must lower as they are."""
    from torch._inductor.lowering import lowerings
    return sorted({str(m.target) for m in gm.graph.nodes if callable(m.target)
                   and m.target not in lowerings and m.target.overloadpacket not in lowerings})


def test_pass_leaves_no_polar(n=32):
    """The device pass alone, on a traced polar of a transposed view: no aten.polar and no complex value
    is left, and the real pairs are polar's."""
    x = torch.randn(1, n, n)
    gm = make_fx(_rope_like, decomposition_table=select_decomp_table(), tracing_mode="fake")(x)
    pytorchsim_post_grad_pass(gm)
    left = [str(m.target) for m in gm.graph.nodes
            if m.target is torch.ops.aten.polar.default
            or (isinstance(m.meta.get("val"), torch.Tensor) and m.meta["val"].is_complex())]
    assert not left, f"left after the pass: {left}"
    assert not _unlowered(gm), f"no Inductor lowering for {_unlowered(gm)}"
    check_result("polar pass", gm(x), _rope_like(x))


if __name__ == "__main__":
    device = torch.device("npu:0")
    test_pass_leaves_no_polar()
    test_polar_on_transposed(device)
    test_polar_contiguous(device)
