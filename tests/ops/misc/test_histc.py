import os
import sys
import torch
import torch._dynamo
from torch import fx
from torch._subclasses.fake_tensor import FakeTensorMode
sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from _pytorchsim_utils import test_result

from pytorchsim.frontend.rewrite_fx_graph import npu_post_grad_pass

aten = torch.ops.aten


def _histc(x, bins, lo, hi):
    return torch.histc(x, bins=bins, min=lo, max=hi)


def test_histc_integral(device, n=64, bins=8):
    # An op with no npu kernel falls back, and the fallback is the CPU one, so
    # what this device supports is what ATen's CPU kernel supports. histc is a
    # place where that is narrower than CUDA:
    #
    #     NotImplementedError: "histogram_cpu" not implemented for 'Int'
    #
    # transformers' MoE router (integrations/moe.py) counts tokens per expert
    # with histc and picks its input dtype by DEVICE NAME -- float for "cpu"
    # and "mps", integral otherwise -- so `npu` gets the branch written for
    # CUDA and the fallback then lands on the very kernel the first branch
    # exists to avoid. That is what stopped DeepSeek-V3 under transformers
    # 5.15.0; rewrite_fx_graph.rewrite_aten_histc counts it on the device.
    #
    # Both integral widths are here because they take different paths back:
    # the counts return in the INPUT's dtype, which is the contract the CUDA
    # branch promised its caller.
    g = torch.Generator().manual_seed(0)
    for dtype in (torch.int32, torch.int64):
        cpu_x = torch.randint(0, bins, (n,), generator=g).to(dtype)
        cpu_out = torch.histc(cpu_x.float(), bins=bins, min=0,
                              max=bins - 1).to(dtype)

        opt_fn = torch.compile(dynamic=False)(_histc)
        npu_out = opt_fn(cpu_x.to(device=device), bins, 0, bins - 1)

        assert npu_out.dtype == dtype, (
            f"histc should return the input's dtype: {npu_out.dtype} != {dtype}")
        test_result(f"histc {dtype}", npu_out, cpu_out)


def test_histc_float_unchanged(device, n=64, bins=8):
    """A floating histc keeps aten's kernel: the device pass leaves it and logs why."""
    g = torch.Generator().manual_seed(1)
    cpu_x = torch.rand(n, generator=g) * (bins - 1)
    cpu_out = torch.histc(cpu_x, bins=bins, min=0, max=bins - 1)

    opt_fn = torch.compile(dynamic=False)(_histc)
    npu_out = opt_fn(cpu_x.to(device=device), bins, 0, bins - 1)

    assert npu_out.dtype == torch.float32, npu_out.dtype
    test_result("histc float32", npu_out, cpu_out)


def _histc_graph(x, bins, lo, hi):
    """A graph of one aten.histc of x as the post-grad passes see it; built by hand, since make_fx cannot
    trace an integral histc on the CPU."""
    mode = FakeTensorMode()
    g = fx.Graph()
    p = g.placeholder("x")
    p.meta["val"] = mode.from_tensor(x)
    h = g.call_function(aten.histc.default, (p, bins, lo, hi))
    with mode:
        h.meta["val"] = torch.empty(bins, dtype=x.dtype)
    g.output(h)
    return fx.GraphModule(torch.nn.Module(), g)


def test_pass_counts_on_the_device(n=64, bins=8):
    """The device pass alone, on a traced integral histc: bins that are the values leave no aten.histc;
    other edges keep one, at float32, and both count what the CPU counts."""
    g = torch.Generator().manual_seed(2)
    x = torch.randint(0, bins, (n,), generator=g).to(torch.int32)
    for lo, hi, left in ((0, bins - 1, 0), (0, 2 * bins - 1, 1)):
        gm = _histc_graph(x, bins, lo, hi)
        npu_post_grad_pass(gm)
        histcs = [n for n in gm.graph.nodes if n.target is aten.histc.default]
        assert len(histcs) == left, f"[{lo}, {hi}]: {len(histcs)} aten.histc left, expected {left}"
        test_result(f"histc pass [{lo}, {hi}]", gm(x), torch.histc(x.float(), bins, lo, hi).to(x.dtype))


if __name__ == "__main__":
    device = torch.device("npu:0")
    test_pass_counts_on_the_device()
    test_histc_integral(device)
    test_histc_float_unchanged(device)
