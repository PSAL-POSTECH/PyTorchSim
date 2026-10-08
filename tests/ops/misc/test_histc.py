import os
import sys
import torch
import torch._dynamo
sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from _pytorchsim_utils import test_result


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
    # 5.15.0, and PyTorchSimFrontend/rewrite_fx_graph.py is the answer:
    # counting is dtype-blind, so it casts in and back out.
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
    # The floating case must keep going to ATen -- the decomposition returns
    # NotImplemented there, which is also what stops it recursing on the cast
    # it just made.
    g = torch.Generator().manual_seed(1)
    cpu_x = torch.rand(n, generator=g) * (bins - 1)
    cpu_out = torch.histc(cpu_x, bins=bins, min=0, max=bins - 1)

    opt_fn = torch.compile(dynamic=False)(_histc)
    npu_out = opt_fn(cpu_x.to(device=device), bins, 0, bins - 1)

    assert npu_out.dtype == torch.float32, npu_out.dtype
    test_result("histc float32", npu_out, cpu_out)


if __name__ == "__main__":
    device = torch.device("npu:0")
    test_histc_integral(device)
    test_histc_float_unchanged(device)
