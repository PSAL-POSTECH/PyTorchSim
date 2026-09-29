"""fp8 tensors as kernel arguments, both directions.

Both formats. They are not interchangeable: E4M3 keeps one more mantissa bit
than E5M2 and one less exponent bit, so the same f32 quantises differently, and
the compiler picks between them with a single token in the vsetvli it emits.

Compared bit for bit, not by tolerance: fp8 is coarse enough that a wrong
format still lands within any sane atol -- E5M2 and E4M3 agree on 1.0.
"""
import os
import sys
import torch
import torch._dynamo
sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from _pytorchsim_utils import test_result


def _bitwise(name, res, ref):
    test_result(name, res.cpu().view(torch.uint8), ref.view(torch.uint8))


def test_fp8_arg_in(device, dtype=torch.float8_e5m2, size=(128,)):
    """An fp8 tensor goes IN: 8 bits off the bus, widened in the kernel."""
    def square(a):
        f = a.to(torch.float32)
        return f * f
    x = torch.randn(size).to(dtype)
    res = torch.compile(dynamic=False)(square)(x.to(device=device))
    _bitwise(f"Fp8ArgIn[{dtype}]", res, square(x))


def test_fp8_arg_out(device, dtype=torch.float8_e5m2, size=(128,)):
    """An fp8 tensor comes OUT: f32 narrowed in the kernel and stored as 8 bits."""
    def narrow(a):
        return (a * 2.0).to(dtype)
    x = torch.randn(size)
    res = torch.compile(dynamic=False)(narrow)(x.to(device=device))
    _bitwise(f"Fp8ArgOut[{dtype}]", res, narrow(x))


if __name__ == "__main__":
    device = torch.device("npu:0")
    for dt in (torch.float8_e5m2, torch.float8_e4m3fn):
        test_fp8_arg_in(device, dt)
        test_fp8_arg_out(device, dt)
