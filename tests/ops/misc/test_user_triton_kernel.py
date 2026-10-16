"""A Triton kernel a model registers (torch.library.triton_op + wrap_triton) runs on the npu.

What breaks it:
  - the output is written through a pointer the kernel computes and a jit helper's tl.store, so a
    role read off the source's `tl.store(<arg>` would call o_ptr unwritten and never read it back;
    the role is torch's own mutation analysis of the kernel.
  - the same kernel launched twice with different constexprs and grids: its source is the same,
    so its compiled kernel must be keyed by what the spec bakes in, not the source alone.
"""
import os
import sys

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"),
                                "tests"))
from _pytorchsim_utils import test_result


@triton.jit
def _store_through(ptrs, x):
    tl.store(ptrs, x)


@triton.jit
def _scale_kernel(x_ptr, o_ptr, N: tl.constexpr, BLOCK: tl.constexpr, ALPHA: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    out_ptrs = o_ptr + offs
    x = tl.load(x_ptr + offs, mask=offs < N, other=0.0)
    _store_through(out_ptrs, x * ALPHA)


@triton_op("torchsim_test::scale", mutates_args=())
def scale(x: torch.Tensor, alpha: float, block: int) -> torch.Tensor:
    o = torch.empty_like(x)
    wrap_triton(_scale_kernel)[(x.numel() // block, 1, 1)](x, o, x.numel(), block, alpha)
    return o


def test_user_triton_kernel(device):
    x = torch.randn(1024)
    y = torch.randn(512)
    f = torch.compile(lambda x, y: (scale(x, 2.0, 256), scale(y, 3.0, 128)), dynamic=False)
    ox, oy = f(x.to(device), y.to(device))
    test_result("UserTritonKernel stored through a helper", ox, x * 2.0)
    test_result("UserTritonKernel relaunched with other constexprs", oy, y * 3.0)


if __name__ == "__main__":
    test_user_triton_kernel(torch.device("npu:0"))
