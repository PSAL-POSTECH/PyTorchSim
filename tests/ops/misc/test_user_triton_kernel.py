"""A Triton kernel a model registers (torch.library.triton_op + wrap_triton) runs on the npu.

What breaks it:
  - the output is written through a pointer the kernel computes and a jit helper's tl.store, so a
    role read off the source's `tl.store(<arg>` would call o_ptr unwritten and never read it back;
    the role is torch's own mutation analysis of the kernel, and the roles recorded say so
    (x_ptr in, o_ptr inout; an all-inout fallback fails the role check).
  - the same kernel launched twice with different constexprs and grids: its source is the same,
    so its compiled kernel must be keyed by what the spec bakes in, not the source alone.
  - the same kernel launched twice with the same constexprs but other extents and grid: Inductor
    gives both one name (its cache keys triton_meta only), so the npu must give each its own.
  - a user kernel that happens to say `M = 128`, `BLOCK_M = 64` and `rm % M`: the mm-template
    rewriter would drop the modulo; a model's source must reach the compiler as written.
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


@triton.jit
def _copy_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=offs < n, other=0.0) + 1.0, mask=offs < n)


@triton.jit
def _wrap_kernel(x_ptr, o_ptr, BLOCK: tl.constexpr):
    M = 128
    BLOCK_M = 64
    rm = tl.arange(0, BLOCK)
    tl.store(o_ptr + rm, tl.load(x_ptr + rm % M) * BLOCK_M)


@triton_op("torchsim_test::scale", mutates_args=())
def scale(x: torch.Tensor, alpha: float, block: int) -> torch.Tensor:
    o = torch.empty_like(x)
    wrap_triton(_scale_kernel)[(x.numel() // block, 1, 1)](x, o, x.numel(), block, alpha)
    return o


@triton_op("torchsim_test::add_one", mutates_args=())
def add_one(x: torch.Tensor) -> torch.Tensor:
    o = torch.empty_like(x)
    wrap_triton(_copy_kernel)[(x.numel() // 128, 1, 1)](x, o, x.numel(), 128)
    return o


@triton_op("torchsim_test::wrap", mutates_args=())
def wrap(x: torch.Tensor) -> torch.Tensor:
    o = x.new_empty(256)
    wrap_triton(_wrap_kernel)[(1, 1, 1)](x, o, 256)
    return o


def _check_roles(kernel, want):
    from pytorchsim.frontend.inductor.codegen import kernel_metadata
    got = {n: r for n, r in kernel_metadata.ROLES_BY_KERNEL.items() if n.startswith(kernel)}
    if not got or any(r != want for r in got.values()):
        print(f"|UserTritonKernel roles of {kernel} Test Failed| got {got}, want {want}")
        sys.exit(1)
    print(f"|UserTritonKernel roles of {kernel} Test Passed| {got}")


def test_user_triton_kernel(device):
    torch.manual_seed(0)
    x = torch.randn(1024)
    y = torch.randn(512)
    f = torch.compile(lambda x, y: (scale(x, 2.0, 256), scale(y, 3.0, 128)), dynamic=False)
    ox, oy = f(x.to(device), y.to(device))
    test_result("UserTritonKernel stored through a helper", ox, x * 2.0)
    test_result("UserTritonKernel relaunched with other constexprs", oy, y * 3.0)
    _check_roles("_scale_kernel", ["in", "inout"])

    a, b = torch.randn(512), torch.randn(1024)
    g = torch.compile(lambda a, b: (add_one(a), add_one(b)), dynamic=False)
    oa, ob = g(a.to(device), b.to(device))
    test_result("UserTritonKernel relaunched with other extents", oa, a + 1.0)
    test_result("UserTritonKernel relaunched with other extents (second)", ob, b + 1.0)

    w = torch.randn(128)
    ow = torch.compile(wrap, dynamic=False)(w.to(device))
    test_result("UserTritonKernel source reaches the compiler as written", ow,
                w[torch.arange(256) % 128] * 64)


if __name__ == "__main__":
    test_user_triton_kernel(torch.device("npu:0"))
