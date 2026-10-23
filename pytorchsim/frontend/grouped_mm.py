"""aten._grouped_mm on the npu (2-D A grouped by offs, times 3-D B) as one Triton kernel in fused_moe form: a
program is a block of BM rows of one group, its group, first row and row count read from a table the TOG reads as
runtime attributes; a block without rows runs its K loop zero times. Registered through torch.library."""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from pytorchsim import config

logger = config.setup_logger()


def _tiles(n, k, esize):
    """(BM, BN, BK): BM the lanes, BN and BK the largest tile the machine's mm rule gives a BM-row block."""
    from pytorchsim.codegen import tiling
    bm = int(config.CONFIG_MACHINE.vector_lanes)
    acc = 4
    _, tn, tk = tiling.gemm_tile_candidates(bm, n, k, acc + esize, esize)[0]
    return bm, tn, tk


@triton.jit
def _grouped_mm_kernel(a_ptr, b_ptr, grp_ptr, start_ptr, rows_ptr, c_ptr,
                       sbg: tl.constexpr, sbk: tl.constexpr, sbn: tl.constexpr, K: tl.constexpr, KP: tl.constexpr,
                       N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    g = tl.load(grp_ptr + pid_m)
    nr = tl.load(rows_ptr + pid_m)
    rm = tl.arange(0, BM)
    rows = tl.load(start_ptr + pid_m) + rm
    live = rm < nr
    rn = pid_n * BN + tl.arange(0, BN)
    ncols = rn < N
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, tl.where(nr > 0, KP, 0), BK):
        kk = k0 + rk
        a = tl.load(a_ptr + rows[:, None] * K + kk[None, :], mask=live[:, None] & (kk < K)[None, :], other=0.0)
        b = tl.load(b_ptr + g * sbg + kk[:, None] * sbk + rn[None, :] * sbn,
                    mask=(kk < K)[:, None] & ncols[None, :], other=0.0)
        acc += tl.dot(a, b)
    tl.store(c_ptr + rows[:, None] * N + rn[None, :], acc.to(c_ptr.dtype.element_ty),
             mask=live[:, None] & ncols[None, :])


def _blocks(offs, s, g, bm):
    """Per block of at most `bm` rows of one group: its group, first row and row count (0 past the last group's
    rows); the block count is the most any offs over `s` rows and `g` groups needs, and no more than `s`."""
    end = offs.to(torch.int64)
    start = torch.cat([end.new_zeros(1), end[:-1]])
    size = end - start
    bend = torch.cumsum((size + bm - 1) // bm, 0)
    bstart = torch.cat([bend.new_zeros(1), bend[:-1]])
    nb = min(s, (s + g * (bm - 1)) // bm)
    b = torch.arange(nb, device=offs.device)
    grp = (b[:, None] >= bend[None, :]).sum(1).clamp(max=g - 1)
    j = b - bstart.index_select(0, grp)
    first = start.index_select(0, grp) + j * bm
    rows = (size.index_select(0, grp) - j * bm).clamp(min=0, max=bm)
    return grp.to(torch.int32), first.to(torch.int32), rows.to(torch.int32), nb


@triton_op("torchsim::grouped_mm", mutates_args=())
def grouped_mm(a: torch.Tensor, b: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    s, k = a.shape
    g, _, n = b.shape
    bm, bn, bk = _tiles(n, k, a.element_size())
    grp, first, rows, nb = _blocks(offs, s, g, bm)
    c = a.new_empty(s, n)
    wrap_triton(_grouped_mm_kernel)[(nb, triton.cdiv(n, bn), 1)](
        a.contiguous(), b, grp, first, rows, c, b.stride(0), b.stride(1), b.stride(2),
        k, triton.cdiv(k, bk) * bk, n, bm, bn, bk)
    return c


def _math(a, b, offs, bias, out_dtype):
    """Group by group with matmul: rows offs[g-1]:offs[g] of A times B[g] (aten's semantics for 2-D x 3-D)."""
    out, lo = [], 0
    for gi, hi in enumerate(offs.tolist()):
        out.append(a[lo:hi] @ b[gi])
        lo = hi
    c = torch.cat(out) if out else a.new_empty(0, b.shape[-1])
    if bias is not None:
        c = c + bias
    return c.to(out_dtype) if out_dtype is not None else c


def _decline(a, b, offs, bias, out_dtype):
    """Why the kernel does not take this call, or None."""
    if a.dim() != 2 or b.dim() != 3 or offs is None:
        return f"A {a.dim()}-D, B {b.dim()}-D, offs {'given' if offs is not None else 'none'} (it takes 2-D x 3-D)"
    if bias is not None or (out_dtype is not None and out_dtype != a.dtype):
        return "a bias or another out_dtype"
    if a.dtype != b.dtype:
        return f"A {a.dtype} and B {b.dtype}"
    if torch.is_grad_enabled() and (a.requires_grad or b.requires_grad):
        return "a gradient (the kernel has no backward)"
    return None


def _aten_grouped_mm(a, b, offs=None, bias=None, out_dtype=None):
    why = _decline(a, b, offs, bias, out_dtype)
    if why is not None:
        logger.info("[grouped_mm] %s x %s: math, %s", tuple(a.shape), tuple(b.shape), why)
        return _math(a, b, offs, bias, out_dtype)
    if not torch.compiler.is_compiling():
        return _math(a, b, offs, bias, out_dtype)
    return grouped_mm(a, b, offs)


_lib = None


def install():
    """Register aten._grouped_mm's npu kernel above autograd and functionalization, and at the backend key,
    where a call made under inference_mode lands."""
    global _lib
    if _lib is None:
        _lib = torch.library.Library("aten", "IMPL")
        for key in ("AutogradPrivateUse1", "PrivateUse1"):
            _lib.impl("_grouped_mm", _aten_grouped_mm, key)
