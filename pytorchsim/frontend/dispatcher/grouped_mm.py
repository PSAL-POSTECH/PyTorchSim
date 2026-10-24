"""aten._grouped_mm and transformers' grouped_mm_fallback (2-D A grouped by offs, times 3-D B) as one
Triton kernel in fused_moe form: a program is BM rows of one group, read from a table the TOG reads as
runtime attributes. No backward, so a call needing a gradient runs group by group."""

from typing import Optional

import torch
import triton
import triton.language as tl
from torch.library import wrap_triton

from pytorchsim import config
from pytorchsim.frontend.dispatcher.custom_fused_kernel import CustomFusedKernel, Decline


class GroupedMM(CustomFusedKernel):
    name = "torchsim::grouped_mm"
    #: By name: transformers defines grouped_mm_fallback only once imported, after install.
    takes = (torch.ops.aten._grouped_mm.default, "transformers::grouped_mm_fallback")

    def kernels(self):
        return (_grouped_mm_kernel,)

    def arguments(self, op, a, b, offs=None, bias=None, out_dtype=None):
        """aten's call, and transformers' (A, B, offs) as the same call without bias or out_dtype."""
        return a, b, offs, bias, out_dtype

    def plan(self, a, b, offs, bias, out_dtype):
        """(BM, BN, BK) for 2-D A times 3-D B grouped by offs, in one dtype, with no bias."""
        if a.dim() != 2 or b.dim() != 3 or offs is None:
            return Decline(f"A {a.dim()}-D, B {b.dim()}-D, offs {'given' if offs is not None else 'none'} "
                           "(it takes 2-D x 3-D)")
        if bias is not None or (out_dtype is not None and out_dtype != a.dtype):
            return Decline("a bias or another out_dtype")
        if a.dtype != b.dtype:
            return Decline(f"A {a.dtype} and B {b.dtype}")
        tiles = _tiles(b.shape[2], a.shape[1], a.element_size())
        if tiles is None:
            return Decline(f"no mm tile of {config.CONFIG_MACHINE.vector_lanes} rows fits the scratchpad")
        bm, bn, bk = tiles
        return {"BM": bm, "BN": bn, "BK": bk}

    def launch(self, plan, a: torch.Tensor, b: torch.Tensor, offs: Optional[torch.Tensor],
               bias: Optional[torch.Tensor], out_dtype: Optional[torch.dtype]) -> torch.Tensor:
        s, k = a.shape
        g, _, n = b.shape
        bm, bn, bk = plan["BM"], plan["BN"], plan["BK"]
        grp, first, rows, nb = _blocks(offs, s, g, bm)
        c = a.new_empty(s, n)
        wrap_triton(_grouped_mm_kernel)[(nb, triton.cdiv(n, bn), 1)](
            a.contiguous(), b, grp, first, rows, c, b.stride(0), b.stride(1), b.stride(2),
            k, triton.cdiv(k, bk) * bk, n, bm, bn, bk)
        return c

    def fallback(self, a, b, offs, bias, out_dtype):
        return _math(a, b, offs, bias, out_dtype)


GROUPED_MM = GroupedMM()


def install():
    GROUPED_MM.install()


def _tiles(n, k, esize):
    """(BM, BN, BK): BM the lanes, BN and BK the largest tile the machine's mm rule gives a BM-row block;
    None when it gives none."""
    from pytorchsim.frontend import tiling
    bm = int(config.CONFIG_MACHINE.vector_lanes)
    acc = 4
    tiles = tiling.gemm_tile_candidates(bm, n, k, acc + esize, esize)
    if not tiles:
        return None
    _, tn, tk = tiles[0]
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


def _math(a, b, offs, bias, out_dtype):
    """Group by group with matmul: rows offs[g-1]:offs[g] of A times B[g] into an [S, N] output whose rows
    past offs[-1] are zero, as transformers' fallback gives them; a bias is refused, as aten refuses it."""
    if bias is not None:
        raise RuntimeError("Bias not supported yet")
    c = a.new_zeros(a.shape[0], b.shape[-1])
    lo = 0
    for gi, hi in enumerate(offs.tolist()):
        c[lo:hi] = a[lo:hi] @ b[gi]
        lo = hi
    return c.to(out_dtype) if out_dtype is not None else c
