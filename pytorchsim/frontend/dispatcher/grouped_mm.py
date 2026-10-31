"""aten._grouped_mm and transformers' grouped_mm_fallback (2-D A grouped by offs, times 3-D B) as one
Triton kernel: a program is one group with rows, BM rows at a time, read from a table the TOG reads as runtime
attributes; a B tile holding all of K is read once per group. No backward: a gradient runs group by group."""

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
        tiles = _tiles(b.shape[2], a.shape[1], a.element_size(), -(-a.shape[0] // min(b.shape[0], a.shape[0])))
        if tiles is None:
            return Decline(f"no mm tile of {config.CONFIG_MACHINE.vector_lanes} rows fits the scratchpad")
        bm, bn, bk = tiles
        return {"BM": bm, "BN": bn, "BK": bk}

    def launch(self, plan, a: torch.Tensor, b: torch.Tensor, offs: Optional[torch.Tensor],
               bias: Optional[torch.Tensor], out_dtype: Optional[torch.dtype]) -> torch.Tensor:
        s, k = a.shape
        g, _, n = b.shape
        bm, bn, bk = plan["BM"], plan["BN"], plan["BK"]
        programs = min(g, s)
        c = a.new_empty(s, n)
        wrap_triton(_grouped_mm_kernel)[(programs, triton.cdiv(n, bn), 1)](
            a.contiguous(), b, _groups(offs, programs), offs, c, b.stride(0), b.stride(1), b.stride(2),
            k, triton.cdiv(k, bk) * bk, n, g, bm, bn, bk)
        return c

    def fallback(self, a, b, offs, bias, out_dtype):
        return _math(a, b, offs, bias, out_dtype)


GROUPED_MM = GroupedMM()


def install():
    GROUPED_MM.install()


def _tiles(n, k, esize, rows):
    """(BM, BN, BK) of the machine's mm tiles for BM = lanes, ranked: all of K first (each weight read once), then
    the fewest N tiles where a group of `rows` rows is one row block, else no masked columns first; None for none."""
    from pytorchsim.frontend import tiling
    bm = int(config.CONFIG_MACHINE.vector_lanes)
    acc = 4
    tiles = tiling.gemm_tile_candidates(bm, n, k, acc + esize, esize)
    if not tiles:
        return None
    blocks = lambda t: -(-n // t[1])
    one = rows <= bm
    rank = lambda t: (t[2] < k, blocks(t) if one else n % t[1] != 0, n % t[1] != 0 if one else blocks(t))
    _, tn, tk = min(tiles, key=rank)
    return bm, tn, tk


@triton.jit
def _grouped_mm_kernel(a_ptr, b_ptr, group_ptr, offs_ptr, c_ptr,
                       sbg: tl.constexpr, sbk: tl.constexpr, sbn: tl.constexpr, K: tl.constexpr, KP: tl.constexpr,
                       N: tl.constexpr, G: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """Group g, the program_id(0)-th group with rows (rows offs[g-1]:offs[g]): its rows times B[g], BM rows at a
    time, a B tile of all of K (BK == KP) read once for the group, as v6e's grouped matmul reads each weight once.
    A program past the last such group has g == G and reads nothing."""
    pid = tl.program_id(0)
    pid_n = tl.program_id(1)
    g = tl.load(group_ptr + pid)
    live_g = g < G
    s0 = tl.load(offs_ptr + g - 1, mask=live_g & (g > 0), other=0)
    nr = tl.load(offs_ptr + g, mask=live_g, other=0) - s0
    bg = b_ptr + g.to(tl.int64) * sbg
    rm = tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    ncols = rn < N
    rk = tl.arange(0, BK)
    if BK == KP:
        if nr > 0:
            b = tl.load(bg + rk[:, None] * sbk + rn[None, :] * sbn,
                        mask=(rk < K)[:, None] & ncols[None, :], other=0.0)
            for m0 in range(0, nr, BM):
                rows = s0 + m0 + rm
                live = (m0 + rm) < nr
                a = tl.load(a_ptr + rows[:, None] * K + rk[None, :], mask=live[:, None] & (rk < K)[None, :],
                            other=0.0)
                acc = tl.dot(a, b)
                tl.store(c_ptr + rows[:, None] * N + rn[None, :], acc.to(c_ptr.dtype.element_ty),
                         mask=live[:, None] & ncols[None, :])
    else:
        for m0 in range(0, nr, BM):
            rows = s0 + m0 + rm
            live = (m0 + rm) < nr
            acc = tl.zeros((BM, BN), dtype=tl.float32)
            for k0 in range(0, KP, BK):
                kk = k0 + rk
                a = tl.load(a_ptr + rows[:, None] * K + kk[None, :], mask=live[:, None] & (kk < K)[None, :],
                            other=0.0)
                b = tl.load(bg + kk[:, None] * sbk + rn[None, :] * sbn,
                            mask=(kk < K)[:, None] & ncols[None, :], other=0.0)
                acc += tl.dot(a, b)
            tl.store(c_ptr + rows[:, None] * N + rn[None, :], acc.to(c_ptr.dtype.element_ty),
                     mask=live[:, None] & ncols[None, :])


def _groups(offs, programs):
    """From offs (each group's end row), the group of each of `programs` programs: the i-th group with rows for
    program i, the count of groups whose running count of groups with rows is at most i (G for one left over)."""
    ranks = torch.cumsum(offs > torch.nn.functional.pad(offs[:-1], (1, 0)), 0)
    return (ranks[None, :] <= torch.arange(programs, device=offs.device)[:, None]).sum(1)


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
