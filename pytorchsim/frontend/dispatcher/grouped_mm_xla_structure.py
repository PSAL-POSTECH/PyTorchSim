"""Validation only (TORCHSIM_VALIDATION_XLA_STRUCTURE=megablox): aten._grouped_mm as megablox's structure, the one
XLA runs on v6e -- a program per (row tile, group) visit, V = tiles_m + G - 1 at most, each reading its group's
weight again and computing all TM rows of the tile, TM as torch_xla picks it; only the group's rows are stored."""

from typing import Optional

import torch
import triton
import triton.language as tl
from torch.library import wrap_triton

from pytorchsim.frontend.dispatcher.custom_fused_kernel import Decline
from pytorchsim.frontend.dispatcher.grouped_mm import GroupedMM, _math

#: torch_xla 2.9's gmm tiling (custom_kernel._get_tiling_size_for_gmm_kernel), which both its registered experts
#: path and the stock one take: TM from twice a group's mean rows, TK and TN from K and N, each a multiple of 128.
TM_LIMIT, TKN_LIMIT = 512, 2048


class GroupedMMMegablox(GroupedMM):
    name = "torchsim::grouped_mm_xla_structure"

    def kernels(self):
        return (_megablox_kernel,)

    def plan(self, a, b, offs, bias, out_dtype):
        """(TM, BN, BK): torch_xla's tiling; TK and TN as the largest power of two within it (triton's blocks), TM
        exactly or declined, as the visits are counted in it."""
        declined = super().plan(a, b, offs, bias, out_dtype)
        if isinstance(declined, str):
            return declined
        tm, tk, tn = _xla_tiling(a.shape[0], a.shape[1], b.shape[2], b.shape[0])
        if tm & (tm - 1):
            return Decline(f"torch_xla's row tile {tm} is no power of two, which a triton block must be")
        return {"BM": tm, "BN": 1 << (tn.bit_length() - 1), "BK": 1 << (tk.bit_length() - 1)}

    def launch(self, plan, a: torch.Tensor, b: torch.Tensor, offs: Optional[torch.Tensor],
               bias: Optional[torch.Tensor], out_dtype: Optional[torch.dtype]) -> torch.Tensor:
        s, k = a.shape
        g, _, n = b.shape
        tm, bn, bk = plan["BM"], plan["BN"], plan["BK"]
        group, tile = _visits(offs, tm, triton.cdiv(s, tm) + g - 1)
        c = a.new_zeros(s, n)
        wrap_triton(_megablox_kernel)[(group.shape[0], triton.cdiv(n, bn), 1)](
            a.contiguous(), b, group, tile, offs, c, b.stride(0), b.stride(1), b.stride(2),
            s, k, triton.cdiv(k, bk) * bk, n, g, tm, bn, bk)
        return c

    def fallback(self, a, b, offs, bias, out_dtype):
        return _math(a, b, offs, bias, out_dtype)


GROUPED_MM_MEGABLOX = GroupedMMMegablox()


def install():
    GROUPED_MM_MEGABLOX.install()


def _xla_tiling(m, k, n, g):
    """(TM, TK, TN) as torch_xla 2.9's gmm picks them when no tiling is given."""
    def within(x, limit):
        if x <= 128:
            return 128
        if x < limit:
            return (x + 127) // 128 * 128
        return next((c for c in range(limit, 511, -128) if x % c == 0), limit)
    return min(within(2 * m // g, TM_LIMIT), m), within(k, TKN_LIMIT), within(n, TKN_LIMIT)


def _visits(offs, tm, visits):
    """(group, row tile) of each of `visits` programs, as megablox's make_group_metadata orders them: groups in
    order, each over the row tiles its rows touch; group G past the last visit."""
    start = torch.nn.functional.pad(offs[:-1], (1, 0))
    first = torch.div(start, tm, rounding_mode="floor")
    tiles = torch.where(offs > start, torch.div(offs - 1, tm, rounding_mode="floor") - first + 1, 0)
    ends = torch.cumsum(tiles, 0)
    v = torch.arange(visits, device=offs.device)
    group = (ends[None, :] <= v[:, None]).sum(1)
    g = group.clamp(max=offs.shape[0] - 1)
    tile = first[g] + v - (ends[g] - tiles[g])
    return group, tile


@triton.jit
def _megablox_kernel(a_ptr, b_ptr, group_ptr, tile_ptr, offs_ptr, c_ptr,
                     sbg: tl.constexpr, sbk: tl.constexpr, sbn: tl.constexpr, M: tl.constexpr, K: tl.constexpr,
                     KP: tl.constexpr, N: tl.constexpr, G: tl.constexpr, TM: tl.constexpr, BN: tl.constexpr,
                     BK: tl.constexpr):
    """Visit pid: all TM rows of its row tile times its group's weight, read over K by BK; rows of the group stored."""
    pid = tl.program_id(0)
    pid_n = tl.program_id(1)
    g = tl.load(group_ptr + pid)
    live_g = g < G
    t = tl.load(tile_ptr + pid)
    s0 = tl.load(offs_ptr + g - 1, mask=live_g & (g > 0), other=0)
    s1 = tl.load(offs_ptr + g, mask=live_g, other=0)
    if s1 > s0:
        rows = t * TM + tl.arange(0, TM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        bg = b_ptr + g.to(tl.int64) * sbg
        acc = tl.zeros((TM, BN), dtype=tl.float32)
        for k0 in range(0, KP, BK):
            kk = k0 + rk
            a = tl.load(a_ptr + rows[:, None] * K + kk[None, :], mask=(rows < M)[:, None] & (kk < K)[None, :],
                        other=0.0)
            b = tl.load(bg + kk[:, None] * sbk + rn[None, :] * sbn, mask=(kk < K)[:, None] & (rn < N)[None, :],
                        other=0.0)
            acc += tl.dot(a, b)
        own = (rows >= s0) & (rows < s1)
        tl.store(c_ptr + rows[:, None] * N + rn[None, :], acc.to(c_ptr.dtype.element_ty),
                 mask=own[:, None] & (rn < N)[None, :])
