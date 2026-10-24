"""sort on the npu: a bitonic network per row whose partners tl.gather reads, as the v6e sorts; a row is an
[L, D] tile padded to a power of two, its key one int64 (float bits made ordered, flipped when descending)
carried as two int32 words with the int32 index, so every sort is stable. Needs PTB #335's tl.gather."""

import torch
import triton
import triton.language as tl
from torch.library import wrap_triton

from pytorchsim import config
from pytorchsim.frontend.dispatcher.custom_fused_kernel import (
    CustomFusedKernel, Decline, fits_scratchpad, next_pow2)

_KEYS = (torch.int32, torch.int64, torch.float32)


class Sort(CustomFusedKernel):
    name = "torchsim::sort"
    takes = (torch.ops.aten.sort.stable, torch.ops.aten.sort.default)

    def kernels(self):
        return (_sort_kernel,)

    def arguments(self, op, x, *args, **kwargs):
        """(x, a non-negative dim, descending) of sort.stable's keywords or sort's positionals."""
        if op is torch.ops.aten.sort.stable:
            dim, descending = kwargs.get("dim", -1), kwargs.get("descending", False)
        else:
            dim, descending = (list(args) + [-1, False][len(args):])[:2]
            dim, descending = kwargs.get("dim", dim), kwargs.get("descending", descending)
        return x, dim % max(x.dim(), 1), bool(descending)

    def plan(self, x, dim, descending):
        """The [L, D] tile of a row of the last axis padded to n, on L = min(lanes, n) lanes."""
        if x.dtype not in _KEYS:
            return Decline(f"{x.dtype} is not a key the kernel orders")
        if x.dim() == 0 or x.numel() == 0:
            return Decline("an empty or 0-d input")
        if dim != x.dim() - 1:
            return Decline(f"dim {dim} is not the last of {x.dim()}")
        n = next_pow2(x.shape[-1])
        lanes = min(int(config.CONFIG_MACHINE.vector_lanes), n)
        if not fits_scratchpad(_tiles(lanes, n // lanes)):
            return Decline(f"a row of {x.shape[-1]} overruns the scratchpad budget")
        return {"L": lanes, "D": n // lanes, "LOG": n.bit_length() - 1}

    def launch(self, plan, x: torch.Tensor, dim: int,
               descending: bool) -> tuple[torch.Tensor, torch.Tensor]:
        e = int(x.shape[-1])
        v = torch.empty_like(x)
        i = torch.empty(x.shape, dtype=torch.int64, device=x.device)
        wrap_triton(_sort_kernel)[(x.numel() // e, 1, 1)](
            x.contiguous(), v, i, e, plan["L"], plan["D"], plan["LOG"], descending,
            x.dtype == torch.float32)
        return v, i

    def fallback(self, x, dim, descending):
        return _network(x, dim, descending)


SORT = Sort()


def install():
    SORT.install()


@triton.jit
def _sort_kernel(x_ptr, v_ptr, i_ptr, E, L: tl.constexpr, D: tl.constexpr, LOG: tl.constexpr,
                 DESCENDING: tl.constexpr, FLOAT: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, L)[:, None]
    reg = tl.arange(0, D)[None, :]
    pos = reg * L + lane
    live = pos < E
    if FLOAT:
        got = tl.load(x_ptr + row * E + pos, mask=live, other=0.0)
        bits = tl.where(got != got, 0x7FC00000, tl.where(got == 0, 0, got.to(tl.int32, bitcast=True)))
        k64 = (bits ^ ((bits >> 31) & 0x7FFFFFFF)).to(tl.int64)
    else:
        k64 = tl.load(x_ptr + row * E + pos, mask=live, other=0).to(tl.int64)
    if DESCENDING:
        k64 = ~k64
    k64 = tl.where(live, k64, 0x7FFFFFFFFFFFFFFF)
    hi = (k64 >> 32).to(tl.int32)
    lo = k64.to(tl.int32)
    i = pos.to(tl.int32)
    for s in tl.static_range(1, LOG + 1):
        for j in tl.static_range(s):
            if (1 << (s - 1 - j)) < L:
                at = tl.broadcast_to(lane ^ (1 << (s - 1 - j)), (L, D))
                hp = tl.gather(hi, at, 0)
                lp = tl.gather(lo, at, 0)
                ip = tl.gather(i, at, 0)
            else:
                at = tl.broadcast_to(reg ^ ((1 << (s - 1 - j)) // L), (L, D))
                hp = tl.gather(hi, at, 1)
                lp = tl.gather(lo, at, 1)
                ip = tl.gather(i, at, 1)
            lower = (pos & (1 << (s - 1 - j))) == 0
            up = (pos & (1 << s)) == 0
            lou, lpu = lo.to(tl.uint32, bitcast=True), lp.to(tl.uint32, bitcast=True)
            less = (hi < hp) | ((hi == hp) & ((lou < lpu) | ((lo == lp) & (i < ip))))
            take = (less == lower) == up
            hi = tl.where(take, hi, hp)
            lo = tl.where(take, lo, lp)
            i = tl.where(take, i, ip)
    k64 = (hi.to(tl.int64) << 32) | lo.to(tl.uint32, bitcast=True).to(tl.int64)
    if DESCENDING:
        k64 = ~k64
    if FLOAT:
        bits = k64.to(tl.int32)
        val = (bits ^ ((bits >> 31) & 0x7FFFFFFF)).to(tl.float32, bitcast=True)
    else:
        val = k64
    tl.store(v_ptr + row * E + pos, val.to(v_ptr.dtype.element_ty), mask=live)
    tl.store(i_ptr + row * E + pos, i.to(tl.int64), mask=live)


def _tiles(lanes, d):
    """One program's tiles: the int64 key, three words and their partners, a step's masks, the index."""
    return [(lanes, d, 8)] + [(lanes, d, 4)] * 8 + [(lanes, d, 8)]


def _network(x, dim, descending):
    """The kernel's network as aten ops along `dim`: (values, int64 indices), equal keys in their order."""
    if x.dim() == 0:
        v, i = _network(x.view(1), 0, descending)
        return v.view(()), i.view(())
    if x.shape[dim] == 0:
        return x.clone(), torch.empty(x.shape, dtype=torch.int64, device=x.device)
    t = x.movedim(dim, -1)
    e = int(t.shape[-1])
    n = next_pow2(e)
    if t.is_floating_point():
        f = t.double() if t.element_size() == 8 else t.float()
        f = torch.where(f == 0, torch.zeros_like(f), f)
        bits = torch.where(f.isnan(), torch.full_like(f, float("nan")), f)
        bits = bits.view(torch.int64 if f.element_size() == 8 else torch.int32)
        key = (bits ^ ((bits >> (8 * f.element_size() - 1)) & torch.iinfo(bits.dtype).max)).to(torch.int64)
    else:
        key = t.to(torch.int64)
    if descending:
        key = ~key
    key = torch.cat([key, key.new_full((*key.shape[:-1], n - e), torch.iinfo(torch.int64).max)], -1)
    pos = torch.arange(n, device=x.device).expand(key.shape)
    idx = pos
    for s in range(1, n.bit_length()):
        for j in range(s):
            partner = pos ^ (1 << (s - 1 - j))
            kp, ip = key.gather(-1, partner), idx.gather(-1, partner)
            less = (key < kp) | ((key == kp) & (idx < ip))
            take = (less == ((pos & (1 << (s - 1 - j))) == 0)) == ((pos & (1 << s)) == 0)
            key, idx = torch.where(take, key, kp), torch.where(take, idx, ip)
    idx = idx[..., :e]
    return t.gather(-1, idx).movedim(-1, dim), idx.movedim(-1, dim)
