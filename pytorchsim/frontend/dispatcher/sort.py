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
BLOCK_REGS = 64


class Sort(CustomFusedKernel):
    name = "torchsim::sort"
    takes = (torch.ops.aten.sort.stable, torch.ops.aten.sort.default)

    def kernels(self):
        return (_sort_kernel, _sort_blocks, _merge_far, _merge_near)

    def arguments(self, op, x, *args, **kwargs):
        """(x, a non-negative dim, descending) of sort.stable's keywords or sort's positionals."""
        if op is torch.ops.aten.sort.stable:
            dim, descending = kwargs.get("dim", -1), kwargs.get("descending", False)
        else:
            dim, descending = (list(args) + [-1, False][len(args):])[:2]
            dim, descending = kwargs.get("dim", dim), kwargs.get("descending", descending)
        return x, dim % max(x.dim(), 1), bool(descending)

    def plan(self, x, dim, descending):
        """The [L, D] tile of a row of the last axis padded to n, on L = min(lanes, n) lanes; a row
        longer than BLOCK_REGS registers a lane is sorted in blocks of that many and merged."""
        if x.dtype not in _KEYS:
            return Decline(f"{x.dtype} is not a key the kernel orders")
        if x.dim() == 0 or x.numel() == 0:
            return Decline("an empty or 0-d input")
        if dim != x.dim() - 1:
            return Decline(f"dim {dim} is not the last of {x.dim()}")
        n = next_pow2(x.shape[-1])
        lanes = min(int(config.CONFIG_MACHINE.vector_lanes), n)
        d = min(n // lanes, BLOCK_REGS)
        if not fits_scratchpad(_tiles(lanes, d)):
            return Decline(f"a row of {x.shape[-1]} overruns the scratchpad budget")
        return {"L": lanes, "D": d, "LOG": n.bit_length() - 1, "N": n}

    def launch(self, plan, x: torch.Tensor, dim: int,
               descending: bool) -> tuple[torch.Tensor, torch.Tensor]:
        e = int(x.shape[-1])
        v = torch.empty_like(x)
        i = torch.empty(x.shape, dtype=torch.int64, device=x.device)
        rows, lanes, d, n = x.numel() // e, plan["L"], plan["D"], plan["N"]
        flt = x.dtype == torch.float32
        if lanes * d == n:
            wrap_triton(_sort_kernel)[(rows, 1, 1)](
                x.contiguous(), v, i, e, lanes, d, plan["LOG"], descending, flt)
            return v, i
        block = lanes * d
        log_b = block.bit_length() - 1
        k = torch.empty((rows * n,), dtype=torch.int64, device=x.device)
        j = torch.empty((rows * n,), dtype=torch.int32, device=x.device)
        wrap_triton(_sort_blocks)[(rows, n // block, 1)](
            x.contiguous(), k, j, e, n, lanes, d, log_b, descending, flt)
        for s in range(log_b + 1, plan["LOG"] + 1):
            for far in range(s - 1, log_b - 1, -1):
                wrap_triton(_merge_far)[(rows, n // (2 * block), 1)](k, j, n, s, 1 << far, lanes, d)
            wrap_triton(_merge_near)[(rows, n // block, 1)](
                k, j, v, i, e, n, s, lanes, d, log_b, descending, flt, s == plan["LOG"])
        return v, i

    def fallback(self, x, dim, descending):
        return _network(x, dim, descending)


SORT = Sort()


def install():
    SORT.install()


@triton.jit
def _keys(x_ptr, at, live, DESCENDING: tl.constexpr, FLOAT: tl.constexpr):
    """The ordered int64 keys at `at` (float bits made ordered, flipped when descending), the
    largest key past the row's end."""
    if FLOAT:
        got = tl.load(x_ptr + at, mask=live, other=0.0)
        bits = tl.where(got != got, 0x7FC00000, tl.where(got == 0, 0, got.to(tl.int32, bitcast=True)))
        k64 = (bits ^ ((bits >> 31) & 0x7FFFFFFF)).to(tl.int64)
    else:
        k64 = tl.load(x_ptr + at, mask=live, other=0).to(tl.int64)
    if DESCENDING:
        k64 = ~k64
    return tl.where(live, k64, 0x7FFFFFFFFFFFFFFF)


@triton.jit
def _store_sorted(v_ptr, i_ptr, at, live, k64, i, DESCENDING: tl.constexpr, FLOAT: tl.constexpr):
    """The keys back as the input's values, and the indices as int64, where `live`."""
    if DESCENDING:
        k64 = ~k64
    if FLOAT:
        bits = k64.to(tl.int32)
        val = (bits ^ ((bits >> 31) & 0x7FFFFFFF)).to(tl.float32, bitcast=True)
    else:
        val = k64
    tl.store(v_ptr + at, val.to(v_ptr.dtype.element_ty), mask=live)
    tl.store(i_ptr + at, i.to(tl.int64), mask=live)


@triton.jit
def _exchange(hi, lo, i, lane, reg, pos, up, STEP: tl.constexpr, L: tl.constexpr, D: tl.constexpr):
    """One compare-exchange of the network at distance STEP inside the [L, D] tile; `up` where the
    pair is put in ascending order."""
    if STEP < L:
        at = tl.broadcast_to(lane ^ STEP, (L, D))
        hp = tl.gather(hi, at, 0)
        lp = tl.gather(lo, at, 0)
        ip = tl.gather(i, at, 0)
    else:
        at = tl.broadcast_to(reg ^ (STEP // L), (L, D))
        hp = tl.gather(hi, at, 1)
        lp = tl.gather(lo, at, 1)
        ip = tl.gather(i, at, 1)
    lower = (pos & STEP) == 0
    lou, lpu = lo.to(tl.uint32, bitcast=True), lp.to(tl.uint32, bitcast=True)
    less = (hi < hp) | ((hi == hp) & ((lou < lpu) | ((lo == lp) & (i < ip))))
    take = (less == lower) == up
    return tl.where(take, hi, hp), tl.where(take, lo, lp), tl.where(take, i, ip)


@triton.jit
def _sort_kernel(x_ptr, v_ptr, i_ptr, E, L: tl.constexpr, D: tl.constexpr, LOG: tl.constexpr,
                 DESCENDING: tl.constexpr, FLOAT: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, L)[:, None]
    reg = tl.arange(0, D)[None, :]
    pos = reg * L + lane
    live = pos < E
    k64 = _keys(x_ptr, row * E + pos, live, DESCENDING, FLOAT)
    hi = (k64 >> 32).to(tl.int32)
    lo = k64.to(tl.int32)
    i = pos.to(tl.int32)
    for s in tl.static_range(1, LOG + 1):
        up = (pos & (1 << s)) == 0
        for j in tl.static_range(s):
            hi, lo, i = _exchange(hi, lo, i, lane, reg, pos, up, 1 << (s - 1 - j), L, D)
    k64 = (hi.to(tl.int64) << 32) | lo.to(tl.uint32, bitcast=True).to(tl.int64)
    _store_sorted(v_ptr, i_ptr, row * E + pos, live, k64, i, DESCENDING, FLOAT)


@triton.jit
def _sort_blocks(x_ptr, k_ptr, j_ptr, E, N, L: tl.constexpr, D: tl.constexpr, LOGB: tl.constexpr,
                 DESCENDING: tl.constexpr, FLOAT: tl.constexpr):
    """Each [L, D] block of a row sorted, ascending or descending by its place, as the network's
    first LOGB stages leave it: keys and indices into the row's scratch."""
    row = tl.program_id(0)
    lane = tl.arange(0, L)[:, None]
    reg = tl.arange(0, D)[None, :]
    pos = tl.program_id(1) * (L * D) + reg * L + lane
    live = pos < E
    k64 = _keys(x_ptr, row * E + pos, live, DESCENDING, FLOAT)
    hi = (k64 >> 32).to(tl.int32)
    lo = k64.to(tl.int32)
    i = pos.to(tl.int32)
    for s in tl.static_range(1, LOGB + 1):
        up = (pos & (1 << s)) == 0
        for j in tl.static_range(s):
            hi, lo, i = _exchange(hi, lo, i, lane, reg, pos, up, 1 << (s - 1 - j), L, D)
    k64 = (hi.to(tl.int64) << 32) | lo.to(tl.uint32, bitcast=True).to(tl.int64)
    tl.store(k_ptr + row * N + pos, k64)
    tl.store(j_ptr + row * N + pos, i)


@triton.jit
def _merge_far(k_ptr, j_ptr, N, S, DIST, L: tl.constexpr, D: tl.constexpr):
    """Stage S's compare-exchange at distance DIST, a block or more: one block of the pairs' lower
    halves against the block DIST above it, element by element."""
    row = tl.program_id(0)
    q = tl.program_id(1) * (L * D)
    t = tl.arange(0, D)[None, :] * L + tl.arange(0, L)[:, None]
    pos = (q // DIST) * (2 * DIST) + q % DIST + t
    at = row * N + pos
    k = tl.load(k_ptr + at)
    kp = tl.load(k_ptr + at + DIST)
    i = tl.load(j_ptr + at)
    ip = tl.load(j_ptr + at + DIST)
    hi, hp = (k >> 32).to(tl.int32), (kp >> 32).to(tl.int32)
    lo, lp = k.to(tl.int32), kp.to(tl.int32)
    lou, lpu = lo.to(tl.uint32, bitcast=True), lp.to(tl.uint32, bitcast=True)
    less = (hi < hp) | ((hi == hp) & ((lou < lpu) | ((lo == lp) & (i < ip))))
    take = less == ((pos & (1 << S)) == 0)
    tl.store(k_ptr + at, tl.where(take, k, kp))
    tl.store(k_ptr + at + DIST, tl.where(take, kp, k))
    tl.store(j_ptr + at, tl.where(take, i, ip))
    tl.store(j_ptr + at + DIST, tl.where(take, ip, i))


@triton.jit
def _merge_near(k_ptr, j_ptr, v_ptr, i_ptr, E, N, S, L: tl.constexpr, D: tl.constexpr,
                LOGB: tl.constexpr, DESCENDING: tl.constexpr, FLOAT: tl.constexpr,
                FINAL: tl.constexpr):
    """Stage S's compare-exchanges inside each block, below distance L * D; FINAL writes the
    sorted values and indices, else the keys and indices go back to the scratch."""
    row = tl.program_id(0)
    lane = tl.arange(0, L)[:, None]
    reg = tl.arange(0, D)[None, :]
    pos = tl.program_id(1) * (L * D) + reg * L + lane
    k64 = tl.load(k_ptr + row * N + pos)
    i = tl.load(j_ptr + row * N + pos)
    hi = (k64 >> 32).to(tl.int32)
    lo = k64.to(tl.int32)
    up = (pos & (1 << S)) == 0
    for j in tl.static_range(LOGB):
        hi, lo, i = _exchange(hi, lo, i, lane, reg, pos, up, 1 << (LOGB - 1 - j), L, D)
    k64 = (hi.to(tl.int64) << 32) | lo.to(tl.uint32, bitcast=True).to(tl.int64)
    if FINAL:
        _store_sorted(v_ptr, i_ptr, row * E + pos, pos < E, k64, i, DESCENDING, FLOAT)
    else:
        tl.store(k_ptr + row * N + pos, k64)
        tl.store(j_ptr + row * N + pos, i)


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
