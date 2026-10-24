"""aten.sort on the npu: a bitonic network per row whose partners are read by tl.gather, as the v6e sorts.
A row is an [L, D] tile padded to a power of two, on as many lanes as it fills; its key is one int64 (float bits made ordered, flipped
when descending) carried as two int32 words with the int32 index, so every sort it runs is stable."""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from pytorchsim import config

logger = config.setup_logger()

_KEYS = (torch.int32, torch.int64, torch.float32)


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
    """The tiles one program of _sort_kernel holds: the int64 key, three words and their partners,
    the masks of a step, and the int64 index stored."""
    return [(lanes, d, 8)] + [(lanes, d, 4)] * 8 + [(lanes, d, 8)]


def _next_pow2(n):
    return 1 << max(0, (int(n) - 1).bit_length())


def _plan(x):
    """(L, D, LOG) of the [L, D] tile one program sorts a row of x's last axis in: the row padded to a
    power of two n, on L = min(lanes, n) lanes."""
    n = _next_pow2(x.shape[-1])
    lanes = min(int(config.CONFIG_MACHINE.vector_lanes), n)
    return lanes, n // lanes, n.bit_length() - 1


@triton_op("torchsim::sort", mutates_args=())
def sort(x: torch.Tensor, descending: bool) -> tuple[torch.Tensor, torch.Tensor]:
    e = int(x.shape[-1])
    lanes, d, log = _plan(x)
    v = torch.empty_like(x)
    i = torch.empty(x.shape, dtype=torch.int64, device=x.device)
    wrap_triton(_sort_kernel)[(x.numel() // e, 1, 1)](
        x.contiguous(), v, i, e, lanes, d, log, descending, x.dtype == torch.float32)
    return v, i


def _decline(x, dim):
    """Why the kernel does not take this call, or None."""
    from pytorchsim.frontend.attention import _fits
    if x.dtype not in _KEYS:
        return f"{x.dtype} is not a key the kernel orders"
    if x.dim() == 0 or x.numel() == 0:
        return "an empty or 0-d input"
    if dim != x.dim() - 1:
        return f"dim {dim} is not the last of {x.dim()}"
    lanes, d, _ = _plan(x)
    if not _fits(_tiles(lanes, d)):
        return f"a row of {x.shape[-1]} overruns the scratchpad budget"
    if torch.is_grad_enabled() and x.requires_grad:
        return "a gradient (the kernel has no backward)"
    return None


def _aten_sort_stable(x, *, stable=None, dim=-1, descending=False):
    dim = dim % max(x.dim(), 1)
    why = _decline(x, dim)
    if why is not None:
        logger.info("[sort] %s %s: the network in aten ops, %s", tuple(x.shape), x.dtype, why)
        return _network(x, dim, descending)
    if not torch.compiler.is_compiling():
        return _network(x, dim, descending)
    return sort(x, descending)


def _aten_sort(x, dim=-1, descending=False):
    return _aten_sort_stable(x, stable=True, dim=dim, descending=descending)


def _network(x, dim, descending):
    """The kernel's network as aten ops along `dim`: (values, int64 indices), equal keys in their order."""
    if x.dim() == 0:
        v, i = _network(x.view(1), 0, descending)
        return v.view(()), i.view(())
    if x.shape[dim] == 0:
        return x.clone(), torch.empty(x.shape, dtype=torch.int64, device=x.device)
    t = x.movedim(dim, -1)
    e = int(t.shape[-1])
    n = _next_pow2(e)
    if t.is_floating_point():
        f = t.float()
        f = torch.where(f == 0, torch.zeros_like(f), f)
        bits = torch.where(f.isnan(), torch.full_like(f, float("nan")), f).view(torch.int32)
        key = (bits ^ ((bits >> 31) & 0x7FFFFFFF)).to(torch.int64)
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


_lib = None


def install():
    """Register aten.sort's npu kernel above autograd and functionalization, and at the backend key."""
    global _lib
    if _lib is None:
        _lib = torch.library.Library("aten", "IMPL")
        for key in ("AutogradPrivateUse1", "PrivateUse1"):
            _lib.impl("sort.stable", _aten_sort_stable, key)
            _lib.impl("sort", _aten_sort, key)
