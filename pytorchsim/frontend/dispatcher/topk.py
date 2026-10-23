"""topk on the npu: one Triton kernel where it takes the call, the same rounds as aten ops elsewhere.

The kernel puts a row on each lane and runs K rounds of max (min), first index of it, and mask,
gathering the K values and indices of its rows in registers and storing each once, as a TPU
kernel would. Its backward scatters the values' gradient to the indices, aten's own topk backward.
TopK is a CustomFusedKernel (custom_fused_kernel.py), which routes aten::topk's calls.
"""

import torch
import triton
import triton.language as tl
from torch.library import wrap_triton

from pytorchsim import config
from pytorchsim.frontend.dispatcher.custom_fused_kernel import (
    CustomFusedKernel, Decline, fits_scratchpad, next_pow2)


class TopK(CustomFusedKernel):
    name = "torchsim::topk"
    takes = (torch.ops.aten.topk.default,)

    def kernels(self):
        return (_topk_kernel,)

    def arguments(self, op, x, k, dim=-1, largest=True, sorted=True):
        """A non-negative dim; the values are always sorted."""
        return x, int(k), dim % max(x.dim(), 1), bool(largest)

    def plan(self, x, k, dim, largest):
        """A program takes a row on each lane, those past the last masked, halved while its tiles
        overrun the scratchpad budget."""
        if not x.is_floating_point():
            return Decline(f"{x.dtype} is not a floating dtype")
        if x.dim() == 0 or x.numel() == 0:
            return Decline("an empty or 0-d input")
        if dim != x.dim() - 1:
            return Decline(f"dim {dim} is not the last of {x.dim()}")
        E = int(x.shape[-1])
        ep, kp = next_pow2(E), next_pow2(k)
        xblock = int(config.CONFIG_MACHINE.vector_lanes)
        while xblock > 1 and not fits_scratchpad(_tiles(xblock, ep, kp, x.dtype.itemsize)):
            xblock //= 2
        if not fits_scratchpad(_tiles(xblock, ep, kp, x.dtype.itemsize)):
            return Decline(f"a row of {E} overruns the scratchpad budget")
        return {"XBLOCK": xblock, "EP": ep, "KP": kp}

    def launch(self, plan, x: torch.Tensor, k: int, dim: int,
               largest: bool) -> tuple[torch.Tensor, torch.Tensor]:
        E = int(x.shape[-1])
        rows = x.numel() // E
        v = torch.empty(*x.shape[:-1], k, dtype=x.dtype, device=x.device)
        i = torch.empty(*x.shape[:-1], k, dtype=torch.int64, device=x.device)
        wrap_triton(_topk_kernel)[(triton.cdiv(rows, plan["XBLOCK"]), 1, 1)](
            x.contiguous(), v, i, rows, E, plan["EP"], k, plan["KP"], plan["XBLOCK"], largest)
        return v, i

    def fallback(self, x, k, dim, largest):
        return _rounds(x, k, dim, largest)

    def setup_context(self, ctx, inputs, output):
        x, _, dim, _ = inputs
        ctx.dim, ctx.shape = dim, x.shape
        ctx.save_for_backward(output[1])

    def backward(self, ctx, grad_values, grad_indices):
        (indices,) = ctx.saved_tensors
        grad = grad_values.new_zeros(ctx.shape).scatter(ctx.dim, indices, grad_values)
        return grad, None, None, None


TOPK = TopK()


def install():
    TOPK.install()


@triton.jit
def _topk_kernel(x_ptr, v_ptr, i_ptr, rows, E: tl.constexpr, EP: tl.constexpr, K: tl.constexpr,
                 KP: tl.constexpr, XBLOCK: tl.constexpr, LARGEST: tl.constexpr):
    r = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
    c = tl.arange(0, EP)[None, :]
    j = tl.arange(0, KP)[None, :]
    live = r < rows
    if LARGEST:
        spent = float("-inf")
    else:
        spent = float("inf")
    x = tl.load(x_ptr + r * E + c, mask=live & (c < E), other=spent)
    vals = tl.zeros([XBLOCK, KP], dtype=x.dtype)
    idxs = tl.zeros([XBLOCK, KP], dtype=tl.int32)
    for n in tl.static_range(K):
        if LARGEST:
            best = tl.max(x, axis=1)[:, None]
        else:
            best = tl.min(x, axis=1)[:, None]
        at = tl.min(tl.where(x == best, c, EP), axis=1)[:, None]
        vals = tl.where(j == n, best, vals)
        idxs = tl.where(j == n, at, idxs)
        x = tl.where(c == at, spent, x)
    out = live & (j < K)
    tl.store(v_ptr + r * K + j, vals, mask=out)
    tl.store(i_ptr + r * K + j, idxs.to(tl.int64), mask=out)


def _tiles(xblock, ep, kp, esize):
    """The tiles one program of _topk_kernel holds: the row and its masked copy, the index
    candidates, and the gathered values and int64 indices."""
    return [(xblock, ep, esize), (xblock, ep, esize), (xblock, ep, 4), (xblock, kp, esize),
            (xblock, kp, 8)]


def _spent(dtype, largest):
    """The value a taken element is replaced with, so it cannot win again."""
    info = torch.finfo(dtype) if dtype.is_floating_point else torch.iinfo(dtype)
    return info.min if largest else info.max


def _rounds(x, k, dim, largest):
    """The kernel's rounds as aten ops: (values, int64 indices), the first index of each best."""
    if x.dim() == 0:
        v, i = _rounds(x.view(1), k, 0, largest)
        return v.view(x.shape[:0] if k else (0,)), i.view(x.shape[:0] if k else (0,))
    if k == 0:
        shape = [*x.shape[:dim], 0, *x.shape[dim + 1:]]
        return x.new_empty(shape), x.new_empty(shape, dtype=torch.int64)
    n = int(x.shape[dim])
    seat_t = torch.int32 if n < 2**31 else torch.int64
    shape = [1] * x.dim()
    shape[dim] = n
    ar = torch.arange(n, dtype=seat_t, device=x.device).view(shape)
    seat = ar.expand(x.shape)
    late = torch.full(x.shape, n, dtype=seat_t, device=x.device)
    gone = torch.full(x.shape, _spent(x.dtype, largest), dtype=x.dtype, device=x.device)
    work, vals, idxs = x, [], []
    for _ in range(k):
        best = work.amax(dim, keepdim=True) if largest else work.amin(dim, keepdim=True)
        where = torch.where(work == best, seat, late).amin(dim, keepdim=True)
        vals.append(best)
        idxs.append(where.to(torch.int64))
        work = torch.where(ar == where, gone, work)
    return torch.cat(vals, dim), torch.cat(idxs, dim)
