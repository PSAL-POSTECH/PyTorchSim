"""SDPA's overrideable op on the npu: a splash-style prefill or a GQA decode kernel where
`fused_plan` takes the call, SDPA's math elsewhere. `_fused_sdp_choice` sends this op only calls
with no mask, dropout or gradient."""

import math
from typing import Optional

import torch
import triton
import triton.language as tl
from torch.library import wrap_triton

from pytorchsim import config
from pytorchsim.frontend.dispatcher.custom_fused_kernel import (
    CustomFusedKernel, Decline, fits_scratchpad, next_pow2)


class FusedSDPA(CustomFusedKernel):
    name = "torchsim::sdpa_fused"
    takes = (torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default,)

    def kernels(self):
        return (_prefill_kernel, _decode_kernel)

    def arguments(self, op, query, key, value, attn_bias=None, dropout_p=0.0, is_causal=False,
                  return_debug_mask=False, *, scale=None):
        """SDPA's default scale, 1 / sqrt(D), made explicit; no debug mask is returned."""
        scale = 1.0 / math.sqrt(query.shape[-1]) if scale is None else scale
        return query, key, value, attn_bias, float(dropout_p), bool(is_causal), float(scale)

    def plan(self, query, key, value, attn_bias, dropout_p, is_causal, scale):
        """SDPA never sends a mask or dropout here, but the op can be called directly with them."""
        if attn_bias is not None or dropout_p:
            return Decline("a mask or dropout")
        return fused_plan(query, key, value, is_causal)

    def launch(self, plan, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
               attn_bias: Optional[torch.Tensor], dropout_p: float, is_causal: bool,
               scale: float) -> torch.Tensor:
        q, k, v = query.contiguous(), key.contiguous(), value.contiguous()
        B, Hq, L, D = q.shape
        H, S = k.shape[1], k.shape[2]
        o = torch.empty_like(q)
        if plan["kernel"] is _decode_kernel:
            wrap_triton(_decode_kernel)[(B * H, 1, 1)](
                q, k, v, o, H, Hq // H, plan["GP"], S, D, plan["BKV"], scale)
        else:
            wrap_triton(_prefill_kernel)[(B * Hq, 1, 1)](
                q, k, v, o, Hq, Hq // H, S, D, plan["BQ"], plan["BKV"], is_causal, scale)
        return o

    def fallback(self, query, key, value, attn_bias, dropout_p, is_causal, scale):
        return _math(query, key, value, attn_bias, dropout_p, is_causal, scale=scale)

    def outputs(self, op, out, query, key, *rest):
        """The op's nine outputs from the attention output; the backward-only ones are empty."""
        B, Hq, L, D = query.shape
        empty = lambda dtype: torch.empty(0, dtype=dtype, device=query.device)
        lse = torch.empty(B, Hq, L, dtype=torch.float32, device=query.device)
        return (out, lse, empty(torch.int32), empty(torch.int32), L, key.shape[-2],
                empty(torch.int64), empty(torch.int64), empty(query.dtype))


FUSED_SDPA = FusedSDPA()


def install():
    FUSED_SDPA.install()


@triton.jit
def _prefill_kernel(q_ptr, k_ptr, v_ptr, o_ptr, HQ: tl.constexpr, G: tl.constexpr,
                    S: tl.constexpr, D: tl.constexpr, BQ: tl.constexpr, BKV: tl.constexpr,
                    CAUSAL: tl.constexpr, SCALE: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // HQ
    h = pid % HQ
    qbase = pid * (S * D)
    kbase = (b * (HQ // G) + h // G) * (S * D)
    rn = tl.arange(0, BKV)
    rd = tl.arange(0, D)
    for qb in tl.static_range(S // BQ):
        rm = qb * BQ + tl.arange(0, BQ)
        q = tl.load(q_ptr + qbase + rm[:, None] * D + rd[None, :])
        m_i = tl.full((BQ,), -1.0e30, dtype=tl.float32)
        l_i = tl.zeros((BQ,), dtype=tl.float32)
        acc = tl.zeros((BQ, D), dtype=tl.float32)
        for start in range(0, qb * BQ if CAUSAL else S, BKV):
            n = start + rn
            kt = tl.load(k_ptr + kbase + n[None, :] * D + rd[:, None])
            v = tl.load(v_ptr + kbase + n[:, None] * D + rd[None, :])
            qk = tl.dot(q, kt, out_dtype=tl.float32) * SCALE
            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
            m_i = m_new
        if CAUSAL:
            n = qb * BQ + rn
            kt = tl.load(k_ptr + kbase + n[None, :] * D + rd[:, None])
            v = tl.load(v_ptr + kbase + n[:, None] * D + rd[None, :])
            qk = tl.dot(q, kt, out_dtype=tl.float32) * SCALE
            qk = tl.where(rm[:, None] >= n[None, :], qk, -1.0e30)
            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
            m_i = m_new
        tl.store(o_ptr + qbase + rm[:, None] * D + rd[None, :],
                 (acc / l_i[:, None]).to(o_ptr.dtype.element_ty))


@triton.jit
def _decode_kernel(q_ptr, k_ptr, v_ptr, o_ptr, HKV: tl.constexpr, G: tl.constexpr,
                   GP: tl.constexpr, S: tl.constexpr, D: tl.constexpr, BKV: tl.constexpr,
                   SCALE: tl.constexpr):
    pid = tl.program_id(0)
    rg = tl.arange(0, GP)
    rd = tl.arange(0, D)
    rows = rg[:, None] < G
    qbase = pid * (G * D)
    kbase = pid * (S * D)
    q = tl.load(q_ptr + qbase + rg[:, None] * D + rd[None, :], mask=rows, other=0.0)
    m_i = tl.full((GP,), -1.0e30, dtype=tl.float32)
    l_i = tl.zeros((GP,), dtype=tl.float32)
    acc = tl.zeros((GP, D), dtype=tl.float32)
    for start in range(0, S, BKV):
        t = start + tl.arange(0, BKV)
        kt = tl.load(k_ptr + kbase + t[None, :] * D + rd[:, None])
        v = tl.load(v_ptr + kbase + t[:, None] * D + rd[None, :])
        qk = tl.dot(q, kt, out_dtype=tl.float32) * SCALE
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
        m_i = m_new
    tl.store(o_ptr + qbase + rg[:, None] * D + rd[None, :],
             (acc / l_i[:, None]).to(o_ptr.dtype.element_ty), mask=rows)


def _is_p2(n):
    return n > 0 and n & (n - 1) == 0


_DOT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)

#: The block of the v6e splash kernel the prefill is measured against (block_q = block_kv = 1024);
#: a smaller block pays the matrix unit's fill and drain more often.
SPLASH_BLOCK = 1024


def _register_rows(dtype):
    """Rows of `dtype` in one vector register, the least a dot dimension is padded to."""
    return int(config.CONFIG_MACHINE.vlen_bits) // (8 * dtype.itemsize)


def _prefill_tiles(BQ, BKV, D, esize, causal):
    """One query block's tiles: q, K^T, V, f32 scores, p, the f32 accumulators, the output."""
    tiles = [(BQ, D, esize), (D, BKV, esize), (BKV, D, esize), (BQ, BKV, esize), (BQ, D, esize)]
    tiles += [(BQ, BKV, 4)] * (5 if causal else 4) + [(BQ, D, 4)] * 4
    return tiles


def _decode_tiles(GP, BKV, D, esize):
    """The tiles one KV step of _decode_kernel creates (the same chain over GP rows)."""
    return _prefill_tiles(GP, BKV, D, esize, causal=False)


def fused_plan(query, key, value, is_causal):
    """The fused kernel and its blocks, or a Decline. prefill (L == S): BQ = BKV = min(S,
    SPLASH_BLOCK), halved to fit the scratchpad; decode (L == 1, not causal): heads padded to GP
    rows, the largest KV step that fits. S and D are powers of two in whole vector registers."""
    if query.dim() != 4 or key.dim() != 4 or value.shape != key.shape:
        return Decline("q, k, v not 4-D, or v not shaped as k")
    dtype = query.dtype
    if dtype not in _DOT_DTYPES or key.dtype != dtype or value.dtype != dtype:
        return Decline(f"q {dtype}, k {key.dtype}, v {value.dtype}")
    B, Hq, L, D = query.shape
    H, S = key.shape[1], key.shape[2]
    if H == 0 or Hq % H or key.shape[0] != B or key.shape[3] != D:
        return Decline(f"{H} KV heads for {Hq} query heads, or k's batch or D not q's")
    rows = _register_rows(dtype)
    if not (_is_p2(D) and _is_p2(S)) or D % rows or S < rows:
        return Decline(f"S {S} and D {D} not powers of two of at least {rows} rows")
    esize = dtype.itemsize
    if L == S:
        floor = min(S, int(config.CONFIG_MACHINE.vector_lanes))
        block = min(S, SPLASH_BLOCK)
        while (block > floor
               and not fits_scratchpad(_prefill_tiles(block, block, D, esize, is_causal))):
            block //= 2
        if block % rows or not fits_scratchpad(_prefill_tiles(block, block, D, esize, is_causal)):
            return Decline(f"a block of {block} overruns the scratchpad budget")
        return {"kernel": _prefill_kernel, "BQ": block, "BKV": block}
    if L == 1 and not is_causal:
        gp = max(rows, next_pow2(Hq // H))
        bkv = S
        while bkv >= rows and not fits_scratchpad(_decode_tiles(gp, bkv, D, esize)):
            bkv //= 2
        if bkv < rows:
            return Decline(f"a KV step of {rows} overruns the scratchpad budget")
        return {"kernel": _decode_kernel, "GP": gp, "BKV": bkv}
    return Decline(f"L {L} is neither S {S} nor a non-causal 1")


def _math(query, key, value, attn_bias=None, dropout_p=0.0, is_causal=False, *, scale=None):
    """SDPA by its math backend's op: masks, dropout, top-left is_causal; differentiable."""
    return torch.ops.aten._scaled_dot_product_attention_math(
        query, key, value, attn_bias, dropout_p, is_causal, None, scale=scale,
        enable_gqa=query.size(-3) != key.size(-3))[0]
