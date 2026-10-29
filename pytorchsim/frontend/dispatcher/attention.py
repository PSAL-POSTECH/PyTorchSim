"""SDPA's overrideable op and torchsim::attention (a causal window and a logit softcap SDPA cannot
state) on the npu: a splash-style prefill or a GQA (or MLA) decode kernel where `fused_plan` takes the call,
SDPA's math elsewhere. `_fused_sdp_choice` sends the aten op only calls with no mask, dropout or gradient."""

import math
from typing import Optional

import torch
import triton
import triton.language as tl
from torch.library import wrap_triton
from triton.language.extra import libdevice

from pytorchsim import config
from pytorchsim.frontend.dispatcher.custom_fused_kernel import (
    CustomFusedKernel, Decline, fits_scratchpad, next_pow2)

ATTENTION = "torchsim::attention"
ATTENTION_SCHEMA = ("attention(Tensor query, Tensor key, Tensor value, float scale, bool is_causal=True, "
                    "int window=0, float softcap=0.0) -> Tensor")


class FusedSDPA(CustomFusedKernel):
    name = "torchsim::sdpa_fused"
    takes = (torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default, ATTENTION)

    def kernels(self):
        return (_prefill_kernel, _decode_kernel)

    def arguments(self, op, query, key, value, *rest, **kwargs):
        """The call as the triton_op's arguments: SDPA's default scale, 1 / sqrt(D), made explicit and
        no debug mask; torchsim::attention's through attention_arguments."""
        if op == ATTENTION:
            return attention_arguments(query, key, value, *rest, **kwargs)
        return _sdpa_arguments(query, key, value, *rest, **kwargs)

    def plan(self, query, key, value, attn_bias, dropout_p, is_causal, scale, window, softcap):
        """A mask or dropout declines (the op can be called directly with them), as does a window over
        a query block that is not causal."""
        if attn_bias is not None or dropout_p:
            return Decline("a mask or dropout")
        if window and not is_causal and query.shape[-2] != 1:
            return Decline(f"a window of {window} on a non-causal prefill")
        return fused_plan(query, key, value, is_causal, window=window, softcap=softcap)

    def launch(self, plan, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
               attn_bias: Optional[torch.Tensor], dropout_p: float, is_causal: bool,
               scale: float, window: int, softcap: float) -> torch.Tensor:
        B, Hq, L, D = query.shape
        H, S, DV = key.shape[1], key.shape[2], value.shape[3]
        if plan["kernel"] is _decode_kernel:
            q, k, v = query.contiguous(), key.contiguous(), value.contiguous()
            o = q.new_empty(B, Hq, L, DV)
            kv0 = (S - window) // plan["BKV"] * plan["BKV"] if window else 0
            wrap_triton(_decode_kernel)[(B * H, 1, 1)](
                q, k, v, o, H, Hq // H, plan["GP"], S, D, plan["DA"], D - plan["DA"], DV, plan["BKV"],
                scale, softcap, window, kv0)
            return o
        q, k, v = (t if t.stride(-1) == 1 else t.contiguous() for t in (query, key, value))
        o = q.new_empty(B, Hq, L, DV)
        wrap_triton(_prefill_kernel)[(B * Hq, 1, 1)](
            q, k, v, o, Hq, Hq // H, S, D, plan["DA"], D - plan["DA"], DV, plan["BQ"], plan["BKV"],
            is_causal, scale, softcap, window, *q.stride()[:3], *k.stride()[:3], *v.stride()[:3],
            *o.stride()[:3])
        return o

    def fallback(self, query, key, value, attn_bias, dropout_p, is_causal, scale, window, softcap):
        if window or softcap:
            return _capped_math(query, key, value, attn_bias, dropout_p, is_causal, scale, window, softcap)
        return _math(query, key, value, attn_bias, dropout_p, is_causal, scale=scale)

    def outputs(self, op, out, query, key, *rest):
        """The aten op's nine outputs from the attention output, the backward-only ones empty;
        torchsim::attention's output alone."""
        if op == ATTENTION:
            return out
        B, Hq, L, D = query.shape
        empty = lambda dtype: torch.empty(0, dtype=dtype, device=query.device)
        lse = torch.empty(B, Hq, L, dtype=torch.float32, device=query.device)
        return (out, lse, empty(torch.int32), empty(torch.int32), L, key.shape[-2],
                empty(torch.int64), empty(torch.int64), empty(query.dtype))


FUSED_SDPA = FusedSDPA()

_lib = None


def install():
    """Define torchsim::attention (its math everywhere but the npu), then register FUSED_SDPA."""
    global _lib
    if _lib is None:
        _lib = torch.library.Library("torchsim", "FRAGMENT")
        _lib.define(ATTENTION_SCHEMA)
        _lib.impl("attention", _attention_math, "CompositeImplicitAutograd")
    FUSED_SDPA.install()


def _sdpa_arguments(query, key, value, attn_bias=None, dropout_p=0.0, is_causal=False,
                    return_debug_mask=False, *, scale=None):
    scale = 1.0 / math.sqrt(query.shape[-1]) if scale is None else scale
    return query, key, value, attn_bias, float(dropout_p), bool(is_causal), float(scale), 0, 0.0


def attention_arguments(query, key, value, scale, is_causal=True, window=0, softcap=0.0):
    """torchsim::attention as the triton_op's arguments: causality is bottom-right (the L queries are
    the last L of the S keys), a key attended when its distance behind the query is under `window`
    (0: any); a causal 1 < L < S query block takes the band as a boolean mask."""
    L, S = query.shape[-2], key.shape[-2]
    window = int(window) if 0 < window < S else 0
    bias = None
    if is_causal and L == 1:
        is_causal = False
    elif is_causal and L != S:
        rows = torch.arange(L, device=query.device)[:, None] + (S - L)
        cols = torch.arange(S, device=query.device)[None, :]
        bias = (cols <= rows) & ((rows - cols < window) if window else True)
        is_causal, window = False, 0
    return query, key, value, bias, 0.0, bool(is_causal), float(scale), window, float(softcap or 0.0)


def _attention_math(query, key, value, scale, is_causal=True, window=0, softcap=0.0):
    """torchsim::attention off the npu: its arguments through the fallback."""
    return FUSED_SDPA.fallback(*attention_arguments(query, key, value, scale, is_causal, window, softcap))


@triton.jit
def _attend(q, qx, m_i, l_i, acc, k_ptr, v_ptr, kbase, vbase, rm, n, KS: tl.constexpr, VS: tl.constexpr,
            DA: tl.constexpr, DB: tl.constexpr, DV: tl.constexpr, SCALE: tl.constexpr, SOFTCAP: tl.constexpr,
            DIAG: tl.constexpr, WINDOW: tl.constexpr, SEEN: tl.constexpr):
    """One KV block's online-softmax step for query rows `rm`, keys `n` (a key KS, a value VS apart): DIAG keeps
    keys at or before the row, WINDOW (0: none) those less than it behind, in one select; SEEN says every row
    already has a key, so a row this block masks whole adds nothing without zeroing p."""
    ra = tl.arange(0, DA)
    rv = tl.arange(0, DV)
    kt = tl.load(k_ptr + kbase + n[None, :] * KS + ra[:, None])
    v = tl.load(v_ptr + vbase + n[:, None] * VS + rv[None, :])
    qk = tl.dot(q, kt, out_dtype=tl.float32)
    if DB > 0:
        rb = tl.arange(0, DB)
        kx = tl.load(k_ptr + kbase + n[None, :] * KS + DA + rb[:, None])
        qk += tl.dot(qx, kx, out_dtype=tl.float32)
    qk = qk * SCALE
    if SOFTCAP > 0:
        qk = libdevice.tanh(qk * (1.0 / SOFTCAP)) * SOFTCAP
    if WINDOW > 0:
        behind = rm[:, None] - n[None, :]
    if DIAG and WINDOW > 0:
        qk = tl.where((behind >= 0) & (behind < WINDOW), qk, -1.0e30)
    elif DIAG:
        qk = tl.where(rm[:, None] >= n[None, :], qk, -1.0e30)
    elif WINDOW > 0:
        qk = tl.where(behind < WINDOW, qk, -1.0e30)
    m_new = tl.maximum(m_i, tl.max(qk, axis=1))
    alpha = tl.exp(m_i - m_new)
    p = tl.exp(qk - m_new[:, None])
    if WINDOW > 0 and not SEEN:
        p = tl.where(behind < WINDOW, p, 0.0)
    l_i = l_i * alpha + tl.sum(p, axis=1)
    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
    return m_new, l_i, acc


@triton.jit
def _prefill_kernel(q_ptr, k_ptr, v_ptr, o_ptr, HQ: tl.constexpr, G: tl.constexpr,
                    S: tl.constexpr, D: tl.constexpr, DA: tl.constexpr, DB: tl.constexpr,
                    DV: tl.constexpr, BQ: tl.constexpr, BKV: tl.constexpr, CAUSAL: tl.constexpr,
                    SCALE: tl.constexpr, SOFTCAP: tl.constexpr, WINDOW: tl.constexpr,
                    QB: tl.constexpr, QH: tl.constexpr, QS: tl.constexpr, KB: tl.constexpr, KH: tl.constexpr,
                    KS: tl.constexpr, VB: tl.constexpr, VH: tl.constexpr, VS: tl.constexpr, OB: tl.constexpr,
                    OH: tl.constexpr, OS: tl.constexpr):
    """One (batch, query head) a program; q, k, v and o each by its batch, head and row strides, a row's last
    dimension contiguous, so a transposed view is read or written where it lies."""
    pid = tl.program_id(0)
    b = pid // HQ
    h = pid % HQ
    qbase = b * QB + h * QH
    kbase = b * KB + (h // G) * KH
    vbase = b * VB + (h // G) * VH
    rn = tl.arange(0, BKV)
    ra = tl.arange(0, DA)
    rv = tl.arange(0, DV)
    for qb in tl.static_range(S // BQ):
        rm = qb * BQ + tl.arange(0, BQ)
        q = tl.load(q_ptr + qbase + rm[:, None] * QS + ra[None, :])
        qx = q
        if DB > 0:
            qx = tl.load(q_ptr + qbase + rm[:, None] * QS + DA + tl.arange(0, DB)[None, :])
        m_i = tl.full((BQ,), -1.0e30, dtype=tl.float32)
        l_i = tl.zeros((BQ,), dtype=tl.float32)
        acc = tl.zeros((BQ, DV), dtype=tl.float32)
        lo = (qb * BQ - WINDOW + 1) // BKV * BKV if WINDOW > 0 else 0
        lo = lo if lo > 0 else 0
        if CAUSAL and WINDOW == 0:
            for start in range(0, qb * BQ, BKV):
                m_i, l_i, acc = _attend(q, qx, m_i, l_i, acc, k_ptr, v_ptr, kbase, vbase, rm, start + rn,
                                        KS, VS, DA, DB, DV, SCALE, SOFTCAP, False, 0, True)
            m_i, l_i, acc = _attend(q, qx, m_i, l_i, acc, k_ptr, v_ptr, kbase, vbase, rm, qb * BQ + rn,
                                    KS, VS, DA, DB, DV, SCALE, SOFTCAP, True, 0, True)
        elif CAUSAL:
            m_i, l_i, acc = _attend(q, qx, m_i, l_i, acc, k_ptr, v_ptr, kbase, vbase, rm, qb * BQ + rn,
                                    KS, VS, DA, DB, DV, SCALE, SOFTCAP, True, WINDOW if WINDOW < BQ else 0, True)
            whole = (qb * BQ + BQ - WINDOW + BKV - 1) // BKV * BKV
            whole = lo if whole < lo else (qb * BQ if whole > qb * BQ else whole)
            for start in range(lo, whole, BKV):
                m_i, l_i, acc = _attend(q, qx, m_i, l_i, acc, k_ptr, v_ptr, kbase, vbase, rm, start + rn,
                                        KS, VS, DA, DB, DV, SCALE, SOFTCAP, False, WINDOW, True)
            for start in range(whole, qb * BQ, BKV):
                m_i, l_i, acc = _attend(q, qx, m_i, l_i, acc, k_ptr, v_ptr, kbase, vbase, rm, start + rn,
                                        KS, VS, DA, DB, DV, SCALE, SOFTCAP, False, 0, True)
        else:
            for start in range(lo, S, BKV):
                m_i, l_i, acc = _attend(q, qx, m_i, l_i, acc, k_ptr, v_ptr, kbase, vbase, rm, start + rn,
                                        KS, VS, DA, DB, DV, SCALE, SOFTCAP, False, WINDOW, False)
        tl.store(o_ptr + b * OB + h * OH + rm[:, None] * OS + rv[None, :],
                 (acc / l_i[:, None]).to(o_ptr.dtype.element_ty))


@triton.jit
def _decode_kernel(q_ptr, k_ptr, v_ptr, o_ptr, HKV: tl.constexpr, G: tl.constexpr,
                   GP: tl.constexpr, S: tl.constexpr, D: tl.constexpr, DA: tl.constexpr,
                   DB: tl.constexpr, DV: tl.constexpr, BKV: tl.constexpr, SCALE: tl.constexpr,
                   SOFTCAP: tl.constexpr, WINDOW: tl.constexpr, KV0: tl.constexpr):
    """q and K rows are D = DA + DB wide (DB 0, or MLA's rope part), V rows DV."""
    pid = tl.program_id(0)
    rg = tl.arange(0, GP)
    ra = tl.arange(0, DA)
    rv = tl.arange(0, DV)
    rows = rg[:, None] < G
    qbase = pid * (G * D)
    kbase = pid * (S * D)
    vbase = pid * (S * DV)
    pos = tl.full((GP,), S - 1, dtype=tl.int32)
    q = tl.load(q_ptr + qbase + rg[:, None] * D + ra[None, :], mask=rows, other=0.0)
    qx = q
    if DB > 0:
        qx = tl.load(q_ptr + qbase + rg[:, None] * D + DA + tl.arange(0, DB)[None, :], mask=rows, other=0.0)
    m_i = tl.full((GP,), -1.0e30, dtype=tl.float32)
    l_i = tl.zeros((GP,), dtype=tl.float32)
    acc = tl.zeros((GP, DV), dtype=tl.float32)
    for start in range(KV0, S, BKV):
        m_i, l_i, acc = _attend(q, qx, m_i, l_i, acc, k_ptr, v_ptr, kbase, vbase, pos,
                                start + tl.arange(0, BKV), D, DV, DA, DB, DV, SCALE, SOFTCAP, False, WINDOW, False)
    tl.store(o_ptr + pid * (G * DV) + rg[:, None] * DV + rv[None, :],
             (acc / l_i[:, None]).to(o_ptr.dtype.element_ty), mask=rows)


def _is_p2(n):
    return n > 0 and n & (n - 1) == 0


_DOT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)

#: The block of the v6e splash kernel the prefill is measured against (block_q = block_kv = 1024);
#: a smaller block pays the matrix unit's fill and drain more often.
SPLASH_BLOCK = 1024

#: The most keys a step of the v6e ragged paged attention kernel the decode is measured against reads
#: (16 pages of 128): a longer step streams past what that kernel sustains.
RPA_BLOCK = 2048


def _register_rows(dtype):
    """Rows of `dtype` in one vector register, the least a dot dimension is padded to."""
    return int(config.CONFIG_MACHINE.vlen_bits) // (8 * dtype.itemsize)


def _prefill_tiles(BQ, BKV, D, esize, causal, DV=None, extra=0):
    """One query block's tiles: q, K^T, V, f32 scores, p, the f32 accumulators, the output; `extra`
    more f32 score tiles (a softcap's, a window's mask)."""
    DV = D if DV is None else DV
    tiles = [(BQ, D, esize), (D, BKV, esize), (BKV, DV, esize), (BQ, BKV, esize), (BQ, DV, esize)]
    tiles += [(BQ, BKV, 4)] * ((5 if causal else 4) + extra) + [(BQ, DV, 4)] * 4
    return tiles


def _decode_tiles(GP, BKV, DK, DV, esize, extra=0):
    """The tiles one KV step of _decode_kernel creates: q, K^T DK wide, the rest as prefill's at DV."""
    return [(GP, DK, esize), (DK, BKV, esize)] + _prefill_tiles(GP, BKV, DV, esize, False, extra=extra)[2:]


def _split_head(D, rows):
    """k's head dimension as DA + DB, DA the largest power of two in it and DB 0 or a power of two,
    each whole vector registers of `rows`; None when it is not."""
    DA = 1 << (D.bit_length() - 1) if D > 0 else 0
    DB = D - DA
    if DA % rows or (DB and (not _is_p2(DB) or DB % rows)):
        return None
    return DA


def fused_plan(query, key, value, is_causal, window=0, softcap=0.0):
    """The fused kernel and its blocks, or a Decline. prefill (L == S): BQ = BKV = min(S,
    SPLASH_BLOCK), halved to fit the scratchpad, k's D one or two powers of two, v's its own; decode
    (L == 1, not causal): heads padded to GP rows, the largest KV step up to RPA_BLOCK that fits."""
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4 or value.shape[:3] != key.shape[:3]:
        return Decline("q, k, v not 4-D, or v's batch, heads or S not k's")
    dtype = query.dtype
    if dtype not in _DOT_DTYPES or key.dtype != dtype or value.dtype != dtype:
        return Decline(f"q {dtype}, k {key.dtype}, v {value.dtype}")
    B, Hq, L, D = query.shape
    H, S, DV = key.shape[1], key.shape[2], value.shape[3]
    if H == 0 or Hq % H or key.shape[0] != B or key.shape[3] != D:
        return Decline(f"{H} KV heads for {Hq} query heads, or k's batch or D not q's")
    rows = _register_rows(dtype)
    DA = _split_head(D, rows)
    if DA is None or not (_is_p2(DV) and _is_p2(S)) or DV % rows or S < rows:
        return Decline(f"S {S}, k's D {D} and v's {DV} not powers of two (k's: two) of at least {rows} rows")
    esize = dtype.itemsize
    extra = (2 if softcap else 0) + (1 if window else 0)
    if L == S:
        floor = min(S, int(config.CONFIG_MACHINE.vector_lanes))
        block = min(S, SPLASH_BLOCK)
        tiles = lambda block: _prefill_tiles(block, block, D, esize, is_causal, DV, extra)
        while block > floor and not fits_scratchpad(tiles(block)):
            block //= 2
        if block % rows or not fits_scratchpad(tiles(block)):
            return Decline(f"a block of {block} overruns the scratchpad budget")
        return {"kernel": _prefill_kernel, "BQ": block, "BKV": block, "DA": DA}
    if L == 1 and not is_causal:
        gp = max(rows, next_pow2(Hq // H))
        bkv = min(S, RPA_BLOCK)
        while bkv >= rows and not fits_scratchpad(_decode_tiles(gp, bkv, D, DV, esize, extra)):
            bkv //= 2
        if bkv < rows:
            return Decline(f"a KV step of {rows} overruns the scratchpad budget")
        return {"kernel": _decode_kernel, "GP": gp, "BKV": bkv, "DA": DA}
    return Decline(f"L {L} is neither S {S} nor a non-causal 1")


def _math(query, key, value, attn_bias=None, dropout_p=0.0, is_causal=False, *, scale=None):
    """SDPA by its math backend's op: masks, dropout, top-left is_causal; differentiable."""
    return torch.ops.aten._scaled_dot_product_attention_math(
        query, key, value, attn_bias, dropout_p, is_causal, None, scale=scale,
        enable_gqa=query.size(-3) != key.size(-3))[0]


def _capped_math(query, key, value, attn_bias, dropout_p, is_causal, scale, window, softcap):
    """attention as aten ops in float32: scores softcapped, a top-left causal mask, the window behind each
    query (the L queries the last L of the S keys), then the mask; differentiable."""
    L, S = query.shape[-2], key.shape[-2]
    group = query.shape[-3] // key.shape[-3]
    k, v = (t.float().repeat_interleave(group, dim=-3) if group > 1 else t.float() for t in (key, value))
    s = query.float() @ k.transpose(-2, -1) * scale
    if softcap:
        s = torch.tanh(s / softcap) * softcap
    rows = torch.arange(L, device=query.device)[:, None]
    cols = torch.arange(S, device=query.device)[None, :]
    if is_causal:
        s = s.masked_fill(cols > rows, float("-inf"))
    if window:
        s = s.masked_fill(rows + (S - L) - cols >= window, float("-inf"))
    if attn_bias is not None:
        s = s.masked_fill(~attn_bias, float("-inf")) if attn_bias.dtype == torch.bool else s + attn_bias
    p = torch.softmax(s, dim=-1)
    if dropout_p:
        p = torch.nn.functional.dropout(p, dropout_p)
    return (p @ v).to(query.dtype)
