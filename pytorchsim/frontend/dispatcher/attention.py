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
        """Dropout declines (the op can be called directly with it), as does a window over a query
        block that is not causal, and a mask whose axes the prefill kernel cannot read."""
        if dropout_p:
            return Decline(f"a dropout of {dropout_p}")
        if window and not is_causal and query.shape[-2] != 1:
            return Decline(f"a window of {window} on a non-causal prefill")
        refused = _refuse_bias(query, key, attn_bias) if attn_bias is not None else None
        if refused is not None:
            return refused
        return fused_plan(query, key, value, is_causal, window=window, softcap=softcap,
                          masked=attn_bias is not None)

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
                q, k, v, o, H, Hq // H, plan["GP"], S, plan["SP"], D, plan["DA"], plan["DB"],
                plan["DV"], DV, plan["BKV"], scale, softcap, window, kv0,
                plan["KEYS"], plan["DW"], plan["VW"])
            return o
        q, k, v = (t if t.stride(-1) == 1 else t.contiguous() for t in (query, key, value))
        o = q.new_empty(B, Hq, L, DV)
        m, kind, mstride = _bias_as_strides(attn_bias, (B, Hq, L, S))
        wrap_triton(_prefill_kernel)[(B * Hq, 1, 1)](
            q, k, v, o, m, Hq, Hq // H, plan["LP"], plan["SP"], plan["DA"], plan["DB"], plan["DV"],
            plan["BQ"], plan["BKV"], is_causal, scale, softcap, window,
            plan["QROWS"], plan["KEYS"], plan["DW"], plan["VW"], kind,
            *q.stride()[:3], *k.stride()[:3], *v.stride()[:3], *o.stride()[:3], *mstride)
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
def _within(i, limit: tl.constexpr):
    """Which of `i` a padded tile really holds: those under `limit`, or all of them where it is 0."""
    return i < limit if limit > 0 else i >= 0


@triton.jit
def _attend(q, qx, m_i, l_i, acc, k_ptr, v_ptr, kbase, vbase, rm, n, KS: tl.constexpr, VS: tl.constexpr,
            DA: tl.constexpr, DB: tl.constexpr, DV: tl.constexpr, SCALE: tl.constexpr, SOFTCAP: tl.constexpr,
            DIAG: tl.constexpr, WINDOW: tl.constexpr, SEEN: tl.constexpr, KEYS: tl.constexpr,
            DW: tl.constexpr, VW: tl.constexpr, QROWS: tl.constexpr, m_ptr=None, mbase=0,
            MQ: tl.constexpr = 0, MK: tl.constexpr = 0, MKIND: tl.constexpr = 0):
    """One KV block's online-softmax step for query rows `rm`, keys `n` (a key KS, a value VS apart): DIAG and
    WINDOW keep the keys at or before the row and less than it behind, MKIND 1/2 those a boolean or additive mask
    at m_ptr allows, KEYS and DW/VW those a padded tile has (0: none); SEEN: every row already has a key."""
    ra = tl.arange(0, DA)
    rv = tl.arange(0, DV)
    kp = k_ptr + kbase + n[None, :] * KS + ra[:, None]
    vp = v_ptr + vbase + n[:, None] * VS + rv[None, :]
    if KEYS > 0 or DW > 0:
        kt = tl.load(kp, mask=_within(n, KEYS)[None, :] & _within(ra, DW)[:, None], other=0.0)
    else:
        kt = tl.load(kp)
    if KEYS > 0 or VW > 0:
        v = tl.load(vp, mask=_within(n, KEYS)[:, None] & _within(rv, VW)[None, :], other=0.0)
    else:
        v = tl.load(vp)
    qk = tl.dot(q, kt, out_dtype=tl.float32)
    if DB > 0:
        rb = tl.arange(0, DB)
        bp = k_ptr + kbase + n[None, :] * KS + DA + rb[:, None]
        if KEYS > 0 or DW > 0:
            kx = tl.load(bp, mask=_within(n, KEYS)[None, :] & _within(DA + rb, DW)[:, None], other=0.0)
        else:
            kx = tl.load(bp)
        qk += tl.dot(qx, kx, out_dtype=tl.float32)
    qk = qk * SCALE
    if SOFTCAP > 0:
        qk = libdevice.tanh(qk * (1.0 / SOFTCAP)) * SOFTCAP
    if MKIND > 0:
        mp = m_ptr + mbase + rm[:, None] * MQ + n[None, :] * MK
        live = _within(rm, QROWS)[:, None] & _within(n, KEYS)[None, :]
    if MKIND == 1:
        keep = tl.load(mp, mask=live, other=1) != 0
        qk = tl.where(keep, qk, -1.0e30)
    elif MKIND == 2:
        bias = tl.load(mp, mask=live, other=0.0).to(tl.float32)
        keep = bias > -1.0e30
        qk = tl.where(keep, qk + bias, -1.0e30)
    if KEYS > 0:
        qk = tl.where(n[None, :] < KEYS, qk, -1.0e30)
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
    if MKIND > 0:
        p = tl.where(keep, p, 0.0)
    l_i = l_i * alpha + tl.sum(p, axis=1)
    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
    return m_new, l_i, acc


@triton.jit
def _prefill_kernel(q_ptr, k_ptr, v_ptr, o_ptr, m_ptr, HQ: tl.constexpr, G: tl.constexpr,
                    LP: tl.constexpr, SP: tl.constexpr, DA: tl.constexpr, DB: tl.constexpr,
                    DV: tl.constexpr, BQ: tl.constexpr, BKV: tl.constexpr, CAUSAL: tl.constexpr,
                    SCALE: tl.constexpr, SOFTCAP: tl.constexpr, WINDOW: tl.constexpr,
                    QROWS: tl.constexpr, KEYS: tl.constexpr, DW: tl.constexpr, VW: tl.constexpr,
                    MKIND: tl.constexpr,
                    QB: tl.constexpr, QH: tl.constexpr, QS: tl.constexpr, KB: tl.constexpr, KH: tl.constexpr,
                    KS: tl.constexpr, VB: tl.constexpr, VH: tl.constexpr, VS: tl.constexpr, OB: tl.constexpr,
                    OH: tl.constexpr, OS: tl.constexpr, MB: tl.constexpr = 0, MH: tl.constexpr = 0,
                    MQ: tl.constexpr = 0, MK: tl.constexpr = 0):
    """One (batch, query head) a program, LP query rows over SP keys; q, k, v, o and the MKIND (0: none) mask at
    m_ptr each by its batch, head and row strides, a row's last dimension contiguous, so a transposed view is
    read where it lies. QROWS, KEYS, DW and VW (0: none) are what the padding is masked back to."""
    pid = tl.program_id(0)
    b = pid // HQ
    h = pid % HQ
    qbase = b * QB + h * QH
    kbase = b * KB + (h // G) * KH
    vbase = b * VB + (h // G) * VH
    mbase = b * MB + h * MH
    rn = tl.arange(0, BKV)
    ra = tl.arange(0, DA)
    rv = tl.arange(0, DV)
    for qb in tl.static_range(LP // BQ):
        rm = qb * BQ + tl.arange(0, BQ)
        qp = q_ptr + qbase + rm[:, None] * QS + ra[None, :]
        if QROWS > 0 or DW > 0:
            q = tl.load(qp, mask=_within(rm, QROWS)[:, None] & _within(ra, DW)[None, :], other=0.0)
        else:
            q = tl.load(qp)
        qx = q
        if DB > 0:
            rb = tl.arange(0, DB)
            bp = q_ptr + qbase + rm[:, None] * QS + DA + rb[None, :]
            if QROWS > 0 or DW > 0:
                qx = tl.load(bp, mask=_within(rm, QROWS)[:, None] & _within(DA + rb, DW)[None, :], other=0.0)
            else:
                qx = tl.load(bp)
        m_i = tl.full((BQ,), -1.0e30, dtype=tl.float32)
        l_i = tl.zeros((BQ,), dtype=tl.float32)
        acc = tl.zeros((BQ, DV), dtype=tl.float32)
        lo = (qb * BQ - WINDOW + 1) // BKV * BKV if WINDOW > 0 else 0
        lo = lo if lo > 0 else 0
        if CAUSAL and WINDOW == 0:
            for start in range(0, qb * BQ, BKV):
                m_i, l_i, acc = _attend(q, qx, m_i, l_i, acc, k_ptr, v_ptr, kbase, vbase, rm, start + rn,
                                        KS, VS, DA, DB, DV, SCALE, SOFTCAP, False, 0, True, KEYS, DW, VW, QROWS, m_ptr, mbase, MQ, MK, MKIND)
            m_i, l_i, acc = _attend(q, qx, m_i, l_i, acc, k_ptr, v_ptr, kbase, vbase, rm, qb * BQ + rn,
                                    KS, VS, DA, DB, DV, SCALE, SOFTCAP, True, 0, True, KEYS, DW, VW, QROWS, m_ptr, mbase, MQ, MK, MKIND)
        elif CAUSAL:
            m_i, l_i, acc = _attend(q, qx, m_i, l_i, acc, k_ptr, v_ptr, kbase, vbase, rm, qb * BQ + rn,
                                    KS, VS, DA, DB, DV, SCALE, SOFTCAP, True, WINDOW if WINDOW < BQ else 0,
                                    True, KEYS, DW, VW, QROWS, m_ptr, mbase, MQ, MK, MKIND)
            whole = (qb * BQ + BQ - WINDOW + BKV - 1) // BKV * BKV
            whole = lo if whole < lo else (qb * BQ if whole > qb * BQ else whole)
            for start in range(lo, whole, BKV):
                m_i, l_i, acc = _attend(q, qx, m_i, l_i, acc, k_ptr, v_ptr, kbase, vbase, rm, start + rn,
                                        KS, VS, DA, DB, DV, SCALE, SOFTCAP, False, WINDOW, True,
                                        KEYS, DW, VW, QROWS, m_ptr, mbase, MQ, MK, MKIND)
            for start in range(whole, qb * BQ, BKV):
                m_i, l_i, acc = _attend(q, qx, m_i, l_i, acc, k_ptr, v_ptr, kbase, vbase, rm, start + rn,
                                        KS, VS, DA, DB, DV, SCALE, SOFTCAP, False, 0, True, KEYS, DW, VW, QROWS, m_ptr, mbase, MQ, MK, MKIND)
        else:
            for start in range(lo, SP, BKV):
                m_i, l_i, acc = _attend(q, qx, m_i, l_i, acc, k_ptr, v_ptr, kbase, vbase, rm, start + rn,
                                        KS, VS, DA, DB, DV, SCALE, SOFTCAP, False, WINDOW, False,
                                        KEYS, DW, VW, QROWS, m_ptr, mbase, MQ, MK, MKIND)
        op = o_ptr + b * OB + h * OH + rm[:, None] * OS + rv[None, :]
        out = (acc / l_i[:, None]).to(o_ptr.dtype.element_ty)
        if QROWS > 0 or VW > 0:
            tl.store(op, out, mask=_within(rm, QROWS)[:, None] & _within(rv, VW)[None, :])
        else:
            tl.store(op, out)


@triton.jit
def _decode_kernel(q_ptr, k_ptr, v_ptr, o_ptr, HKV: tl.constexpr, G: tl.constexpr,
                   GP: tl.constexpr, S: tl.constexpr, SP: tl.constexpr, D: tl.constexpr,
                   DA: tl.constexpr, DB: tl.constexpr, DV: tl.constexpr, DVT: tl.constexpr,
                   BKV: tl.constexpr, SCALE: tl.constexpr, SOFTCAP: tl.constexpr,
                   WINDOW: tl.constexpr, KV0: tl.constexpr, KEYS: tl.constexpr, DW: tl.constexpr,
                   VW: tl.constexpr):
    """q and K rows are D wide and covered by DA + DB (DB 0, or MLA's rope part), V rows DVT covered by DV.
    The KV steps run to SP, and KEYS, DW, VW (0: none) are what that padding is masked back to."""
    pid = tl.program_id(0)
    rg = tl.arange(0, GP)
    ra = tl.arange(0, DA)
    rv = tl.arange(0, DV)
    rows = rg[:, None] < G
    qbase = pid * (G * D)
    kbase = pid * (S * D)
    vbase = pid * (S * DVT)
    pos = tl.full((GP,), S - 1, dtype=tl.int32)
    qp = q_ptr + qbase + rg[:, None] * D + ra[None, :]
    if DW > 0:
        q = tl.load(qp, mask=rows & (ra < DW)[None, :], other=0.0)
    else:
        q = tl.load(qp, mask=rows, other=0.0)
    qx = q
    if DB > 0:
        rb = tl.arange(0, DB)
        bp = q_ptr + qbase + rg[:, None] * D + DA + rb[None, :]
        if DW > 0:
            qx = tl.load(bp, mask=rows & (DA + rb < DW)[None, :], other=0.0)
        else:
            qx = tl.load(bp, mask=rows, other=0.0)
    m_i = tl.full((GP,), -1.0e30, dtype=tl.float32)
    l_i = tl.zeros((GP,), dtype=tl.float32)
    acc = tl.zeros((GP, DV), dtype=tl.float32)
    for start in range(KV0, SP, BKV):
        m_i, l_i, acc = _attend(q, qx, m_i, l_i, acc, k_ptr, v_ptr, kbase, vbase, pos,
                                start + tl.arange(0, BKV), D, DVT, DA, DB, DV, SCALE, SOFTCAP,
                                False, WINDOW, False, KEYS, DW, VW, 0, None, 0, 0, 0, 0)
    op = o_ptr + pid * (G * DVT) + rg[:, None] * DVT + rv[None, :]
    out = (acc / l_i[:, None]).to(o_ptr.dtype.element_ty)
    if VW > 0:
        tl.store(op, out, mask=rows & (rv < VW)[None, :])
    else:
        tl.store(op, out, mask=rows)


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
    more f32 score tiles (a softcap's, a window's mask, a padded shape's)."""
    DV = D if DV is None else DV
    tiles = [(BQ, D, esize), (D, BKV, esize), (BKV, DV, esize), (BQ, BKV, esize), (BQ, DV, esize)]
    tiles += [(BQ, BKV, 4)] * ((5 if causal else 4) + extra) + [(BQ, DV, 4)] * 4
    return tiles


def _decode_tiles(GP, BKV, DK, DV, esize, extra=0):
    """The tiles one KV step of _decode_kernel creates: q, K^T DK wide, the rest as prefill's at DV."""
    return [(GP, DK, esize), (DK, BKV, esize)] + _prefill_tiles(GP, BKV, DV, esize, False, extra=extra)[2:]


def _enumerate_head_splits(D, rows):
    """Every (DA, DB) a dot can take a head of `D` as: powers of two of whole `rows`-row registers, DB 0
    or one more, together covering D -- what they cover past it a zero-padded q contributes nothing for."""
    top = max(rows, next_pow2(D))
    widths = [w for w in (rows << i for i in range(top.bit_length())) if w <= top]
    return [(da, db) for da in widths for db in (0, *widths) if da + db >= D]


def _choose_head_split(D, rows):
    """k's head dimension as DA + DB: the narrowest cover, one dot before two, the wider DA first."""
    return min(_enumerate_head_splits(D, rows), key=lambda s: (s[0] + s[1], 1 if s[1] else 0, -s[0]))


def _choose_prefill_blocks(L, S, DP, DVP, esize, causal, extra, rows):
    """BQ over L and BKV over S, each a power of two from min(SPLASH_BLOCK, what covers it) down to one
    row a lane, halved together until one query block's tiles fit the scratchpad; None when none does."""
    lanes = int(config.CONFIG_MACHINE.vector_lanes)
    qcap, kcap = (min(SPLASH_BLOCK, max(rows, next_pow2(n))) for n in (L, S))
    qfloor, kfloor = min(qcap, lanes), min(kcap, lanes)
    for t in range(max(qcap // qfloor, kcap // kfloor).bit_length()):
        bq, bkv = max(qfloor, qcap >> t), max(kfloor, kcap >> t)
        if fits_scratchpad(_prefill_tiles(bq, bkv, DP, esize, causal, DVP, extra)):
            return bq, bkv
    return None


def _choose_decode_step(GP, S, DP, DVP, esize, extra, rows):
    """The KV step: a power of two up to RPA_BLOCK, halved until one step's tiles fit the scratchpad."""
    kcap = min(RPA_BLOCK, max(rows, next_pow2(S)))
    for t in range((kcap // rows).bit_length()):
        bkv = max(rows, kcap >> t)
        if fits_scratchpad(_decode_tiles(GP, bkv, DP, DVP, esize, extra)):
            return bkv
    return None


def _padding_masks(L, LP, S, SP, D, DP, DV, DVP):
    """What a kernel masks the padding it was handed back to: each true extent where its tile is
    wider, 0 where it is not -- and a 0 is what tells the kernel to read that axis unmasked."""
    QROWS, KEYS, DW, VW = (n if n != p else 0
                           for n, p in ((L, LP), (S, SP), (D, DP), (DV, DVP)))
    return {"QROWS": QROWS, "KEYS": KEYS, "DW": DW, "VW": VW}


def _refuse_bias(query, key, bias):
    """A Decline for a mask the prefill kernel cannot read -- a dtype no dot takes, or axes that are
    not [B, Hq, L, S] with every broadcast one a 1 -- and None for a mask it can."""
    want = (query.shape[0], query.shape[1], query.shape[2], key.shape[2])
    if bias.dtype != torch.bool and bias.dtype not in _DOT_DTYPES:
        return Decline(f"a mask of {bias.dtype}")
    if bias.dim() > 4:
        return Decline(f"a {bias.dim()}-D mask")
    got = (1,) * (4 - bias.dim()) + tuple(bias.shape)
    if any(n not in (1, w) for n, w in zip(got, want)):
        return Decline(f"a mask of {got}, which is not {want} with its broadcast axes 1")
    return None


def _bias_as_strides(bias, want):
    """(the mask, its MKIND, its batch/head/row/key strides) for the prefill kernel: a broadcast axis
    carries stride 0, so one row or one head is read for all of them; (None, 0, zeros) for no mask."""
    if bias is None:
        return None, 0, (0, 0, 0, 0)
    m = bias.contiguous()
    shape = (1,) * (4 - m.dim()) + tuple(m.shape)
    strides = (0,) * (4 - m.dim()) + tuple(m.stride())
    return (m, 1 if m.dtype == torch.bool else 2,
            tuple(0 if n == 1 and w != 1 else t for n, t, w in zip(shape, strides, want)))


def fused_plan(query, key, value, is_causal, window=0, softcap=0.0, masked=False):
    """The fused kernel and its blocks, or a Decline. Query rows, keys and both head widths are padded to
    the blocks a dot takes and masked back, so a shape is declined for what the kernel cannot state (a
    top-left causal L != S, a mask on a decode, a scratchpad it overruns), not for its size."""
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4 or value.shape[:3] != key.shape[:3]:
        return Decline("q, k, v not 4-D, or v's batch, heads or S not k's")
    dtype = query.dtype
    if dtype not in _DOT_DTYPES or key.dtype != dtype or value.dtype != dtype:
        return Decline(f"q {dtype}, k {key.dtype}, v {value.dtype}")
    B, Hq, L, D = query.shape
    H, S, DV = key.shape[1], key.shape[2], value.shape[3]
    if H == 0 or Hq % H or key.shape[0] != B or key.shape[3] != D:
        return Decline(f"{H} KV heads for {Hq} query heads, or k's batch or D not q's")
    if min(B, Hq, L, S, D, DV) < 1:
        return Decline(f"an empty B {B}, Hq {Hq}, L {L}, S {S}, k's D {D} or v's {DV}")
    if is_causal and L != S:
        return Decline(f"a top-left causal L {L} that is not S {S}")
    rows = _register_rows(dtype)
    DA, DB = _choose_head_split(D, rows)
    DVP = max(rows, next_pow2(DV))
    esize = dtype.itemsize
    exact = DA + DB == D and DVP == DV and _is_p2(S) and S >= rows
    extra = (2 if softcap else 0) + (1 if window else 0) + (2 if masked else 0)
    if L == 1 and S != 1:
        if masked:
            return Decline(f"a mask on an L 1 decode over S {S}")
        GP = max(rows, next_pow2(Hq // H))
        BKV = _choose_decode_step(GP, S, DA + DB, DVP, esize, extra + (0 if exact else 2), rows)
        if BKV is None:
            return Decline(f"a KV step of {rows} overruns the scratchpad budget")
        SP = -(-S // BKV) * BKV
        return {"kernel": _decode_kernel, "GP": GP, "BKV": BKV, "DA": DA, "DB": DB, "DV": DVP,
                "SP": SP, **_padding_masks(1, 1, S, SP, D, DA + DB, DV, DVP)}
    extra += 0 if exact and _is_p2(L) and L >= rows else 2
    blocks = _choose_prefill_blocks(L, S, DA + DB, DVP, esize, is_causal, extra, rows)
    if blocks is None:
        return Decline(f"no query block of L {L} over S {S} fits the scratchpad budget")
    BQ, BKV = blocks
    LP, SP = -(-L // BQ) * BQ, -(-S // BKV) * BKV
    return {"kernel": _prefill_kernel, "BQ": BQ, "BKV": BKV, "DA": DA, "DB": DB, "DV": DVP,
            "LP": LP, "SP": SP, **_padding_masks(L, LP, S, SP, D, DA + DB, DV, DVP)}


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
