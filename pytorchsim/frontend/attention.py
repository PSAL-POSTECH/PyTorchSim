"""SDPA's overrideable backend on the npu: Inductor's decomposition of the fused op launches the
Triton kernels here (a splash-style prefill, a GQA decode), the rest by math."""

import math

import torch
import triton
import triton.language as tl

MAX_BLOCK = 1024
MIN_ROWS = 4
DECODE_BKV = 4096


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


def _p2(n):
    return 1 << max(0, (int(n) - 1).bit_length())


def _is_p2(n):
    return n > 0 and n & (n - 1) == 0


def kernel_takes(q, k, is_causal):
    """Whether the fused kernels take these shapes: power-of-two lengths and head dim, a prefill
    (L == S) or a decode (L == 1, not causal), H dividing Hq."""
    B, Hq, L, D = q.shape
    H, S = k.shape[1], k.shape[2]
    if Hq % H or not (_is_p2(D) and _is_p2(S)):
        return False
    if L == 1:
        return not is_causal and S >= MIN_ROWS
    return L == S


def _math(q, k, v, is_causal, scale):
    """SDPA by its definition, for the shapes the kernels do not take."""
    Hq, H = q.shape[1], k.shape[1]
    if Hq != H:
        k = k.repeat_interleave(Hq // H, 1)
        v = v.repeat_interleave(Hq // H, 1)
    s = (q.float() @ k.float().transpose(-1, -2)) * scale
    if is_causal:
        L, S = q.shape[-2], k.shape[-2]
        keep = torch.ones(L, S, dtype=torch.bool, device=q.device).tril(S - L)
        s = s.masked_fill(~keep, float("-inf"))
    return (torch.softmax(s, -1) @ v.float()).to(q.dtype)


def _run(kernel, grid, out, **args):
    """Launch a Triton kernel as Inductor's functional kernel node and return the tensor it writes."""
    from torch._higher_order_ops.triton_kernel_wrap import (
        kernel_side_table, triton_kernel_wrapper_functional)
    return triton_kernel_wrapper_functional(
        kernel_idx=kernel_side_table.add_kernel(kernel),
        constant_args_idx=kernel_side_table.add_constant_args({}), grid=[grid],
        tma_descriptor_metadata={}, kwargs=args, tensors_to_clone=[out])[out]


def _launch(q, k, v, is_causal, scale):
    """The fused kernel's launch, as a functional kernel node wherever it is traced."""
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    B, Hq, L, D = q.shape
    H, S = k.shape[1], k.shape[2]
    G = Hq // H
    o = torch.empty_like(q)
    if L == 1:
        return _run(_decode_kernel, (B * H, 1, 1), "o_ptr", q_ptr=q, k_ptr=k, v_ptr=v, o_ptr=o,
                    HKV=H, G=G, GP=max(MIN_ROWS, _p2(G)), S=S, D=D,
                    BKV=min(S, DECODE_BKV), SCALE=scale)
    block = min(S, MAX_BLOCK)
    return _run(_prefill_kernel, (B * Hq, 1, 1), "o_ptr", q_ptr=q, k_ptr=k, v_ptr=v, o_ptr=o,
                HQ=Hq, G=G, S=S, D=D, BQ=block, BKV=block, CAUSAL=is_causal, SCALE=scale)


def _overrideable(query, key, value, attn_bias=None, dropout_p=0.0, is_causal=False,
                  return_debug_mask=False, *, scale=None, fused=True):
    """aten._scaled_dot_product_fused_attention_overrideable by the fused kernels (or math when
    not fused), the rest of its outputs (backward's) empty."""
    B, Hq, L, D = query.shape
    S = key.shape[-2]
    scale = 1.0 / math.sqrt(D) if scale is None else scale
    if fused and attn_bias is None and dropout_p == 0.0 and kernel_takes(query, key, is_causal):
        out = _launch(query, key, value, is_causal, scale)
    else:
        out = _math(query, key, value, is_causal, scale)
    empty = lambda dtype: torch.empty(0, dtype=dtype, device=query.device)
    lse = torch.empty(B, Hq, L, dtype=torch.float32, device=query.device)
    return (out, lse, empty(torch.int32), empty(torch.int32), L, S,
            empty(torch.int64), empty(torch.int64), empty(query.dtype))


_lib = None


def install():
    """Register the decomposition Inductor takes for the overrideable SDPA op, and math eagerly."""
    global _lib
    from torch._inductor.decomposition import decompositions
    op = torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default
    decompositions[op] = _overrideable
    if _lib is None:
        _lib = torch.library.Library("aten", "IMPL")
        _lib.impl("_scaled_dot_product_fused_attention_overrideable",
                  lambda *a, **kw: _overrideable(*a, **kw, fused=False), "PrivateUse1")
