"""SDPA's overrideable backend on the npu: the fused Triton kernels (a splash-style prefill, a GQA
decode) where they take the call, SDPA's own math everywhere else.

The device's `_fused_sdp_choice` sends a call to `aten._scaled_dot_product_fused_attention_overrideable`
when that op can carry it (no mask, no dropout, no gradient, H dividing Hq). Its npu kernel here is
the one place that decides whether a fused kernel takes the call (`fused_plan`); everything it
declines runs `aten._scaled_dot_product_attention_math`, the op SDPA's math backend runs.

The route is registered through torch.library: the op's npu kernel is registered at
AutogradPrivateUse1, so it runs above functionalization when torch.compile traces (and at
PrivateUse1 too, which is where a call lands under inference_mode, which skips autograd), and calls
`torchsim::sdpa_fused`, a `triton_op` whose body launches the kernel through `wrap_triton`.
Inductor traces into that body and gets a user-defined Triton kernel, which the npu wrapper
compiles with torchsim_compile. Eagerly the same op computes by math, since a Triton kernel
cannot launch on the npu outside a compiled graph. A call fused_plan declines is logged with its
shape. ROUTED counts the calls that ran each route: a fused kernel's launches, and the math calls
made outside a compiled graph (math compiled into one is Inductor's own kernels, not counted).
"""

import math

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from pytorchsim import config

logger = config.setup_logger()

#: The calls that ran each route, by kernel name or "math"; a trace is not a call.
ROUTED = {}


def _count(route):
    ROUTED[route] = ROUTED.get(route, 0) + 1


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


_DOT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)

#: The query and key block of the splash kernel the prefill is measured against: the v6e
#: runs of PyTorchSim-Triton-Backend's validation archive name it
#: splash_mha_fwd_..._block_kv_compute_1024_..._block_q_1024. A smaller block runs the same
#: rows as more, shorter steps, each paying the matrix unit's fill and drain again.
SPLASH_BLOCK = 1024


def _register_rows(dtype):
    """Rows of `dtype` in one vector register, the least a dot dimension may be here. PTB refuses
    only a last register under MIN_VEC = 2 words; padding dots to whole registers is our choice
    (measured not slower), not the compiler's rule."""
    return int(config.CONFIG_MACHINE.vlen_bits) // (8 * dtype.itemsize)


def _lane_bytes(tiles):
    """An upper bound on the bytes per lane the [rows, cols] tiles take: each in its own buffer (no
    reuse), banked across whichever axis leaves the most per lane. The compiler picks the axis
    (a decode's [GP, D] accumulator lands on its GP rows, its [GP, BKV] scores on BKV), so the
    bound takes the shorter one, at most the lane count."""
    lanes = int(config.CONFIG_MACHINE.vector_lanes)
    return sum(rows * cols * nbytes // min(rows, cols, lanes) for rows, cols, nbytes in tiles)


def _prefill_tiles(BQ, BKV, D, esize, causal):
    """The tiles one query block of _prefill_kernel creates: q, K^T, V, the f32 scores through exp,
    p in the input dtype, the f32 accumulator chain and the stored block."""
    tiles = [(BQ, D, esize), (D, BKV, esize), (BKV, D, esize), (BQ, BKV, esize), (BQ, D, esize)]
    tiles += [(BQ, BKV, 4)] * (5 if causal else 4) + [(BQ, D, 4)] * 4
    return tiles


def _decode_tiles(GP, BKV, D, esize):
    """The tiles one KV step of _decode_kernel creates (the same chain over GP rows)."""
    return _prefill_tiles(GP, BKV, D, esize, causal=False)


def _fits(tiles):
    """Whether `tiles` fit one lane's scratchpad budget, and each is a tensor Triton builds."""
    from pytorchsim.frontend.inductor.codegen import tiling
    big = tiling.triton_max_numel()
    return (all(r * c <= big for r, c, _ in tiles)
            and _lane_bytes(tiles) <= int(config.CONFIG_MACHINE.spad_budget))


def fused_plan(query, key, value, is_causal):
    """The fused kernel and its blocks that take this unmasked, dropout-free call, or None for
    SDPA's math. The one predicate: _fused_sdp_choice only says which op carries the call.

    prefill  L == S. One program per (batch, query head); BQ = BKV is splash's block, min(S,
             SPLASH_BLOCK), halved while one query block's scratch overruns the budget (the compiler
             gives the static query loop's blocks one set), down to one row per lane, below which
             it declines.
    decode   L == 1, not causal (a causal L == 1 attends key 0 alone, SDPA's top-left mask). One
             program per (batch, KV head), its G query heads padded to GP rows; one KV step of the
             largest power of two that fits, since every further step adds its own DMAs.
    Both need power-of-two S and D, D and the blocks a whole number of vector registers.
    """
    if query.dim() != 4 or key.dim() != 4 or value.shape != key.shape:
        return None
    dtype = query.dtype
    if dtype not in _DOT_DTYPES or key.dtype != dtype or value.dtype != dtype:
        return None
    B, Hq, L, D = query.shape
    H, S = key.shape[1], key.shape[2]
    if H == 0 or Hq % H or key.shape[0] != B or key.shape[3] != D:
        return None
    rows = _register_rows(dtype)
    if not (_is_p2(D) and _is_p2(S)) or D % rows or S < rows:
        return None
    esize = dtype.itemsize
    if L == S:
        floor = min(S, int(config.CONFIG_MACHINE.vector_lanes))
        block = min(S, SPLASH_BLOCK)
        while block > floor and not _fits(_prefill_tiles(block, block, D, esize, is_causal)):
            block //= 2
        if block % rows or not _fits(_prefill_tiles(block, block, D, esize, is_causal)):
            return None
        return {"kernel": "prefill", "BQ": block, "BKV": block}
    if L == 1 and not is_causal:
        gp = max(rows, _p2(Hq // H))
        bkv = S
        while bkv >= rows and not _fits(_decode_tiles(gp, bkv, D, esize)):
            bkv //= 2
        if bkv < rows:
            return None
        return {"kernel": "decode", "GP": gp, "BKV": bkv}
    return None


def _math(query, key, value, attn_bias=None, dropout_p=0.0, is_causal=False, *, scale=None):
    """SDPA by the op its math backend runs: masks, dropout and is_causal (top-left) as SDPA
    defines them, differentiable."""
    return torch.ops.aten._scaled_dot_product_attention_math(
        query, key, value, attn_bias, dropout_p, is_causal, None, scale=scale,
        enable_gqa=query.size(-3) != key.size(-3))[0]


@triton_op("torchsim::sdpa_fused", mutates_args=())
def sdpa_fused(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, is_causal: bool,
               scale: float, kernel: str, block_q: int, block_kv: int,
               rows: int) -> torch.Tensor:
    """One fused kernel's launch, as a compiled graph sees it. The kernel's arguments are passed
    by position: Inductor reads a keyword argument of a wrap_triton launch as a possible backend
    option, and asks for the device's Triton backend options, which the npu has none of."""
    q, k, v = query.contiguous(), key.contiguous(), value.contiguous()
    B, Hq, L, D = q.shape
    H, S = k.shape[1], k.shape[2]
    o = torch.empty_like(q)
    if kernel == "decode":
        wrap_triton(_decode_kernel)[(B * H, 1, 1)](
            q, k, v, o, H, Hq // H, rows, S, D, block_kv, scale)
    else:
        wrap_triton(_prefill_kernel)[(B * Hq, 1, 1)](
            q, k, v, o, Hq, Hq // H, S, D, block_q, block_kv, is_causal, scale)
    return o


@sdpa_fused.register_kernel("npu")
def _sdpa_fused_eager(query, key, value, is_causal, scale, kernel, block_q, block_kv, rows):
    _count("math")
    return _math(query, key, value, is_causal=is_causal, scale=scale)


def _overrideable(query, key, value, attn_bias=None, dropout_p=0.0, is_causal=False,
                  return_debug_mask=False, *, scale=None):
    """aten._scaled_dot_product_fused_attention_overrideable on the npu: the fused kernel where
    `fused_plan` takes the call, else math; the outputs only a backward reads are empty. SDPA
    never sends a mask, dropout or gradient here, but the op can be called directly with them:
    such a call runs math, which states them all (and is differentiable)."""
    B, Hq, L, D = query.shape
    S = key.shape[-2]
    scale = 1.0 / math.sqrt(D) if scale is None else scale
    direct = (attn_bias is not None or dropout_p
              or torch.is_grad_enabled() and any(t.requires_grad for t in (query, key, value)))
    plan = None if direct else fused_plan(query, key, value, is_causal)
    if plan is None:
        logger.info("SDPA on the npu runs math, not a fused kernel: q %s k %s %s causal=%s%s",
                    tuple(query.shape), tuple(key.shape), query.dtype, is_causal,
                    " (mask, dropout or grad)" if direct else "")
        out = _math(query, key, value, attn_bias, dropout_p, is_causal, scale=scale)
        if not torch.compiler.is_compiling():
            _count("math")
    else:
        out = sdpa_fused(query, key, value, is_causal, scale, plan["kernel"],
                         plan.get("BQ", 1), plan["BKV"], plan.get("GP", 1))
    empty = lambda dtype: torch.empty(0, dtype=dtype, device=query.device)
    lse = torch.empty(B, Hq, L, dtype=torch.float32, device=query.device)
    return (out, lse, empty(torch.int32), empty(torch.int32), L, S,
            empty(torch.int64), empty(torch.int64), empty(query.dtype))


_lib = None


def install():
    """Register the overrideable op's npu kernel above autograd and functionalization, and at the
    backend key, where a call made under inference_mode lands. Count each fused kernel launch."""
    global _lib
    if _lib is None:
        from pytorchsim.backend import codecache
        codecache.observe_launches(_prefill_kernel, lambda: _count("prefill"))
        codecache.observe_launches(_decode_kernel, lambda: _count("decode"))
        _lib = torch.library.Library("aten", "IMPL")
        for key in ("AutogradPrivateUse1", "PrivateUse1"):
            _lib.impl("_scaled_dot_product_fused_attention_overrideable", _overrideable, key)
