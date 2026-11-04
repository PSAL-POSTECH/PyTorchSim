"""Validation only (TORCHSIM_VALIDATION_KVCACHE_LAYOUT=paged): the decode read as v6e's ragged paged attention
reads it -- one program a sequence walking its KV in steps, every KV head in it, K and V read in place from the
harness's tokenwise pages by their strides, not made contiguous per head. Prefill and every value as FusedSDPA."""

from typing import Optional

import torch
import triton
import triton.language as tl
from torch.library import wrap_triton

from pytorchsim.frontend.dispatcher.attention import FusedSDPA, _attend, _decode_kernel


class PagedSDPA(FusedSDPA):
    def kernels(self):
        return super().kernels() + (_paged_decode_kernel,)

    def launch(self, plan, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
               attn_bias: Optional[torch.Tensor], dropout_p: float, is_causal: bool,
               scale: float, window: int, softcap: float) -> torch.Tensor:
        if plan["kernel"] is not _decode_kernel:
            return super().launch(plan, query, key, value, attn_bias, dropout_p, is_causal, scale, window, softcap)
        B, Hq, L, D = query.shape
        H, S, DV = key.shape[1], key.shape[2], value.shape[3]
        k, v = (t if t.stride(-1) == 1 else t.contiguous() for t in (key, value))
        q = query.contiguous()
        o = q.new_empty(B, Hq, L, DV)
        kv0 = (S - window) // plan["BKV"] * plan["BKV"] if window else 0
        wrap_triton(_paged_decode_kernel)[(B, 1, 1)](
            q, k, v, o, H, Hq // H, plan["GP"], S, D, plan["DA"], D - plan["DA"], DV, plan["BKV"],
            scale, softcap, window, kv0, plan["KEYS"], plan["DW"], plan["VW"],
            *k.stride()[:3], *v.stride()[:3])
        return o


PAGED_SDPA = PagedSDPA()


@triton.jit
def _paged_decode_kernel(q_ptr, k_ptr, v_ptr, o_ptr, HKV: tl.constexpr, G: tl.constexpr,
                         GP: tl.constexpr, S: tl.constexpr, D: tl.constexpr, DA: tl.constexpr,
                         DB: tl.constexpr, DV: tl.constexpr, BKV: tl.constexpr, SCALE: tl.constexpr,
                         SOFTCAP: tl.constexpr, WINDOW: tl.constexpr, KV0: tl.constexpr,
                         KEYS: tl.constexpr, DW: tl.constexpr, VW: tl.constexpr,
                         KB: tl.constexpr, KH: tl.constexpr, KS: tl.constexpr,
                         VB: tl.constexpr, VH: tl.constexpr, VS: tl.constexpr):
    """Sequence pid: each KV head's G query heads over its keys, K and V at (batch, head, token) strides.
    KEYS, DW and VW (0: none) are what the padding is masked back to, as _decode_kernel reads them."""
    b = tl.program_id(0)
    rg = tl.arange(0, GP)
    ra = tl.arange(0, DA)
    rv = tl.arange(0, DV)
    rows = rg[:, None] < G
    pos = tl.full((GP,), S - 1, dtype=tl.int32)
    for h in tl.static_range(HKV):
        qbase = (b * HKV + h) * (G * D)
        q = tl.load(q_ptr + qbase + rg[:, None] * D + ra[None, :], mask=rows, other=0.0)
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
        for start in range(KV0, S, BKV):
            m_i, l_i, acc = _attend(q, qx, m_i, l_i, acc, k_ptr, v_ptr, b * KB + h * KH, b * VB + h * VH, pos,
                                    start + tl.arange(0, BKV), KS, VS, DA, DB, DV, SCALE, SOFTCAP, False, WINDOW,
                                    False, KEYS, DW, VW, 0, None, 0, 0, 0, 0)
        op = o_ptr + (b * HKV + h) * (G * DV) + rg[:, None] * DV + rv[None, :]
        out = (acc / l_i[:, None]).to(o_ptr.dtype.element_ty)
        if VW > 0:
            tl.store(op, out, mask=rows & (rv < VW)[None, :])
        else:
            tl.store(op, out, mask=rows)
