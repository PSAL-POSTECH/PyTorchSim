"""The "torchsim" attention a transformers config can ask for: on the npu torchsim::attention, which takes the
layer's sliding window and logit softcap into the fused kernel as the TPU's splash does (SDPA can state neither);
off it transformers' own sdpa, or the model's eager attention for a softcap, on eager_mask's band.

Imported the way result_check is (tests/ on sys.path): ``import fused_attention``; ``register()`` once.
"""

import sys

import torch

NAME = "torchsim"


def register():
    """Register NAME with transformers' attention and mask interfaces (the mask as eager's float band)."""
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, eager_mask
    AttentionInterface.register(NAME, attention_forward)
    AttentionMaskInterface.register(NAME, eager_mask)


def attention_forward(module, query, key, value, attention_mask, dropout=0.0, scaling=None,
                      sliding_window=None, softcap=None, **kwargs):
    """(output [B, L, Hq, Dv], None). On the npu the mask is read as the causal band of an unpadded batch,
    which `is_causal` and `sliding_window` restate; elsewhere it is applied as transformers applies it."""
    if query.device.type != "npu":
        return _reference(module, query, key, value, attention_mask, dropout, scaling, sliding_window,
                          softcap, **kwargs)
    scale = query.shape[-1] ** -0.5 if scaling is None else scaling
    out = torch.ops.torchsim.attention(query, key, value, float(scale), bool(getattr(module, "is_causal", True)),
                                       int(sliding_window or 0), float(softcap or 0.0))
    return out.transpose(1, 2).contiguous(), None


def _reference(module, query, key, value, attention_mask, dropout, scaling, sliding_window, softcap, **kwargs):
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    if softcap is None:
        return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout,
                                      scaling=scaling, **kwargs)
    eager = sys.modules[type(module).__module__].eager_attention_forward
    return eager(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, softcap=softcap,
                 sliding_window=sliding_window, **kwargs)
