"""A softcap layer's CPU reference (the model's eager attention, causal only through its mask) is the same
when one_layer drops its plain causal mask as with it, as on the npu. Breaks when a dropped mask leaves eager
attending every key: tiny Gemma2 prefill, window wider than S."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"))

import fused_attention  # noqa: E402
import one_layer  # noqa: E402


def _layer_with_and_without_mask(sliding_window):
    from transformers import Gemma2Config, Gemma2Model
    fused_attention.register()
    cfg = Gemma2Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=2, head_dim=16, sliding_window=sliding_window,
                       attn_logit_softcapping=50.0, final_logit_softcapping=30.0,
                       attn_implementation=fused_attention.NAME)
    torch.manual_seed(0)
    body = Gemma2Model(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 16), generator=torch.Generator().manual_seed(0))
    out = []
    for layer in body.layers:
        args, kwargs = one_layer._run_until(body, layer, False, input_ids=ids, use_cache=False)
        dropped = one_layer._drop_plain_mask(kwargs)
        with torch.no_grad():
            out.append((one_layer._first_tensor(layer(*args, **kwargs)),
                        one_layer._first_tensor(layer(*args, **dropped)), dropped["attention_mask"] is None))
    return out


def test_dropped_causal_mask_keeps_the_reference():
    for kept, dropped, was_dropped in _layer_with_and_without_mask(4096):
        assert was_dropped
        torch.testing.assert_close(dropped, kept)


def test_window_narrower_than_s_keeps_its_mask():
    sliding, full = _layer_with_and_without_mask(8)
    assert not sliding[2] and full[2]
    torch.testing.assert_close(sliding[1], sliding[0])
    torch.testing.assert_close(full[1], full[0])


if __name__ == "__main__":
    test_dropped_causal_mask_keeps_the_reference()
    test_window_narrower_than_s_keeps_its_mask()
    print("PASS")
