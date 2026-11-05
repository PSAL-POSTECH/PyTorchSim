"""one_layer's decode step writes its own K and V into slot S - 1 of the cache it reads, in place, headwise
and in tokenwise pages: a tiny float32 Qwen3 layer on the CPU against HF's own prefill. Breaks when the
harness hands the layer a prefilled cache and drops the K and V it computed (k_proj then DCE'd)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"))

import one_layer  # noqa: E402

S = 8


def test_decode_writes_its_slot_in_place(layout="headwise"):
    from unittest import mock
    from transformers import DynamicCache, Qwen3Config, Qwen3Model
    from pytorchsim import config
    cfg = Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=1,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=64)
    torch.manual_seed(0)
    body = Qwen3Model(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, S), generator=torch.Generator().manual_seed(0))
    seen = []
    hook = body.layers[0].register_forward_pre_hook(
        lambda m, a, k: seen.append(k.get("past_key_values")), with_kwargs=True)
    try:
        with mock.patch.object(config, "CONFIG_KVCACHE_LAYOUT", layout):
            ref, out = one_layer.run(body, body.layers[0], cfg, ids, torch.device("cpu"), decode=True,
                                     compile_model=False)
    finally:
        hook.remove()
    torch.testing.assert_close(out, ref)
    filled = DynamicCache(config=cfg)
    with torch.no_grad():
        body(input_ids=ids, past_key_values=filled, use_cache=True)
    written = seen[-1].layers[0]
    for name in ("keys", "values"):
        got, want = getattr(written, name), getattr(filled.layers[0], name)
        assert got.shape[2] == S, f"{name}: the step's cache has {got.shape[2]} slots, not {S}"
        torch.testing.assert_close(got, want)


def test_decode_writes_its_tokenwise_page_in_place():
    test_decode_writes_its_slot_in_place("tokenwise")


if __name__ == "__main__":
    test_decode_writes_its_slot_in_place()
    test_decode_writes_its_tokenwise_page_in_place()
    print("PASS")
