"""Mixtral-8x7B on the Triton route: one decoder layer of the published config, random weights.

``--part decode`` (default) runs ``MixtralForCausalLM`` on a ``--seq`` token prompt plus 3 decode steps;
the other parts run one piece of the block. Each is judged against the same module on CPU.
"""

import argparse
import copy
import os
import sys

import torch
from transformers import DynamicCache, MixtralConfig
from transformers.models.mixtral.modeling_mixtral import (
    MixtralAttention,
    MixtralForCausalLM,
    MixtralRMSNorm,
    MixtralRotaryEmbedding,
    MixtralSparseMoeBlock,
)

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from result_check import check_result

# mistralai/Mixtral-8x7B-v0.1 config.json, published values.
_MIXTRAL_8X7B = dict(
    vocab_size=32000,
    hidden_size=4096,
    intermediate_size=14336,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=8,
    head_dim=128,
    hidden_act="silu",
    max_position_embeddings=32768,
    rms_norm_eps=1e-5,
    rope_theta=1e6,
    sliding_window=None,
    num_local_experts=8,
    num_experts_per_tok=2,
    router_aux_loss_coef=0.02,
    tie_word_embeddings=False,
    bos_token_id=1,
    eos_token_id=2,
    attention_dropout=0.0,
)


def build_config(layers=1):
    """The published Mixtral-8x7B config with only ``num_hidden_layers`` cut to ``layers``."""
    return MixtralConfig(**{**_MIXTRAL_8X7B, "num_hidden_layers": layers}, attn_implementation="sdpa")


def _check_seq(config, seq):
    if seq > config.max_position_embeddings:
        raise ValueError(f"seq {seq} exceeds max_position_embeddings {config.max_position_embeddings}")


def _causal_mask(seq):
    """Additive (1, 1, seq, seq) causal mask for a module called without the model's mask builder."""
    return torch.full((seq, seq), float("-inf")).triu(1)[None, None]


@torch.no_grad()
def test_decode(device, prompt_length, nr_tokens):
    """Prompt of ``prompt_length`` tokens, then ``nr_tokens - 1`` decode steps on a DynamicCache per side.

    Each step feeds both sides the CPU logits' argmax and compares the logits.
    """
    config = build_config()
    _check_seq(config, prompt_length + nr_tokens)
    torch.manual_seed(0)
    cpu_model = MixtralForCausalLM(config).eval()
    model = copy.deepcopy(cpu_model).to(device=device)
    opt_fn = torch.compile(dynamic=False)(model)

    g = torch.Generator().manual_seed(0)
    cpu_ids = torch.randint(0, config.vocab_size, (1, prompt_length), generator=g)
    cpu_cache = DynamicCache(config=config)
    kv_cache = DynamicCache(config=config)
    for _ in range(nr_tokens):
        res = opt_fn(input_ids=cpu_ids.to(device), past_key_values=kv_cache, use_cache=True).logits
        cpu_res = cpu_model(input_ids=cpu_ids, past_key_values=cpu_cache, use_cache=True).logits
        cpu_ids = cpu_res[:, -1:].argmax(dim=-1)
        print(cpu_ids)
        check_result("Mistral", res, cpu_res)


@torch.no_grad()
def test_attention(device, seq):
    """``MixtralAttention`` with the real rotary embedding and a causal mask."""
    config = build_config()
    _check_seq(config, seq)
    model = MixtralAttention(config, layer_idx=0).eval()
    rope = MixtralRotaryEmbedding(config)

    prompt = torch.randn([1, seq, config.hidden_size], dtype=torch.float32)
    position_ids = torch.arange(seq)[None]
    cos, sin = rope(prompt, position_ids)
    mask = _causal_mask(seq)

    cpu_model = copy.deepcopy(model)
    opt_fn = torch.compile(dynamic=False)(model.to(device=device))
    res, _ = opt_fn(prompt.to(device), (cos.to(device), sin.to(device)), mask.to(device))
    cpu_res, _ = cpu_model(prompt, (cos, sin), mask)
    check_result("Attention", res, cpu_res)


@torch.no_grad()
def test_ffn(device, seq):
    """The sparse MoE FFN: router, top-2 of 8 experts, weighted combine."""
    config = build_config()
    torch.manual_seed(0)
    model = MixtralSparseMoeBlock(config).eval()
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.02)

    prompt = torch.randn([1, seq, config.hidden_size], dtype=torch.float32)
    cpu_model = copy.deepcopy(model)
    opt_fn = torch.compile(dynamic=False)(model.to(device=device))
    res = opt_fn(prompt.to(device))
    cpu_res = cpu_model(prompt)
    check_result("FFN", res, cpu_res)


def test_concat(device, seq):
    """The KV cache append: (1, 8, seq, 128) with one more position along dim 2."""
    config = build_config()
    size1 = (1, config.num_key_value_heads, seq, config.head_dim)
    size2 = (1, config.num_key_value_heads, 1, config.head_dim)

    def concat_tensors(a, b):
        return torch.cat((a, b), dim=2)

    x = torch.randn(size1)
    y = torch.randn(size2)
    opt_fn = torch.compile(dynamic=False)(concat_tensors)
    res = opt_fn(x.to(device=device), y.to(device=device))
    out = concat_tensors(x, y)
    check_result("ConcatTensors", res, out)


@torch.no_grad()
def test_rmsnorm(device, seq):
    """``MixtralRMSNorm`` over the 4096 hidden at the published eps."""
    config = build_config()
    rmsnorm = MixtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    x = torch.randn([1, seq, config.hidden_size], dtype=torch.float32)
    cpu_model = copy.deepcopy(rmsnorm)
    opt_fn = torch.compile(dynamic=False)(rmsnorm.to(device=device))
    res = opt_fn(x.to(device))
    cpu_res = cpu_model(x)
    check_result("RMSNorm", res, cpu_res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mixtral-8x7B decoder layer on the Triton route")
    parser.add_argument("--part", default="decode",
                        choices=["decode", "attention", "ffn", "concat", "rmsnorm"])
    parser.add_argument("--seq", type=int, default=128, help="prompt length (decode) or tokens (the rest)")
    args = parser.parse_args()

    device = torch.device("npu:0")
    if args.part == "decode":
        test_decode(device, args.seq, 3)
    else:
        globals()[f"test_{args.part}"](device, args.seq)
