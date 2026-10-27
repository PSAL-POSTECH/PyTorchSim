"""Mixtral-8x7B on the Triton route: one decoder layer of the published config, random weights, bfloat16 by
default (``--dtype float32`` for the old run).

``--part decode`` (default) runs ``MixtralForCausalLM`` on a ``--seq`` token prompt plus 3 decode steps;
``--part layer`` runs its decoder layer alone (tests/one_layer.py) on ``--seq S`` or one ``--decode S`` step
after a KV cache; the other parts run one piece of the block. Each is judged against the same module's
float32 copy on the CPU (result_check.f32_reference).
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
from result_check import check_result, check_routed_rows, f32, f32_reference
import one_layer

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


# (rtol, atol) per dtype; the bf16 atol is 3x the worst bf16 eager-vs-Inductor CPU diff at S = 128.
_TOL = {"bfloat16": (5e-2, 0.4), "float32": (1e-4, 1e-4)}
_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}
# Share of token rows a bf16 router near-tie may send to other experts (check_routed_rows); fp32 none.
_ROUTED = {"bfloat16": 0.05, "float32": 0.0}


def build_config(layers=1):
    """The published Mixtral-8x7B config with only ``num_hidden_layers`` cut to ``layers``."""
    return MixtralConfig(**{**_MIXTRAL_8X7B, "num_hidden_layers": layers}, attn_implementation="sdpa")


def _check_seq(config, seq):
    if seq > config.max_position_embeddings:
        raise ValueError(f"seq {seq} exceeds max_position_embeddings {config.max_position_embeddings}")


def _causal_mask(seq, dtype):
    """Additive (1, 1, seq, seq) causal mask for a module called without the model's mask builder."""
    return torch.full((seq, seq), float("-inf"), dtype=dtype).triu(1)[None, None]


@torch.no_grad()
def test_layer(device, seq, dtype="bfloat16", decode=False):
    """The decoder layer alone (tests/one_layer.py): a prefill of ``seq`` tokens, or with ``decode`` one step
    at position seq - 1 after a KV cache of the tokens before it."""
    config = build_config()
    _check_seq(config, seq)
    torch.manual_seed(0)
    model = MixtralForCausalLM(config).to(_DTYPES[dtype]).eval()
    ids = torch.randint(0, config.vocab_size, (1, seq), generator=torch.Generator().manual_seed(0))
    cpu_res, res = one_layer.run(model.model, model.model.layers[0], config, ids, device, decode=decode)
    name = f"Mixtral layer ({'decode' if decode else 'prefill'} {seq})"
    check_routed_rows(name, res, cpu_res, *_TOL[dtype], _ROUTED[dtype])


@torch.no_grad()
def test_decode(device, prompt_length, nr_tokens, dtype="bfloat16"):
    """Prompt of ``prompt_length`` tokens, then ``nr_tokens - 1`` decode steps on a DynamicCache per side,
    the CPU side the float32 reference filling its own; each step feeds both the reference logits' argmax."""
    config = build_config()
    _check_seq(config, prompt_length + nr_tokens)
    torch.manual_seed(0)
    cpu_model = MixtralForCausalLM(config).to(_DTYPES[dtype]).eval()
    model = copy.deepcopy(cpu_model).to(device=device)
    cpu_model = f32_reference(cpu_model)
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
        check_routed_rows("Mistral", res, cpu_res, *_TOL[dtype], _ROUTED[dtype])


@torch.no_grad()
def test_attention(device, seq, dtype="bfloat16"):
    """``MixtralAttention`` with the real rotary embedding and a causal mask."""
    config = build_config()
    _check_seq(config, seq)
    model = MixtralAttention(config, layer_idx=0).to(_DTYPES[dtype]).eval()
    rope = MixtralRotaryEmbedding(config)

    prompt = torch.randn([1, seq, config.hidden_size], dtype=_DTYPES[dtype])
    position_ids = torch.arange(seq)[None]
    cos, sin = rope(prompt, position_ids)
    mask = _causal_mask(seq, _DTYPES[dtype])

    cpu_model = f32_reference(model)
    opt_fn = torch.compile(dynamic=False)(model.to(device=device))
    res, _ = opt_fn(prompt.to(device), (cos.to(device), sin.to(device)), mask.to(device))
    cpu_res, _ = cpu_model(*f32((prompt, (cos, sin), mask)))
    check_result("Attention", res, cpu_res, *_TOL[dtype])


@torch.no_grad()
def test_ffn(device, seq, dtype="bfloat16"):
    """The sparse MoE FFN: router, top-2 of 8 experts, weighted combine."""
    config = build_config()
    torch.manual_seed(0)
    model = MixtralSparseMoeBlock(config).eval()
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.02)
    model.to(_DTYPES[dtype])

    prompt = torch.randn([1, seq, config.hidden_size], dtype=_DTYPES[dtype])
    cpu_model = f32_reference(model)
    opt_fn = torch.compile(dynamic=False)(model.to(device=device))
    res = opt_fn(prompt.to(device))
    cpu_res = cpu_model(prompt.float())
    check_routed_rows("FFN", res, cpu_res, *_TOL[dtype], _ROUTED[dtype])


def test_concat(device, seq, dtype="bfloat16"):
    """The KV cache append: (1, 8, seq, 128) with one more position along dim 2."""
    config = build_config()
    size1 = (1, config.num_key_value_heads, seq, config.head_dim)
    size2 = (1, config.num_key_value_heads, 1, config.head_dim)

    def concat_tensors(a, b):
        return torch.cat((a, b), dim=2)

    x = torch.randn(size1, dtype=_DTYPES[dtype])
    y = torch.randn(size2, dtype=_DTYPES[dtype])
    opt_fn = torch.compile(dynamic=False)(concat_tensors)
    res = opt_fn(x.to(device=device), y.to(device=device))
    out = concat_tensors(x, y)
    check_result("ConcatTensors", res, out)


@torch.no_grad()
def test_rmsnorm(device, seq, dtype="bfloat16"):
    """``MixtralRMSNorm`` over the 4096 hidden at the published eps."""
    config = build_config()
    rmsnorm = MixtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(_DTYPES[dtype])

    x = torch.randn([1, seq, config.hidden_size], dtype=_DTYPES[dtype])
    cpu_model = f32_reference(rmsnorm)
    opt_fn = torch.compile(dynamic=False)(rmsnorm.to(device=device))
    res = opt_fn(x.to(device))
    cpu_res = cpu_model(x.float())
    check_result("RMSNorm", res, cpu_res, *_TOL[dtype])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mixtral-8x7B decoder layer on the Triton route")
    parser.add_argument("--part", default="decode",
                        choices=["decode", "layer", "attention", "ffn", "concat", "rmsnorm"])
    one_layer.add_grid_args(parser)
    parser.add_argument("--dtype", default="bfloat16", choices=sorted(_DTYPES))
    args = parser.parse_args()
    if args.decode is not None and args.part != "layer":
        parser.error("--decode runs with --part layer; --part decode takes --seq as its prompt length")

    device = torch.device("npu:0")
    if args.part == "decode":
        test_decode(device, args.seq, 3, args.dtype)
    elif args.part == "layer":
        test_layer(device, args.decode or args.seq, args.dtype, decode=args.decode is not None)
    else:
        globals()[f"test_{args.part}"](device, args.seq, args.dtype)
