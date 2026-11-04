"""Llama 3 at Meta-Llama-3-8B's published sizes, one layer.

Llama 3 is the Llama architecture with three things changed, and each one lands
on a different part of this backend:

  grouped-query attention   8 kv heads against 32 query heads, so k/v are
                            repeated 4x on the head axis before the bmm
  rope_theta 500000         a different inv_freq, same access pattern
  wider SwiGLU              intermediate_size is 3.5x hidden, not 2.7x

GQA is the one that is not just a number: the repeat_kv broadcast has to be
absorbed rather than become a lane-crossing copy. Only the layer count is cut.
Weights and inputs are bfloat16 by default (--dtype float32 for the old run).
--part layer runs LlamaModel's first decoder layer alone (tests/one_layer.py),
on --seq S or on one --decode S step after a KV cache.
"""
import os
import sys
import argparse
import copy
import torch
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaModel
sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from result_check import check_result, f32, f32_reference
import one_layer

# meta-llama/Meta-Llama-3-8B, as published.
LLAMA3_8B = dict(
    vocab_size=128256,
    hidden_size=4096,
    num_attention_heads=32,
    num_key_value_heads=8,
    intermediate_size=14336,
    max_position_embeddings=8192,
    rope_theta=500000.0,
    rms_norm_eps=1e-5,
    hidden_act="silu",
    attention_bias=False,
    mlp_bias=False,
    tie_word_embeddings=False,
    bos_token_id=128000,
    eos_token_id=128001,
)
HEADS, KV_HEADS = LLAMA3_8B["num_attention_heads"], LLAMA3_8B["num_key_value_heads"]

# (rtol, atol) per dtype; the bf16 atol is 3x the worst bf16 eager-vs-Inductor CPU diff at S = 128.
_TOL = {"bfloat16": (5e-2, 0.6), "float32": (1e-3, 1e-3)}


def llama3_config(seq_len, layers=1):
    """Meta-Llama-3-8B with `layers` blocks.

    attn_implementation is named, not left to the default: transformers looks the implementation up by
    this string and None is not a key (KeyError in modeling_llama's forward). It is sdpa, the fused
    kernel the TPU rows are measured on.
    Raises if `seq_len` exceeds the model's 8192 positions.
    """
    if seq_len > LLAMA3_8B["max_position_embeddings"]:
        raise ValueError(f"seq {seq_len} exceeds Llama-3-8B's "
                         f"{LLAMA3_8B['max_position_embeddings']} positions")
    return LlamaConfig(**LLAMA3_8B, num_hidden_layers=layers, use_cache=False,
                       attn_implementation="sdpa")


@torch.no_grad()
def run_decoder_layer_test(device, batch=1, seq_len=128, dtype="float32",
                           rtol=1e-3, atol=1e-3):
    print("\n[Running Llama3 DecoderLayer Test]")
    dtype_map = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(dtype, torch.float32)

    cfg = llama3_config(seq_len)
    print(f"Building Llama3 decoder layer: {HEADS} q heads / {KV_HEADS} kv heads.")
    base_layer = LlamaDecoderLayer(cfg, layer_idx=0).eval()
    cpu_layer = copy.deepcopy(base_layer).eval()

    cpu_layer.to(dtype=torch_dtype, device="cpu")
    layer = base_layer.to(dtype=torch_dtype, device=device)

    g = torch.Generator().manual_seed(0)
    hidden_cpu = torch.randn(batch, seq_len, cfg.hidden_size, generator=g,
                             dtype=torch_dtype)

    min_dtype = torch.finfo(torch_dtype).min
    causal_mask = torch.full((seq_len, seq_len), fill_value=min_dtype,
                             dtype=torch_dtype, device="cpu")
    if seq_len > 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)
    attn_mask_cpu = causal_mask[None, None, :, :].expand(batch, 1, -1, -1)

    position_ids_cpu = torch.arange(seq_len, dtype=torch.long)[None, :].expand(batch, -1)

    # The rotary embedding lives on the model in this transformers version, so
    # the layer needs position_embeddings handed to it directly.
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
    rope_cpu = LlamaRotaryEmbedding(config=cfg).to(dtype=torch_dtype, device="cpu")
    pos_emb_cpu = rope_cpu(hidden_cpu, position_ids_cpu)

    hidden_dev = hidden_cpu.to(device)
    attn_mask_dev = attn_mask_cpu.to(device)
    position_ids_dev = position_ids_cpu.to(device)
    pos_emb_dev = tuple(t.to(device) for t in pos_emb_cpu)

    print("Compiling Llama3 decoder layer with torch.compile(...)")
    compiled_layer = torch.compile(layer, dynamic=False)

    out_cpu = f32_reference(cpu_layer)(hidden_states=hidden_cpu.float(), attention_mask=attn_mask_cpu.float(),
                                       position_ids=position_ids_cpu,
                                       position_embeddings=f32(pos_emb_cpu))[0]
    out_dev = compiled_layer(hidden_states=hidden_dev, attention_mask=attn_mask_dev,
                             position_ids=position_ids_dev,
                             position_embeddings=pos_emb_dev)[0]

    check_result("Llama3 DecoderLayer forward", out_dev, out_cpu, rtol=rtol, atol=atol)
    diff = (out_dev.detach().cpu() - out_cpu.detach().cpu()).abs().max().item()
    print(f"Max diff > {diff}")


@torch.no_grad()
def run_model_test(device, batch=1, seq_len=128, dtype="float32",
                   rtol=1e-3, atol=1e-3):
    print("\n[Running Llama3 Model Test]")
    dtype_map = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(dtype, torch.float32)

    cfg = llama3_config(seq_len)
    print("Building Llama3 model at Llama-3-8B sizes, 1 layer (random init).")
    base_model = LlamaModel(cfg).eval()
    cpu_model = copy.deepcopy(base_model).eval()

    cpu_model.to(dtype=torch_dtype, device="cpu")
    model = base_model.to(dtype=torch_dtype, device=device)

    g = torch.Generator().manual_seed(0)
    input_ids_cpu = torch.randint(low=0, high=cfg.vocab_size, size=(batch, seq_len),
                                  generator=g, dtype=torch.long)

    min_dtype = torch.finfo(torch_dtype).min
    causal_mask = torch.full((seq_len, seq_len), fill_value=min_dtype,
                             dtype=torch_dtype, device="cpu")
    if seq_len > 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)
    attn_mask_cpu = causal_mask[None, None, :, :].expand(batch, 1, -1, -1)

    input_ids_dev = input_ids_cpu.to(device)
    attn_mask_dev = attn_mask_cpu.to(device)

    print("Compiling Llama3 model with torch.compile(...)")
    compiled_model = torch.compile(model, dynamic=False)

    out_cpu = f32_reference(cpu_model)(input_ids=input_ids_cpu, attention_mask=attn_mask_cpu.float())
    out_dev = compiled_model(input_ids=input_ids_dev, attention_mask=attn_mask_dev)

    check_result("Llama3 Model (last_hidden_state)", out_dev.last_hidden_state,
                out_cpu.last_hidden_state, rtol=rtol, atol=atol)
    diff = (out_dev.last_hidden_state.detach().cpu()
            - out_cpu.last_hidden_state.detach().cpu()).abs().max().item()
    print(f"Max diff > {diff}")


@torch.no_grad()
def run_one_layer_test(device, batch=1, seq_len=128, dtype="bfloat16", rtol=None, atol=None, decode=False):
    """LlamaModel's first decoder layer alone (tests/one_layer.py): a prefill of `seq_len` tokens, or with
    `decode` one step at position seq_len - 1 after a KV cache of the tokens before it."""
    print("\n[Running Llama3 one-layer Test]")
    tol = _TOL.get(dtype, _TOL["float32"])
    rtol, atol = tol[0] if rtol is None else rtol, tol[1] if atol is None else atol
    torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    cfg = llama3_config(seq_len)
    torch.manual_seed(0)
    model = LlamaModel(cfg).to(dtype=torch_dtype).eval()
    g = torch.Generator().manual_seed(0)
    input_ids = torch.randint(0, cfg.vocab_size, (batch, seq_len), generator=g, dtype=torch.long)
    out_cpu, out_dev = one_layer.run(model, model.layers[0], cfg, input_ids, device, decode=decode)
    check_result(f"Llama3 layer ({'decode' if decode else 'prefill'} {seq_len})", out_dev, out_cpu,
                 rtol=rtol, atol=atol)
    print("Max diff >", (out_dev.detach().cpu() - out_cpu.detach().cpu()).abs().max().item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Llama 3 (random weights, no tokenizer)")
    parser.add_argument("--part", type=str, default="all", choices=["all", "layer"],
                        help="all = the decoder layer and LlamaModel tests; "
                             "layer = LlamaModel's first layer alone")
    parser.add_argument("--batch", type=int, default=1)
    one_layer.add_grid_args(parser)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--rtol", type=float, default=None, help="default: _TOL of --dtype")
    parser.add_argument("--atol", type=float, default=None, help="default: _TOL of --dtype")
    args = parser.parse_args()
    if args.decode is not None and args.part != "layer":
        parser.error("--decode runs with --part layer")
    tol = _TOL.get(args.dtype, _TOL["float32"])
    args.rtol = tol[0] if args.rtol is None else args.rtol
    args.atol = tol[1] if args.atol is None else args.atol

    sys.path.append(os.environ.get("PYTORCHSIM_ROOT_PATH", "/workspace/PyTorchSim"))
    device = torch.device("npu:0")
    if args.part == "layer":
        run_one_layer_test(device, batch=args.batch, seq_len=args.decode or args.seq, dtype=args.dtype,
                           rtol=args.rtol, atol=args.atol, decode=args.decode is not None)
    else:
        run_decoder_layer_test(device=device, batch=args.batch, seq_len=args.seq,
                               dtype=args.dtype, rtol=args.rtol, atol=args.atol)
        run_model_test(device=device, batch=args.batch, seq_len=args.seq,
                       dtype=args.dtype, rtol=args.rtol, atol=args.atol)
