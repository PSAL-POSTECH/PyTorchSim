import os
import sys
import argparse
import copy
import torch
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding, LlamaModel
sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from result_check import check_result, f32, f32_reference
import one_layer

# meta-llama/Llama-2-7b-hf, as published; only num_hidden_layers is cut.
_LLAMA2_7B = dict(
    vocab_size=32000,
    hidden_size=4096,
    intermediate_size=11008,
    num_attention_heads=32,
    num_key_value_heads=32,
    max_position_embeddings=4096,
    rms_norm_eps=1e-5,
    rope_theta=10000.0,
    rope_scaling=None,
    hidden_act="silu",
    attention_bias=False,
    mlp_bias=False,
    pretraining_tp=1,
    tie_word_embeddings=False,
    bos_token_id=1,
    eos_token_id=2,
    initializer_range=0.02,
)

# (rtol, atol) per dtype; the bf16 atol is 3x the worst bf16 eager-vs-Inductor CPU diff at S = 128.
_TOL = {"bfloat16": (5e-2, 0.4), "float32": (1e-3, 1e-3)}


def llama2_config(seq_len, dtype="float32", layers=1, **overrides):
    """Llama-2-7B at its published sizes with `layers` blocks; raises past its 4096 positions."""
    if seq_len > _LLAMA2_7B["max_position_embeddings"]:
        raise ValueError(f"seq {seq_len} exceeds Llama-2-7B's "
                         f"{_LLAMA2_7B['max_position_embeddings']} positions")
    return LlamaConfig(**_LLAMA2_7B, num_hidden_layers=layers, torch_dtype=dtype, **overrides)

@torch.no_grad()
def run_rmsnorm_test(
    device,
    batch=1,
    seq_len=128,
    dtype="float32",
    rtol=1e-3,
    atol=1e-3,
):
    print("\n[Running LlamaRMSNorm Test]")
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(dtype, torch.float32)

    hidden_size = _LLAMA2_7B["hidden_size"]
    eps = _LLAMA2_7B["rms_norm_eps"]

    print(f"Building LlamaRMSNorm (hidden_size={hidden_size}, eps={eps})")
    base_norm = LlamaRMSNorm(hidden_size=hidden_size, eps=eps).eval()
    cpu_norm = copy.deepcopy(base_norm).eval()

    cpu_norm.to(dtype=torch_dtype, device="cpu")
    model = base_norm.to(dtype=torch_dtype, device=device)

    g = torch.Generator().manual_seed(0)
    hidden_states = torch.randn(batch, seq_len, hidden_size, generator=g, dtype=torch_dtype)
    hs_dev = hidden_states.to(device)

    print("Compiling LlamaRMSNorm with torch.compile(...)")
    compiled_norm = torch.compile(model, dynamic=False)

    out_cpu = cpu_norm(hidden_states)
    out_dev = compiled_norm(hs_dev)

    check_result("LlamaRMSNorm forward", out_dev, out_cpu, rtol=rtol, atol=atol)
    print("Max diff >", (out_dev.detach().cpu() - out_cpu.detach().cpu()).abs().max().item())


@torch.no_grad()
def run_rotary_embedding_test(
    device,
    batch=1,
    seq_len=128,
    dtype="float32",
    rtol=1e-3,
    atol=1e-3,
):
    print("\n[Running LlamaRotaryEmbedding Test]")
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(dtype, torch.float32)

    num_heads = _LLAMA2_7B["num_attention_heads"]
    head_dim = _LLAMA2_7B["hidden_size"] // num_heads

    cfg = llama2_config(seq_len, dtype, attn_implementation="sdpa")
    # The config carries dim, max_position_embeddings and base. Passing them
    # again as kwargs was the 4.43 spelling; transformers removed them, and the
    # config path computes the same inv_freq. head_dim below stays because the
    # test still shapes its own value tensor.
    base_rope = LlamaRotaryEmbedding(config=cfg)

    cpu_rope = copy.deepcopy(base_rope)

    cpu_rope.to(device="cpu")
    model = base_rope.to(device=device)

    g = torch.Generator().manual_seed(0)
    value = torch.randn(batch, num_heads, seq_len, head_dim, generator=g, dtype=torch_dtype)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch, -1)

    val_dev = value.to(device)
    pos_dev = position_ids.to(device)

    print("Compiling LlamaRotaryEmbedding with torch.compile(...)")
    compiled_rope = torch.compile(model, dynamic=False)

    cos_cpu, sin_cpu = cpu_rope(value, position_ids)
    cos_dev, sin_dev = compiled_rope(val_dev, pos_dev)

    print(f"Output dtype check - CPU: {cos_cpu.dtype}, Device: {cos_dev.dtype}")

    check_result("LlamaRotaryEmbedding (Cos)", cos_dev, cos_cpu, rtol=rtol, atol=atol)
    check_result("LlamaRotaryEmbedding (Sin)", sin_dev, sin_cpu, rtol=rtol, atol=atol)

    diff_cos = (cos_dev.detach().cpu() - cos_cpu.detach().cpu()).abs().max().item()
    diff_sin = (sin_dev.detach().cpu() - sin_cpu.detach().cpu()).abs().max().item()
    print(f"Max diff (Cos) > {diff_cos}")
    print(f"Max diff (Sin) > {diff_sin}")

@torch.no_grad()
def run_decoder_layer_test(
    device,
    batch=1,
    seq_len=128,
    dtype="float32",
    rtol=1e-3,
    atol=1e-3,
):
    print("\n[Running LlamaDecoderLayer Test]")
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(dtype, torch.float32)

    cfg = llama2_config(seq_len, dtype, attn_implementation="sdpa")

    print("Building LlamaDecoderLayer at Llama-2-7B sizes.")
    base_layer = LlamaDecoderLayer(cfg, layer_idx=0).eval()
    cpu_layer = copy.deepcopy(base_layer).eval()

    cpu_layer.to(dtype=torch_dtype, device="cpu")
    model = base_layer.to(dtype=torch_dtype, device=device)

    g = torch.Generator().manual_seed(0)
    hidden_states = torch.randn(batch, seq_len, cfg.hidden_size, generator=g, dtype=torch_dtype)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch, -1)

    attention_mask = torch.zeros(batch, 1, seq_len, seq_len, dtype=torch_dtype)
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    attention_mask.masked_fill_(mask, torch.finfo(torch_dtype).min)

    # Shape: (1, seq_len, head_dim) or (batch, seq_len, head_dim)
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    cos = torch.randn(1, seq_len, head_dim, generator=g, dtype=torch_dtype)
    sin = torch.randn(1, seq_len, head_dim, generator=g, dtype=torch_dtype)
    position_embeddings = (cos, sin)

    hs_dev = hidden_states.to(device)
    pos_dev = position_ids.to(device)
    att_dev = attention_mask.to(device)
    pos_emb_dev = (cos.to(device), sin.to(device))

    print("Compiling LlamaDecoderLayer with torch.compile(...)")
    compiled_layer = torch.compile(model, dynamic=False)

    out_cpu = f32_reference(cpu_layer)(
        hidden_states=hidden_states.float(),
        attention_mask=attention_mask.float(),
        position_ids=position_ids,
        position_embeddings=f32(position_embeddings)
    )
    if isinstance(out_cpu, tuple):
        out_cpu = out_cpu[0]

    out_dev = compiled_layer(
        hidden_states=hs_dev,
        attention_mask=att_dev,
        position_ids=pos_dev,
        position_embeddings=pos_emb_dev
    )
    if isinstance(out_dev, tuple):
        out_dev = out_dev[0]

    check_result("LlamaDecoderLayer forward", out_dev, out_cpu, rtol=rtol, atol=atol)
    print("Max diff >", (out_dev.detach().cpu() - out_cpu.detach().cpu()).abs().max().item())

@torch.no_grad()
def run_custom_llama_test(
    device,
    batch=1,
    seq_len=128,
    dtype="float32",
    rtol=1e-3,
    atol=1e-3,
    max_new_tokens=16,
):
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(dtype, torch.float32)

    cfg = llama2_config(seq_len, dtype)

    print("Building LlamaForCausalLM at Llama-2-7B sizes (random init).")
    base_model = LlamaForCausalLM(cfg).eval()
    cpu_model  = copy.deepcopy(base_model).eval()

    cpu_model.to(dtype=torch_dtype, device="cpu")
    model = base_model.to(dtype=torch_dtype, device=device)

    g = torch.Generator().manual_seed(0)
    vocab = cfg.vocab_size
    input_ids_cpu = torch.randint(low=0, high=vocab, size=(batch, seq_len), generator=g, dtype=torch.long)

    min_dtype = torch.finfo(torch_dtype).min
    causal_mask = torch.zeros((seq_len, seq_len), dtype=torch_dtype, device="cpu")

    if seq_len > 1:
        causal_mask = torch.triu(torch.full_like(causal_mask, min_dtype), diagonal=1)

    cache_position = torch.arange(seq_len, device="cpu")
    mask_condition = torch.arange(seq_len, device="cpu") > cache_position.reshape(-1, 1)
    causal_mask.masked_fill_(mask_condition, min_dtype)
    attn_mask_cpu = causal_mask[None, None, :, :].expand(batch, 1, -1, -1)

    input_ids_dev = input_ids_cpu.to(device)
    attn_mask_dev = attn_mask_cpu.to(device)

    # ---- forward comparison (compile vs CPU baseline) ----
    print("Compiling model with torch.compile(...)")
    compiled = torch.compile(model, dynamic=False)

    logits_cpu = cpu_model(input_ids=input_ids_cpu, attention_mask=attn_mask_cpu)#.logits
    logits_dev = compiled(input_ids=input_ids_dev, attention_mask=attn_mask_dev)#.logits

    check_result("Custom Llama forward(logits)", logits_dev, logits_cpu, rtol=rtol, atol=atol)
    print("Max diff >", (logits_dev.detach().cpu() - logits_cpu.detach().cpu()).abs().max().item())

@torch.no_grad()
def run_llama_model_test(
    device,
    batch=1,
    seq_len=128,
    dtype="float32",
    rtol=1e-3,
    atol=1e-3,
):
    print("\n[Running LlamaModel Test]")
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(dtype, torch.float32)

    cfg = llama2_config(seq_len, dtype, use_cache=False)

    print("Building LlamaModel at Llama-2-7B sizes, 1 layer (random init).")
    base_model = LlamaModel(cfg).eval()
    cpu_model = copy.deepcopy(base_model).eval()

    cpu_model.to(dtype=torch_dtype, device="cpu")
    model = base_model.to(dtype=torch_dtype, device=device)

    g = torch.Generator().manual_seed(0)
    input_ids_cpu = torch.randint(low=0, high=cfg.vocab_size, size=(batch, seq_len), generator=g, dtype=torch.long)

    min_dtype = torch.finfo(torch_dtype).min
    causal_mask = torch.full((seq_len, seq_len), fill_value=min_dtype, dtype=torch_dtype, device="cpu")
    if seq_len > 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)
    attn_mask_cpu = causal_mask[None, None, :, :].expand(batch, 1, -1, -1)

    input_ids_dev = input_ids_cpu.to(device)
    attn_mask_dev = attn_mask_cpu.to(device)

    print("Compiling LlamaModel with torch.compile(...)")
    compiled_model = torch.compile(model, dynamic=False)

    out_cpu = f32_reference(cpu_model)(input_ids=input_ids_cpu, attention_mask=attn_mask_cpu.float())
    out_dev = compiled_model(input_ids=input_ids_dev, attention_mask=attn_mask_dev)

    last_hidden_state_cpu = out_cpu.last_hidden_state
    last_hidden_state_dev = out_dev.last_hidden_state

    check_result("LlamaModel (last_hidden_state)", last_hidden_state_dev, last_hidden_state_cpu, rtol=rtol, atol=atol)
    diff = (last_hidden_state_dev.detach().cpu() - last_hidden_state_cpu.detach().cpu()).abs().max().item()
    print(f"Max diff > {diff}")

@torch.no_grad()
def run_one_layer_test(device, batch=1, seq_len=128, dtype="bfloat16", rtol=None, atol=None, decode=False):
    """LlamaModel's first decoder layer alone (tests/one_layer.py): a prefill of `seq_len` tokens, or with
    `decode` one step at position seq_len - 1 after a KV cache of the tokens before it."""
    print("\n[Running Llama-2-7B one-layer Test]")
    tol = _TOL.get(dtype, _TOL["float32"])
    rtol, atol = tol[0] if rtol is None else rtol, tol[1] if atol is None else atol
    cfg = llama2_config(seq_len, dtype, attn_implementation="sdpa")
    torch.manual_seed(0)
    torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    model = LlamaModel(cfg).to(dtype=torch_dtype).eval()
    g = torch.Generator().manual_seed(0)
    input_ids = torch.randint(0, cfg.vocab_size, (batch, seq_len), generator=g, dtype=torch.long)
    out_cpu, out_dev = one_layer.run(model, model.layers[0], cfg, input_ids, device, decode=decode)
    check_result(f"Llama-2-7B layer ({'decode' if decode else 'prefill'} {seq_len})", out_dev, out_cpu,
                 rtol=rtol, atol=atol)
    print("Max diff >", (out_dev.detach().cpu() - out_cpu.detach().cpu()).abs().max().item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Custom Llama (random weights, no tokenizer)")
    parser.add_argument("--part", type=str, default="all", choices=["all", "layer"],
                        help="all = the decoder layer and LlamaModel tests; "
                             "layer = LlamaModel's first layer alone")
    parser.add_argument("--batch", type=int, default=1)
    one_layer.add_grid_args(parser)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--rtol", type=float, default=None, help="default: _TOL of --dtype")
    parser.add_argument("--atol", type=float, default=None, help="default: _TOL of --dtype")
    parser.add_argument("--max_new_tokens", type=int, default=16)
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
        run_decoder_layer_test(device, batch=args.batch, seq_len=args.seq, dtype=args.dtype,
                               rtol=args.rtol, atol=args.atol)
        run_llama_model_test(device, batch=args.batch, seq_len=args.seq, dtype=args.dtype,
                             rtol=args.rtol, atol=args.atol)
