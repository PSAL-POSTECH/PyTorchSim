import os
import sys
import argparse
import copy
import torch
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding, LlamaModel

def test_result(name, out, ref, rtol=1e-4, atol=1e-4):
    if torch.allclose(out.cpu(), ref.cpu(), rtol=rtol, atol=atol):
        msg = f"|{name} Test Passed|"
        print("-" * len(msg)); print(msg); print("-" * len(msg))
    else:
        msg = f"|{name} Test Failed|"
        print("-" * len(msg)); print(msg); print("-" * len(msg))
        diff = (out.cpu().int() - ref.cpu().int()).abs().max().item()
        print("device out:", out.detach().cpu())
        print("cpu ref  :", ref.detach().cpu())
        print(f"Max abs diff: {diff}")
        sys.exit(1)

@torch.no_grad()
def run_rmsnorm_test(
    device,
    batch=1,
    seq_len=32,
    dtype="float32",
    rtol=1e-3,
    atol=1e-3,
):
    print("\n[Running LlamaRMSNorm Test]")
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(dtype, torch.float32)

    hidden_size = 4096
    eps = 1e-6

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

    test_result("LlamaRMSNorm forward", out_dev, out_cpu, rtol=rtol, atol=atol)
    print("Max diff >", (out_dev.detach().cpu() - out_cpu.detach().cpu()).abs().max().item())


@torch.no_grad()
def run_rotary_embedding_test(
    device,
    batch=1,
    seq_len=32,
    dtype="float32",
    rtol=1e-3,
    atol=1e-3,
):
    print("\n[Running LlamaRotaryEmbedding Test]")
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(dtype, torch.float32)

    hidden_size = 4096
    num_heads = 32
    head_dim = hidden_size // num_heads

    cfg = LlamaConfig(
        _name_or_path="custom-llama",
        architectures=["LlamaForCausalLM"],
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        hidden_act="silu",
        hidden_size=4096,
        initializer_range=0.02,
        intermediate_size=11008,
        max_position_embeddings=4096,
        mlp_bias=False,
        model_type="llama",
        num_attention_heads=32,
        num_hidden_layers=1,
        num_key_value_heads=32,
        pretraining_tp=1,
        rms_norm_eps=1e-06,
        rope_scaling=None,
        rope_theta=10000.0,
        tie_word_embeddings=True,
        torch_dtype=dtype,
        transformers_version="4.43.4",
        use_cache=True,
        vocab_size=8192,
        _attn_implementation = "sdpa"
    )
    # Pass dim explicitly to avoid config parsing issues
    base_rope = LlamaRotaryEmbedding(dim=head_dim, max_position_embeddings=cfg.max_position_embeddings, base=cfg.rope_theta, config=cfg)

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

    test_result("LlamaRotaryEmbedding (Cos)", cos_dev, cos_cpu, rtol=rtol, atol=atol)
    test_result("LlamaRotaryEmbedding (Sin)", sin_dev, sin_cpu, rtol=rtol, atol=atol)

    diff_cos = (cos_dev.detach().cpu() - cos_cpu.detach().cpu()).abs().max().item()
    diff_sin = (sin_dev.detach().cpu() - sin_cpu.detach().cpu()).abs().max().item()
    print(f"Max diff (Cos) > {diff_cos}")
    print(f"Max diff (Sin) > {diff_sin}")

@torch.no_grad()
def run_decoder_layer_test(
    device,
    batch=1,
    seq_len=32,
    dtype="float32",
    rtol=1e-3,
    atol=1e-3,
):
    print("\n[Running LlamaDecoderLayer Test]")
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(dtype, torch.float32)

    cfg = LlamaConfig(
        _name_or_path="custom-llama",
        architectures=["LlamaForCausalLM"],
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        hidden_act="silu",
        hidden_size=4096,
        initializer_range=0.02,
        intermediate_size=11008,
        max_position_embeddings=4096,
        mlp_bias=False,
        model_type="llama",
        num_attention_heads=32,
        num_hidden_layers=1,
        num_key_value_heads=32,
        pretraining_tp=1,
        rms_norm_eps=1e-06,
        rope_scaling=None,
        rope_theta=10000.0,
        tie_word_embeddings=True,
        torch_dtype=dtype,
        transformers_version="4.43.4",
        use_cache=True,
        vocab_size=8192,
        _attn_implementation = "sdpa"
    )

    print("Building LlamaDecoderLayer from custom config.")
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

    out_cpu = cpu_layer(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        position_embeddings=position_embeddings
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

    test_result("LlamaDecoderLayer forward", out_dev, out_cpu, rtol=rtol, atol=atol)
    print("Max diff >", (out_dev.detach().cpu() - out_cpu.detach().cpu()).abs().max().item())

@torch.no_grad()
def run_custom_llama_test(
    device,
    batch=1,
    seq_len=32,
    dtype="float32",
    rtol=1e-3,
    atol=1e-3,
    max_new_tokens=16,
):
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(dtype, torch.float32)

    cfg = LlamaConfig(
        _name_or_path="custom-llama",
        architectures=["LlamaForCausalLM"],
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        hidden_act="silu",
        hidden_size=1024,
        initializer_range=0.02,
        intermediate_size=11008,
        max_position_embeddings=4096,
        mlp_bias=False,
        model_type="llama",
        num_attention_heads=32,
        num_hidden_layers=1,
        num_key_value_heads=32,
        pretraining_tp=1,
        rms_norm_eps=1e-06,
        rope_scaling=None,
        rope_theta=10000.0,
        tie_word_embeddings=True,
        torch_dtype=dtype,
        transformers_version="4.43.4",
        use_cache=True,
        vocab_size=8192,
    )

    print("Building LlamaForCausalLM from custom config (random init).")
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

    test_result("Custom Llama forward(logits)", logits_dev, logits_cpu, rtol=rtol, atol=atol)
    print("Max diff >", (logits_dev.detach().cpu() - logits_cpu.detach().cpu()).abs().max().item())

@torch.no_grad()
def run_llama_model_test(
    device,
    batch=1,
    seq_len=32,
    dtype="float32",
    rtol=1e-3,
    atol=1e-3,
):
    print("\n[Running LlamaModel Test]")
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(dtype, torch.float32)

    cfg = LlamaConfig(
        vocab_size=8192,
        hidden_size=1024,
        num_attention_heads=32,
        num_key_value_heads=32,
        intermediate_size=11008 // 4,
        num_hidden_layers=1,
        max_position_embeddings=4096,
        hidden_act="silu",
        use_cache=False,
        torch_dtype=dtype,
    )

    print("Building LlamaModel from custom config (random init).")
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

    out_cpu = cpu_model(input_ids=input_ids_cpu, attention_mask=attn_mask_cpu)
    out_dev = compiled_model(input_ids=input_ids_dev, attention_mask=attn_mask_dev)

    last_hidden_state_cpu = out_cpu.last_hidden_state
    last_hidden_state_dev = out_dev.last_hidden_state

    test_result("LlamaModel (last_hidden_state)", last_hidden_state_dev, last_hidden_state_cpu, rtol=rtol, atol=atol)
    diff = (last_hidden_state_dev.detach().cpu() - last_hidden_state_cpu.detach().cpu()).abs().max().item()
    print(f"Max diff > {diff}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Custom Llama (random weights, no tokenizer)")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    args = parser.parse_args()

    sys.path.append(os.environ.get("PYTORCHSIM_ROOT_PATH", "/workspace/PyTorchSim"))
    device = torch.device("npu:0")
    #test_triu(device, size=(32, 128), diagonal=1)
    torch.compiler.is_compiling = lambda: True # FIXME. How to fix this?
    #run_rmsnorm_test(device)
    #run_rotary_embedding_test(device)
    run_decoder_layer_test(
        device=device,
        batch=args.batch,
        seq_len=args.seq_len,
        dtype=args.dtype,
        rtol=args.rtol,
        atol=args.atol,
    )
    run_llama_model_test(device)
    #run_custom_llama_test(
    #    device=device,
    #    batch=args.batch,
    #    seq_len=args.seq_len,
    #    dtype=args.dtype,
    #    rtol=args.rtol,
    #    atol=args.atol,
    #)
