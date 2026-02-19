import os
import sys
import argparse
import torch


def _dtype_from_str(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(name, torch.float32)


def _build_random_inputs(batch, seq_len, vocab_size, device):
    g = torch.Generator().manual_seed(0)
    input_ids = torch.randint(0, vocab_size, (batch, seq_len), generator=g, dtype=torch.int64)
    return input_ids.to(device)


def _safe_scaled_int(value, scale, min_value=1):
    return max(min_value, int(round(float(value) * float(scale))))


def _round_to_multiple(value, multiple, min_value=1):
    if multiple is None or multiple <= 0:
        return max(min_value, int(value))
    v = max(min_value, int(value))
    return max(min_value, ((v + multiple - 1) // multiple) * multiple)


def _maybe_scale_config(config, scale=1.0, max_layers=None):
    if scale == 1.0 and max_layers is None:
        return config

    if hasattr(config, "hidden_size"):
        config.hidden_size = _safe_scaled_int(config.hidden_size, scale)
    if hasattr(config, "intermediate_size"):
        config.intermediate_size = _safe_scaled_int(config.intermediate_size, scale)
    if hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = _safe_scaled_int(config.num_hidden_layers, scale)
    if hasattr(config, "num_attention_heads"):
        config.num_attention_heads = _safe_scaled_int(config.num_attention_heads, scale)
    if hasattr(config, "num_key_value_heads"):
        config.num_key_value_heads = min(
            _safe_scaled_int(config.num_key_value_heads, scale),
            config.num_attention_heads,
        )

    for name in [
        "n_routed_experts",
        "n_shared_experts",
        "num_local_experts",
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
    ]:
        if hasattr(config, name):
            setattr(config, name, _safe_scaled_int(getattr(config, name), scale))

    # DeepSeek MoE gate expects n_routed_experts to be divisible by n_group.
    if hasattr(config, "n_routed_experts") and hasattr(config, "n_group"):
        config.n_routed_experts = _round_to_multiple(
            config.n_routed_experts,
            config.n_group,
            min_value=max(1, int(config.n_group)),
        )

    if max_layers is not None and hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = max(1, min(int(max_layers), int(config.num_hidden_layers)))

    if hasattr(config, "hidden_size") and hasattr(config, "num_attention_heads"):
        config.hidden_size = max(
            config.num_attention_heads,
            (config.hidden_size // config.num_attention_heads) * config.num_attention_heads,
        )

    return config


def _apply_preset(scale, max_layers, batch, seq_len, preset):
    if preset == "tiny":
        return 0.03, 4, 1, min(seq_len, 16)
    if preset == "small":
        return 0.07, 8, 1, min(seq_len, 32)
    if preset == "medium":
        return 0.10, 12, 1, min(seq_len, 48)
    return scale, max_layers, batch, seq_len


@torch.no_grad()
def run_deep_seek_v3_base_test(
    model_id,
    device,
    init_mode="config-random",
    scale=1.0,
    max_layers=None,
    dtype="float16",
    batch=1,
    seq_len=32,
    use_tokenizer=False,
    prompt="Hello, DeepSeek V3",
    trust_remote_code=False,
    revision=None,
    compile_model=False,
):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    torch_dtype = _dtype_from_str(dtype)

    # Load model config
    config = AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        revision=revision,
    )

    # Some remote model codes expect quantization_config to stay object-like
    # (call .to_dict()), so only disable it for pretrained loading path.
    if init_mode == "pretrained" and getattr(config, "quantization_config", None) is not None:
        config.quantization_config = None

    config = _maybe_scale_config(config, scale=scale, max_layers=max_layers)

    if init_mode == "config-random":
        model = AutoModelForCausalLM.from_config(
            config=config,
            trust_remote_code=trust_remote_code,
        ).eval()
        model = model.to(dtype=torch_dtype)
    elif init_mode == "pretrained":
        # Load model(weights)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            revision=revision,
        ).eval()
    else:
        raise ValueError(f"Unsupported init mode: {init_mode}")

    model = model.to(device)
    model_params = sum(p.numel() for p in model.parameters())
    print("init mode:", init_mode)
    print("scaled hidden_size:", getattr(config, "hidden_size", "n/a"))
    print("scaled num_hidden_layers:", getattr(config, "num_hidden_layers", "n/a"))
    print("scaled num_attention_heads:", getattr(config, "num_attention_heads", "n/a"))
    print("model params:", model_params)

    # Load tokenizer
    if use_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
    else:
        vocab_size = getattr(config, "vocab_size", None)
        if vocab_size is None:
            raise ValueError("Config has no vocab_size; use --use-tokenizer or pass a model with vocab_size.")
        input_ids = _build_random_inputs(batch, seq_len, vocab_size, device)

    if compile_model:
        model = torch.compile(model, dynamic=False)

    out = model(input_ids)
    logits = out.logits
    
    print("logits shape:", tuple(logits.shape))
    print("logits dtype:", logits.dtype)
    print("logits max:", logits.max().item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSeek V3 download-based test")
    parser.add_argument("--model-id", type=str, default=os.environ.get("DEEPSEEK_V3_MODEL_ID", "deepseek-ai/DeepSeek-V3-Base"))
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--init-mode", type=str, default="config-random", choices=["config-random", "pretrained"])
    parser.add_argument("--preset", type=str, default="tiny", choices=["none", "tiny", "small", "medium"])
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--use-tokenizer", action="store_true")
    parser.add_argument("--prompt", type=str, default="Hello, DeepSeek V3")
    parser.add_argument("--compile", action="store_true", default=True)

    args = parser.parse_args()

    if not args.model_id:
        print("Error: --model-id is required (or set DEEPSEEK_V3_MODEL_ID).", file=sys.stderr)
        sys.exit(2)

    args.scale, args.max_layers, args.batch, args.seq_len = _apply_preset(
        args.scale, args.max_layers, args.batch, args.seq_len, args.preset
    )

    device = torch.device("npu:0")

    run_deep_seek_v3_base_test(
        model_id=args.model_id,
        device=device,
        init_mode=args.init_mode,
        scale=args.scale,
        max_layers=args.max_layers,
        dtype=args.dtype,
        batch=args.batch,
        seq_len=args.seq_len,
        use_tokenizer=args.use_tokenizer,
        prompt=args.prompt,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
        compile_model=args.compile,
    )
