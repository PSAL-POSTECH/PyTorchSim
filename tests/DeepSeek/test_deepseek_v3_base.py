import os
import sys
import argparse
import copy
from pathlib import Path
import torch

# recursive compile for some ops that are caused by graph break
torch.npu.register_eager_to_compile([
    "aten::zero_",
    "aten::sum.IntList_out",
    "aten::mul.out",
    "aten::floor_divide",
    "aten::floor_divide.Tensor",
    "aten::floor_divide.Scalar",
    "aten::cat.out",
    "aten::sort.values_stable",
])


def test_result(name, out, cpu_out, rtol=1e-4, atol=1e-4):
    out_cpu = out.cpu()
    max_diff = (out_cpu - cpu_out).abs().max().item()
    mean_diff = (out_cpu - cpu_out).abs().mean().item()
    if torch.allclose(out_cpu, cpu_out, rtol=rtol, atol=atol):
        message = f"|{name} Test Passed|"
        print("-" * len(message))
        print(message)
        print("-" * len(message))
        print(f"Max absolute difference: {max_diff:.6f}")
        print(f"Mean absolute difference: {mean_diff:.6f}")
    else:
        message = f"|{name} Test Failed|"
        print("-" * len(message))
        print(message)
        print("-" * len(message))
        print("NPU out: ", out_cpu)
        print("CPU out: ", cpu_out)
        print(f"Max absolute difference: {max_diff:.6f}")
        print(f"Mean absolute difference: {mean_diff:.6f}")
        exit(1)


def _extract_logits(output):
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, (list, tuple)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Unsupported output type for comparison: {type(output)}")


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
        return 0.03, 1, 1, min(seq_len, 16)
    if preset == "small":
        return 0.07, 8, 1, min(seq_len, 32)
    if preset == "medium":
        return 0.10, 12, 1, min(seq_len, 48)
    return scale, max_layers, batch, seq_len


def _togsim_log_count() -> int:
    log_dir = Path("togsim_results")
    if not log_dir.exists():
        return 0
    return len(list(log_dir.glob("*.log")))


def _assert_simulation_happened(before_count: int, case_name: str):
    after_count = _togsim_log_count()
    if after_count <= before_count:
        raise RuntimeError(
            f"{case_name}: TOGSim log count did not increase "
            f"(before={before_count}, after={after_count})"
        )
    print(f"{case_name}: TOGSim logs increased ({before_count} -> {after_count})")


def test_cat_default(device):
    def cat_default_fn(a, b):
        return torch.cat([a, b], dim=0)

    x = torch.randn(8, 16, device=device)
    y = torch.randn(6, 16, device=device)
    opt_fn = torch.compile(dynamic=False)(cat_default_fn)

    before = _togsim_log_count()
    out = opt_fn(x, y)
    _assert_simulation_happened(before, "cat.default")

    cpu_out = torch.cat([x.cpu(), y.cpu()], dim=0)
    test_result("cat.default", out, cpu_out, rtol=1e-4, atol=1e-4)


def test_cat_out(device):
    def cat_out_fn(a, b, out):
        return torch.ops.aten.cat.out([a, b], 0, out=out)

    x = torch.randn(8, 16, device=device)
    y = torch.randn(6, 16, device=device)
    out_buf = torch.empty(14, 16, device=device)
    opt_fn = torch.compile(dynamic=False)(cat_out_fn)

    before = _togsim_log_count()
    out = opt_fn(x, y, out_buf)
    _assert_simulation_happened(before, "cat.out")

    cpu_out = torch.cat([x.cpu(), y.cpu()], dim=0)
    test_result("cat.out", out, cpu_out, rtol=1e-4, atol=1e-4)
    
    
@torch.no_grad()
def run_deepseek_v3_base(
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
        cpu_input_ids = encoded["input_ids"].cpu()
    else:
        vocab_size = getattr(config, "vocab_size", None)
        if vocab_size is None:
            raise ValueError("Config has no vocab_size; use --use-tokenizer or pass a model with vocab_size.")
        cpu_input_ids = _build_random_inputs(batch, seq_len, vocab_size, torch.device("cpu"))
    input_ids = cpu_input_ids.to(device)

    # CPU version
    model_cpu = copy.deepcopy(model).cpu().eval()
    cpu_out = _extract_logits(model_cpu(cpu_input_ids))

    # NPU version
    model_npu = copy.deepcopy(model_cpu).to(device).eval()
    if compile_model:
        model_npu = torch.compile(model_npu, dynamic=False)
    npu_out = _extract_logits(model_npu(input_ids))

    # Campare results
    test_result(
        "DeepSeek V3 Base",
        npu_out,
        cpu_out,
        rtol=3e-1,
        atol=2e-1,
    )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSeek V3 download-based test")
    parser.add_argument("--model-id", type=str, default=os.environ.get("DEEPSEEK_V3_MODEL_ID", "deepseek-ai/DeepSeek-V3-Base"))
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--init-mode", type=str, default="config-random", choices=["config-random", "pretrained"])
    parser.add_argument("--preset", type=str, default="small", choices=["none", "tiny", "small", "medium"])
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--use-tokenizer", action="store_true")
    parser.add_argument("--prompt", type=str, default="Hello, DeepSeek V3")
    parser.add_argument("--compile", action="store_true", default=True)
    parser.add_argument("--test", type=str, default="e2e", choices=["all", "e2e", "cat"])

    args = parser.parse_args()

    if not args.model_id:
        print("Error: --model-id is required (or set DEEPSEEK_V3_MODEL_ID).", file=sys.stderr)
        sys.exit(2)

    args.scale, args.max_layers, args.batch, args.seq_len = _apply_preset(
        args.scale, args.max_layers, args.batch, args.seq_len, args.preset
    )

    device = torch.device("npu:0")

    if args.test in ("all", "cat"):
        test_cat_default(device)
        test_cat_out(device)
    if args.test in ("all", "e2e"):
        run_deepseek_v3_base(
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
