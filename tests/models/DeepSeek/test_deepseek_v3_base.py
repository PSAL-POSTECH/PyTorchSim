import os
import sys
import argparse
from pathlib import Path
import torch
sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from result_check import check_result, check_routed_rows, f32_reference
import one_layer

# DeepSeek-V3's remote modeling file (trust_remote_code) imports flash_attn, and
# transformers' check_imports statically requires it to be installed even though
# the npu backend never runs flash attention (math-only SDPA). The CI image is
# CPU-only and has no flash_attn, so register an import shim: it satisfies the
# static import check and any flash_attn[.submodule] import. It deliberately
# provides no package metadata, so transformers' is_flash_attn_2_available()
# stays False and the model selects the eager/sdpa path.
import importlib.abc
import importlib.util
import types


class _FlashAttnShim(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path=None, target=None):
        if name == "flash_attn" or name.startswith("flash_attn."):
            return importlib.util.spec_from_loader(name, self, is_package=True)
        return None

    def create_module(self, spec):
        module = types.ModuleType(spec.name)
        module.__path__ = []
        module.__getattr__ = lambda _name: None
        return module

    def exec_module(self, module):
        pass


if importlib.util.find_spec("flash_attn") is None:
    sys.meta_path.insert(0, _FlashAttnShim())

# recursive compile for some ops that are caused by graph break
torch.npu.register_eager_to_compile([
    "aten::zero_",
    "aten::sum.IntList_out",
    # NOT aten::mul.out, and the reason has changed since it was taken out.
    #
    # Registering an op makes the wrapper compile it, which is safe exactly
    # while Inductor can generate code for it. When it cannot, the aten call
    # comes back as a fallback and an ATen implementation that reaches the same
    # overload re-enters the wrapper -- mul.Tensor allocates an out and calls
    # mul.out -- until
    #
    #     RecursionError: maximum recursion depth exceeded
    #
    # DeepSeek-V2's complex RoPE closed that loop. It no longer does:
    # frontend/inductor/postgrad/complex.py rewrites the complex multiply into real
    # arithmetic, Inductor generates a kernel, and no fallback is emitted.
    # Measured both ways, and pinned by
    # tests/ops/misc/test_eager_to_compile_recursion.py.
    #
    # It stays out anyway. The trigger is gone, the SHAPE is not: any op whose
    # compiled form can fall back to itself does this, and nothing in
    # eager_to_compile guards against re-entry.
    "aten::floor_divide",
    "aten::floor_divide.Tensor",
    "aten::floor_divide.Scalar",
    "aten::cat.out",
    "aten::sort.values_stable",
])



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


def _cut_layers(config, max_layers):
    """Keep the first `max_layers` layers; every per-layer size stays the published one."""
    config.num_hidden_layers = min(int(max_layers), int(config.num_hidden_layers))
    return config


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
    check_result("cat.default", out, cpu_out, rtol=1e-4, atol=1e-4)


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
    check_result("cat.out", out, cpu_out, rtol=1e-4, atol=1e-4)
    
    
# (rtol, atol) per dtype; the bf16 atol is 3x the worst bf16 eager-vs-Inductor CPU diff at S = 128.
_TOL = {"bfloat16": (5e-2, 0.4), "float32": (3e-1, 2e-1)}
# Share of token rows a bf16 router near-tie may send to other experts (check_routed_rows); fp32 none.
_ROUTED = {"bfloat16": 0.2, "float32": 0.0}


@torch.no_grad()
def run_deepseek_v3_base(
    model_id,
    device,
    init_mode="config-random",
    max_layers=1,
    dtype="bfloat16",
    batch=1,
    seq_len=128,
    use_tokenizer=False,
    prompt="Hello, DeepSeek V3",
    trust_remote_code=False,
    revision=None,
    compile_model=True,
    part="model",
    decode=False,
):
    """DeepSeek-V3 at its published per-layer size, `max_layers` MoE layers (first_k_dense_replace 0), against
    CPU at `_TOL`: the CausalLM on a prefill of `seq_len` tokens, or with part "layer" its first decoder layer
    alone, on that prefill or with `decode` one step at position seq_len - 1 after a KV cache."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    torch_dtype = _dtype_from_str(dtype)
    rtol, atol = _TOL.get(dtype, _TOL["float32"])
    row_share = _ROUTED.get(dtype, 0.0)

    config = AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        revision=revision,
    )
    if init_mode == "pretrained" and getattr(config, "quantization_config", None) is not None:
        config.quantization_config = None
    if seq_len > config.max_position_embeddings:
        raise ValueError(f"seq {seq_len} exceeds {model_id}'s {config.max_position_embeddings} positions")
    config = _cut_layers(config, max_layers)
    config.first_k_dense_replace = 0

    torch.manual_seed(0)
    if init_mode == "config-random":
        model = AutoModelForCausalLM.from_config(
            config=config,
            trust_remote_code=trust_remote_code,
            dtype=torch_dtype,
        ).eval()
    elif init_mode == "pretrained":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            revision=revision,
        ).eval()
    else:
        raise ValueError(f"Unsupported init mode: {init_mode}")

    print("init mode:", init_mode)
    print("hidden_size:", config.hidden_size)
    print("num_hidden_layers:", config.num_hidden_layers,
          "first_k_dense_replace:", config.first_k_dense_replace)
    print("num_attention_heads:", config.num_attention_heads)
    print("routed experts:", config.n_routed_experts, "top-k:", config.num_experts_per_tok)
    print("model params:", sum(p.numel() for p in model.parameters()))

    if use_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )
        cpu_input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].cpu()
    else:
        vocab_size = getattr(config, "vocab_size", None)
        if vocab_size is None:
            raise ValueError("Config has no vocab_size; use --use-tokenizer or a model with vocab_size.")
        cpu_input_ids = _build_random_inputs(batch, seq_len, vocab_size, torch.device("cpu"))

    if part == "layer":
        cpu_out, npu_out = one_layer.run(model.model, model.model.layers[0], config, cpu_input_ids, device,
                                         decode=decode, compile_model=compile_model)
    else:
        cpu_out = _extract_logits(f32_reference(model)(cpu_input_ids))
        model_npu = model.to(device)
        if compile_model:
            model_npu = torch.compile(model_npu, dynamic=False)
        npu_out = _extract_logits(model_npu(cpu_input_ids.to(device)))

    check_routed_rows(f"DeepSeek V3 Base {part}", npu_out, cpu_out, rtol, atol, row_share)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSeek V3 download-based test")
    parser.add_argument("--model-id", type=str, default=os.environ.get("DEEPSEEK_V3_MODEL_ID", "deepseek-ai/DeepSeek-V3-Base"))
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--trust-remote-code", action="store_true", default=False,
                        help="use the Hub's modeling file instead of "
                             "transformers' own deepseek_v3 (4.x only)")
    parser.add_argument("--init-mode", type=str, default="config-random", choices=["config-random", "pretrained"])
    parser.add_argument("--max-layers", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--batch", type=int, default=1)
    one_layer.add_grid_args(parser)
    parser.add_argument("--part", type=str, default="model", choices=["model", "layer"],
                        help="model = DeepseekV3ForCausalLM; layer = its first decoder layer alone")
    parser.add_argument("--use-tokenizer", action="store_true")
    parser.add_argument("--prompt", type=str, default="Hello, DeepSeek V3")
    parser.add_argument("--compile", action="store_true", default=True)
    parser.add_argument("--test", type=str, default="e2e", choices=["all", "e2e", "cat"])

    args = parser.parse_args()
    if args.decode is not None and args.part != "layer":
        parser.error("--decode runs with --part layer")

    if not args.model_id:
        print("Error: --model-id is required (or set DEEPSEEK_V3_MODEL_ID).", file=sys.stderr)
        sys.exit(2)

    device = torch.device("npu:0")

    if args.test in ("all", "cat"):
        test_cat_default(device)
        test_cat_out(device)
    if args.test in ("all", "e2e"):
        run_deepseek_v3_base(
            model_id=args.model_id,
            device=device,
            init_mode=args.init_mode,
            max_layers=args.max_layers,
            dtype=args.dtype,
            batch=args.batch,
            seq_len=args.decode or args.seq,
            use_tokenizer=args.use_tokenizer,
            prompt=args.prompt,
            trust_remote_code=args.trust_remote_code,
            revision=args.revision,
            compile_model=args.compile,
            part=args.part,
            decode=args.decode is not None,
        )
