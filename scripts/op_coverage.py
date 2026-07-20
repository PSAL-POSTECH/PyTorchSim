"""Op-coverage diagnostic for new LLM models on PyTorchSim.

Runs each model in two phases:
  Phase 1 (enumerate): custom torch.compile backend captures the FX graph and
                       lists every aten op that appears, without touching NPU.
  Phase 2 (run):       torch.compile(model) on npu:0, real forward. On crash,
                       parses the traceback to identify the failing op.

Usage:
  python scripts/op_coverage.py                    # all models
  python scripts/op_coverage.py --models qwen2     # subset
  python scripts/op_coverage.py --enumerate-only   # skip NPU compile (fast)
"""

import argparse
import datetime as _dt
import os
import re
import sys
import traceback
from contextlib import contextmanager

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# ---------------------------------------------------------------------------
# Model registry: each entry returns (model, kwargs_for_forward) on CPU.
# Sizes follow "small but realistic" variants (1-layer) so a forward is cheap
# enough to actually drive through TOGSim.
# ---------------------------------------------------------------------------

def _causal_mask(batch, seq_len, dtype):
    min_v = torch.finfo(dtype).min
    m = torch.full((seq_len, seq_len), min_v, dtype=dtype)
    if seq_len > 1:
        m = torch.triu(m, diagonal=1)
    return m[None, None, :, :].expand(batch, 1, -1, -1).contiguous()


def build_qwen2(batch=1, seq_len=32, dtype=torch.float32):
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
    cfg = Qwen2Config(
        vocab_size=4096,
        hidden_size=1536,
        num_attention_heads=12,
        num_key_value_heads=2,
        intermediate_size=8960,
        num_hidden_layers=2,
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        torch_dtype=dtype,
        use_cache=False,
        _attn_implementation="eager",
    )
    model = Qwen2Model(cfg).eval().to(dtype=dtype)
    input_ids = torch.randint(0, cfg.vocab_size, (batch, seq_len))
    attn_mask = _causal_mask(batch, seq_len, dtype)
    return model, {"input_ids": input_ids, "attention_mask": attn_mask}


def build_gemma(batch=1, seq_len=32, dtype=torch.float32):
    from transformers.models.gemma.configuration_gemma import GemmaConfig
    from transformers.models.gemma.modeling_gemma import GemmaModel
    cfg = GemmaConfig(
        vocab_size=4096,
        hidden_size=2048,
        num_attention_heads=8,
        num_key_value_heads=1,
        intermediate_size=16384,
        num_hidden_layers=2,
        head_dim=256,
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        torch_dtype=dtype,
        use_cache=False,
        _attn_implementation="eager",
    )
    model = GemmaModel(cfg).eval().to(dtype=dtype)
    input_ids = torch.randint(0, cfg.vocab_size, (batch, seq_len))
    attn_mask = _causal_mask(batch, seq_len, dtype)
    return model, {"input_ids": input_ids, "attention_mask": attn_mask}


def build_gemma2(batch=1, seq_len=32, dtype=torch.float32):
    from transformers.models.gemma2.configuration_gemma2 import Gemma2Config
    from transformers.models.gemma2.modeling_gemma2 import Gemma2Model
    cfg = Gemma2Config(
        vocab_size=4096,
        hidden_size=2304,
        num_attention_heads=8,
        num_key_value_heads=4,
        intermediate_size=9216,
        num_hidden_layers=2,
        head_dim=256,
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        torch_dtype=dtype,
        use_cache=False,
        attn_logit_softcapping=50.0,
        final_logit_softcapping=30.0,
        sliding_window=16,
        _attn_implementation="eager",
    )
    model = Gemma2Model(cfg).eval().to(dtype=dtype)
    input_ids = torch.randint(0, cfg.vocab_size, (batch, seq_len))
    attn_mask = _causal_mask(batch, seq_len, dtype)
    return model, {"input_ids": input_ids, "attention_mask": attn_mask}


def build_phi3(batch=1, seq_len=32, dtype=torch.float32):
    from transformers.models.phi3.configuration_phi3 import Phi3Config
    from transformers.models.phi3.modeling_phi3 import Phi3Model
    cfg = Phi3Config(
        vocab_size=4096,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        hidden_size=3072,
        num_attention_heads=32,
        num_key_value_heads=32,
        intermediate_size=8192,
        num_hidden_layers=2,
        max_position_embeddings=4096,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        torch_dtype=dtype,
        use_cache=False,
        _attn_implementation="eager",
    )
    model = Phi3Model(cfg).eval().to(dtype=dtype)
    input_ids = torch.randint(0, cfg.vocab_size, (batch, seq_len))
    attn_mask = _causal_mask(batch, seq_len, dtype)
    return model, {"input_ids": input_ids, "attention_mask": attn_mask}


def _build_lm(cfg, ModelCls, batch, seq_len, dtype):
    """Shared helper: build a causal-LM-style model and matching token+mask inputs."""
    model = ModelCls(cfg).eval().to(dtype=dtype)
    input_ids = torch.randint(0, cfg.vocab_size, (batch, seq_len))
    attn_mask = _causal_mask(batch, seq_len, dtype)
    return model, {"input_ids": input_ids, "attention_mask": attn_mask}


def build_qwen3(batch=1, seq_len=32, dtype=torch.float32):
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model
    cfg = Qwen3Config(
        vocab_size=4096, hidden_size=1024, num_attention_heads=8, num_key_value_heads=4,
        intermediate_size=3072, num_hidden_layers=2, max_position_embeddings=4096,
        rms_norm_eps=1e-6, rope_theta=1000000.0, torch_dtype=dtype, use_cache=False,
        _attn_implementation="eager",
    )
    return _build_lm(cfg, Qwen3Model, batch, seq_len, dtype)


def build_qwen3_moe(batch=1, seq_len=32, dtype=torch.float32):
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeModel
    cfg = Qwen3MoeConfig(
        vocab_size=4096, hidden_size=1024, num_attention_heads=8, num_key_value_heads=4,
        intermediate_size=3072, moe_intermediate_size=768, num_experts=4, num_experts_per_tok=2,
        decoder_sparse_step=1, num_hidden_layers=2, max_position_embeddings=4096,
        rms_norm_eps=1e-6, rope_theta=1000000.0, torch_dtype=dtype, use_cache=False,
        _attn_implementation="eager",
    )
    return _build_lm(cfg, Qwen3MoeModel, batch, seq_len, dtype)


def build_gemma3(batch=1, seq_len=32, dtype=torch.float32):
    from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
    from transformers.models.gemma3.modeling_gemma3 import Gemma3TextModel
    cfg = Gemma3TextConfig(
        vocab_size=4096, hidden_size=2048, num_attention_heads=8, num_key_value_heads=4,
        intermediate_size=8192, head_dim=256, num_hidden_layers=2,
        sliding_window=16, sliding_window_pattern=2,
        max_position_embeddings=4096, rms_norm_eps=1e-6, rope_theta=10000.0,
        torch_dtype=dtype, use_cache=False, _attn_implementation="eager",
    )
    return _build_lm(cfg, Gemma3TextModel, batch, seq_len, dtype)


def build_deepseek_v3(batch=1, seq_len=32, dtype=torch.float32):
    from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3Model
    cfg = DeepseekV3Config(
        vocab_size=4096, hidden_size=1024, num_attention_heads=16, num_key_value_heads=16,
        intermediate_size=4096, moe_intermediate_size=512,
        n_routed_experts=8, num_experts_per_tok=2, n_shared_experts=1,
        n_group=2, topk_group=1,
        q_lora_rank=512, kv_lora_rank=128, qk_rope_head_dim=32, qk_nope_head_dim=32, v_head_dim=64,
        num_hidden_layers=2, first_k_dense_replace=1,
        max_position_embeddings=4096, rms_norm_eps=1e-6, rope_theta=10000.0,
        torch_dtype=dtype, use_cache=False, _attn_implementation="eager",
    )
    return _build_lm(cfg, DeepseekV3Model, batch, seq_len, dtype)


def build_llama4(batch=1, seq_len=32, dtype=torch.float32):
    from transformers.models.llama4.configuration_llama4 import Llama4TextConfig
    from transformers.models.llama4.modeling_llama4 import Llama4TextModel
    cfg = Llama4TextConfig(
        vocab_size=4096, hidden_size=1024, num_attention_heads=8, num_key_value_heads=4,
        intermediate_size=3072, intermediate_size_mlp=3072,
        num_local_experts=4, num_experts_per_tok=1, num_hidden_layers=2, interleave_moe_layer_step=2,
        max_position_embeddings=4096, rms_norm_eps=1e-6, rope_theta=10000.0,
        torch_dtype=dtype, use_cache=False, _attn_implementation="eager",
    )
    return _build_lm(cfg, Llama4TextModel, batch, seq_len, dtype)


def build_glm4(batch=1, seq_len=32, dtype=torch.float32):
    from transformers.models.glm4.configuration_glm4 import Glm4Config
    from transformers.models.glm4.modeling_glm4 import Glm4Model
    cfg = Glm4Config(
        vocab_size=4096, pad_token_id=0, bos_token_id=1, eos_token_id=2,
        hidden_size=1536, num_attention_heads=12, num_key_value_heads=2,
        intermediate_size=4096, num_hidden_layers=2, max_position_embeddings=4096,
        rms_norm_eps=1e-5, rope_theta=10000.0, torch_dtype=dtype, use_cache=False,
        _attn_implementation="eager",
    )
    return _build_lm(cfg, Glm4Model, batch, seq_len, dtype)


def build_olmo2(batch=1, seq_len=32, dtype=torch.float32):
    from transformers.models.olmo2.configuration_olmo2 import Olmo2Config
    from transformers.models.olmo2.modeling_olmo2 import Olmo2Model
    cfg = Olmo2Config(
        vocab_size=4096, hidden_size=2048, num_attention_heads=16, num_key_value_heads=16,
        intermediate_size=8192, num_hidden_layers=2, max_position_embeddings=4096,
        rms_norm_eps=1e-6, rope_theta=10000.0, torch_dtype=dtype, use_cache=False,
        _attn_implementation="eager",
    )
    return _build_lm(cfg, Olmo2Model, batch, seq_len, dtype)


def build_granite(batch=1, seq_len=32, dtype=torch.float32):
    from transformers.models.granite.configuration_granite import GraniteConfig
    from transformers.models.granite.modeling_granite import GraniteModel
    cfg = GraniteConfig(
        vocab_size=4096, hidden_size=2048, num_attention_heads=16, num_key_value_heads=8,
        intermediate_size=8192, num_hidden_layers=2, max_position_embeddings=4096,
        rms_norm_eps=1e-5, rope_theta=10000.0, torch_dtype=dtype, use_cache=False,
        _attn_implementation="eager",
    )
    return _build_lm(cfg, GraniteModel, batch, seq_len, dtype)


def build_phimoe(batch=1, seq_len=32, dtype=torch.float32):
    from transformers.models.phimoe.configuration_phimoe import PhimoeConfig
    from transformers.models.phimoe.modeling_phimoe import PhimoeModel
    cfg = PhimoeConfig(
        vocab_size=4096, hidden_size=1024, num_attention_heads=8, num_key_value_heads=4,
        intermediate_size=3072, num_local_experts=4, num_experts_per_tok=2,
        num_hidden_layers=2, max_position_embeddings=4096,
        rms_norm_eps=1e-5, rope_theta=10000.0, torch_dtype=dtype, use_cache=False,
        _attn_implementation="eager",
    )
    return _build_lm(cfg, PhimoeModel, batch, seq_len, dtype)


def build_mamba2(batch=1, seq_len=32, dtype=torch.float32):
    # State-space model: no attention, no RoPE -- completely different op profile.
    # Invariant: num_heads * head_dim == intermediate_size == expand * hidden_size
    # (modeling_mamba2.py:171 + the view(B, num_heads*head_dim) at line 365).
    from transformers.models.mamba2.configuration_mamba2 import Mamba2Config
    from transformers.models.mamba2.modeling_mamba2 import Mamba2Model
    cfg = Mamba2Config(
        vocab_size=4096, hidden_size=512,
        num_heads=16, head_dim=64,
        state_size=16, chunk_size=16,
        expand=2, n_groups=1,
        num_hidden_layers=2, torch_dtype=dtype, use_cache=False,
    )
    model = Mamba2Model(cfg).eval().to(dtype=dtype)
    input_ids = torch.randint(0, cfg.vocab_size, (batch, seq_len))
    # Mamba has no attention mask; pass none.
    return model, {"input_ids": input_ids}


def build_mllama(batch=1, seq_len=32, dtype=torch.float32):
    # Llama 3.2 Vision -- text branch only (text-only call path).
    # MllamaRotaryEmbedding requires config.rope_scaling["rope_type"]; pass default.
    from transformers.models.mllama.configuration_mllama import MllamaTextConfig
    from transformers.models.mllama.modeling_mllama import MllamaTextModel
    cfg = MllamaTextConfig(
        vocab_size=4096, pad_token_id=0, bos_token_id=1, eos_token_id=2,
        hidden_size=1024, num_attention_heads=8, num_key_value_heads=4,
        intermediate_size=3072, num_hidden_layers=2,
        cross_attention_layers=[],
        max_position_embeddings=4096, rms_norm_eps=1e-5, rope_theta=10000.0,
        rope_scaling={"rope_type": "default"},
        torch_dtype=dtype, use_cache=False, _attn_implementation="eager",
    )
    return _build_lm(cfg, MllamaTextModel, batch, seq_len, dtype)


BUILDERS = {
    "qwen2": build_qwen2,
    "gemma": build_gemma,
    "gemma2": build_gemma2,
    "phi3": build_phi3,
    # Models newly available with transformers 4.51.3
    "qwen3": build_qwen3,
    "qwen3_moe": build_qwen3_moe,
    "gemma3": build_gemma3,
    "deepseek_v3": build_deepseek_v3,
    "llama4": build_llama4,
    "glm4": build_glm4,
    "olmo2": build_olmo2,
    "granite": build_granite,
    "phimoe": build_phimoe,
    "mamba2": build_mamba2,
    "mllama": build_mllama,
}


# ---------------------------------------------------------------------------
# Phase 1: enumerate aten ops by intercepting the FX graph from torch.compile.
# ---------------------------------------------------------------------------

def _node_op_name(target):
    # OpOverload / OpOverloadPacket: has a .name() method returning "aten::mm.default" etc.
    if hasattr(target, "name") and callable(target.name):
        try:
            return target.name()
        except Exception:
            pass
    if hasattr(target, "_schema"):
        try:
            return str(target._schema.name) + (
                "." + target._schema.overload_name if target._schema.overload_name else ""
            )
        except Exception:
            pass
    # torch.* python builtins: use their __module__/__qualname__
    mod = getattr(target, "__module__", "")
    qn = getattr(target, "__qualname__", None) or getattr(target, "__name__", "")
    if mod and qn:
        return f"{mod}.{qn}"
    return str(target)


@torch.no_grad()
def enumerate_ops(model, inputs):
    """Capture the post-AOTAutograd aten graph(s) via aot_module_simplified.

    This is the same level of IR TOGSim/Inductor consumes, so the op set
    matches what the NPU backend actually has to lower.
    """
    from functorch.compile import aot_module_simplified

    seen = set()
    graph_sizes = []

    def fw_compiler(gm, example_inputs):
        graph_sizes.append(sum(1 for _ in gm.graph.nodes))
        for node in gm.graph.nodes:
            if node.op == "call_function":
                seen.add(_node_op_name(node.target))
        return gm.forward

    def dynamo_backend(gm, example_inputs):
        return aot_module_simplified(gm, example_inputs, fw_compiler=fw_compiler)

    torch._dynamo.reset()
    compiled = torch.compile(model, backend=dynamo_backend, dynamic=False)
    compiled(**inputs)
    return sorted(seen), graph_sizes


# ---------------------------------------------------------------------------
# Phase 2: real NPU compile + run. Capture and parse failure tracebacks.
# ---------------------------------------------------------------------------

ATEN_RE = re.compile(r"aten[.:][a-zA-Z_][a-zA-Z0-9_.]*")
NOTIMPL_RE = re.compile(r"NotImplementedError[: ]+(.*)")


def parse_failure(tb_text):
    aten_hits = []
    for m in ATEN_RE.finditer(tb_text):
        op = m.group(0).replace("aten:", "aten.").lstrip(".")
        if op not in aten_hits:
            aten_hits.append(op)
    msg = ""
    nm = NOTIMPL_RE.search(tb_text)
    if nm:
        msg = nm.group(1).strip().splitlines()[0]
    return aten_hits, msg


@torch.no_grad()
def run_on_npu(model, inputs):
    device = torch.device("npu:0")
    model = model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    torch._dynamo.reset()
    compiled = torch.compile(model, dynamic=False)
    out = compiled(**inputs)
    # touch the output to force completion
    if hasattr(out, "last_hidden_state"):
        out.last_hidden_state.cpu()
    elif isinstance(out, torch.Tensor):
        out.cpu()
    return "OK", None, None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_model(name, args, out_dir):
    builder = BUILDERS[name]
    log_path = os.path.join(out_dir, f"{name}.log")
    with open(log_path, "w") as fh:
        def w(s=""):
            print(s)
            fh.write(s + "\n")

        w(f"=== {name} ===")
        w(f"batch={args.batch}  seq_len={args.seq_len}  dtype={args.dtype}")

        try:
            model, inputs = builder(args.batch, args.seq_len, _DTYPE_MAP[args.dtype])
        except Exception as e:
            w(f"[BUILD FAIL] {type(e).__name__}: {e}")
            return {"name": name, "status": "BUILD_FAIL", "ops": [], "fail_op": str(e)}

        # Phase 1
        w("\n[Phase 1] FX op enumeration (eager backend, no NPU)")
        try:
            ops, graph_sizes = enumerate_ops(model, inputs)
            w(f"  graphs: {len(graph_sizes)}  total_nodes_per_graph: {graph_sizes}")
            w(f"  unique aten ops: {len(ops)}")
            for op in ops:
                w(f"    {op}")
        except Exception:
            tb = traceback.format_exc()
            w("[Phase 1 FAIL]\n" + tb)
            ops = []

        if args.enumerate_only:
            return {"name": name, "status": "ENUM_ONLY", "ops": ops, "fail_op": None}

        # Phase 2
        w("\n[Phase 2] torch.compile on npu:0 + forward")
        try:
            status, fail_op, msg = run_on_npu(model, inputs)
            w(f"  status: {status}")
            return {"name": name, "status": status, "ops": ops, "fail_op": None}
        except Exception:
            tb = traceback.format_exc()
            hits, msg = parse_failure(tb)
            w("  status: FAIL")
            if msg:
                w(f"  NotImplemented message: {msg}")
            if hits:
                w(f"  aten ops in traceback (first = most likely culprit):")
                for h in hits[:10]:
                    w(f"    {h}")
            w("\n----- traceback -----\n" + tb)
            return {
                "name": name,
                "status": "FAIL",
                "ops": ops,
                "fail_op": hits[0] if hits else "?",
                "msg": msg,
            }


_DTYPE_MAP = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=list(BUILDERS.keys()),
                   choices=list(BUILDERS.keys()))
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--dtype", default="float32", choices=list(_DTYPE_MAP.keys()))
    p.add_argument("--enumerate-only", action="store_true",
                   help="Skip NPU compile; just list aten ops per model (fast).")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or os.path.join(
        os.environ.get("TORCHSIM_LOG_PATH", os.path.join(REPO_ROOT, "togsim_results")),
        "op_coverage", ts,
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir: {out_dir}")

    results = []
    for name in args.models:
        try:
            results.append(run_model(name, args, out_dir))
        except KeyboardInterrupt:
            print(f"[interrupt] aborted during {name}")
            break
        except Exception:
            traceback.print_exc()
            results.append({"name": name, "status": "DRIVER_ERR", "ops": [], "fail_op": None})

    # Summary
    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w") as fh:
        def w(s=""):
            print(s)
            fh.write(s + "\n")
        w("\n========== SUMMARY ==========")
        w(f"{'model':10s} {'ops':>5s}  {'status':10s}  first_fail")
        for r in results:
            w(f"{r['name']:10s} {len(r['ops']):>5d}  {r['status']:10s}  {r.get('fail_op') or '-'}")
        # Union & overlap across models
        all_ops = set()
        for r in results:
            all_ops.update(r["ops"])
        w(f"\nUnion of aten ops across all models: {len(all_ops)}")
        w("Per-model op set diff (ops unique to this model):")
        for r in results:
            others = set().union(*(set(r2["ops"]) for r2 in results if r2 is not r))
            unique = sorted(set(r["ops"]) - others)
            w(f"  {r['name']}: {len(unique)} unique")
            for op in unique:
                w(f"    {op}")

    print(f"\nWrote: {summary_path}")


if __name__ == "__main__":
    main()
