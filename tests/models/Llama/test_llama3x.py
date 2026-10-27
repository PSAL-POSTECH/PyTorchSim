"""Llama 3.x on the Triton codegen route.

Built from ``LlamaConfig`` with random weights, so it needs no network and no
checkpoint. Every preset is a shipped model at its published sizes; only the
layer count is cut, to one block. ``test_llama3.py`` runs the same 8B block
without the vocab projection; this file adds:

  * the lm head over the real 128256 vocab, at 4096 hidden / 128 head_dim.

  * the 3.2 shapes.  3.2-3B is 24 q heads against 8 kv heads, so repeat_kv
    broadcasts by **3** -- every other GQA shape in the suite is a power of
    two.  3.2-1B and 3B tie lm_head to the embedding, which is a
    buffer-aliasing question rather than a kernel one.

  * 3.1's rope scaling.  It is computed in ``LlamaRotaryEmbedding.__init__``
    and lands in the ``inv_freq`` buffer, so the compiled graph is Llama 3's
    graph with different constants -- see ``_assert_character``, which fails if
    a preset claims a feature the model did not actually get.

Depth is not one of them: the layers are the same block again, so a preset
runs one. Weights and inputs are bfloat16 by default (``--dtype float32`` for
the old run). ``--part layer`` runs the first decoder layer alone
(tests/one_layer.py), on ``--seq S`` or on one ``--decode S`` step after a KV
cache.

    source /workspace/torchsim-env.sh
    python tests/models/Llama/test_llama3x.py --preset 3.1-8b
    python tests/models/Llama/test_llama3x.py --preset 3.1-8b --part layer --decode 1024
"""

import argparse
import copy
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from result_check import check_result
import one_layer


_LLAMA3_ROPE = dict(rope_type="llama3", low_freq_factor=1.0, high_freq_factor=4.0,
                    original_max_position_embeddings=8192)

#: preset -> the published config of that model, minus num_hidden_layers.
_PRESETS = {
    # meta-llama/Meta-Llama-3-8B
    "3-8b": dict(hidden_size=4096, num_attention_heads=32, num_key_value_heads=8,
                 intermediate_size=14336, vocab_size=128256, max_position_embeddings=8192,
                 rope_theta=500000.0, rope_scaling=None, tie_word_embeddings=False),
    # meta-llama/Llama-3.1-8B
    "3.1-8b": dict(hidden_size=4096, num_attention_heads=32, num_key_value_heads=8,
                   intermediate_size=14336, vocab_size=128256, max_position_embeddings=131072,
                   rope_theta=500000.0, rope_scaling=dict(_LLAMA3_ROPE, factor=8.0),
                   tie_word_embeddings=False),
    # meta-llama/Llama-3.2-1B
    "3.2-1b": dict(hidden_size=2048, num_attention_heads=32, num_key_value_heads=8,
                   head_dim=64, intermediate_size=8192, vocab_size=128256,
                   max_position_embeddings=131072, rope_theta=500000.0,
                   rope_scaling=dict(_LLAMA3_ROPE, factor=32.0), tie_word_embeddings=True),
    # meta-llama/Llama-3.2-3B
    "3.2-3b": dict(hidden_size=3072, num_attention_heads=24, num_key_value_heads=8,
                   head_dim=128, intermediate_size=8192, vocab_size=128256,
                   max_position_embeddings=131072, rope_theta=500000.0,
                   rope_scaling=dict(_LLAMA3_ROPE, factor=32.0), tie_word_embeddings=True),
}

#: The number of layers each preset keeps.
_LAYERS = 1

# (rtol, atol) per dtype; the bf16 atol is 3x the worst bf16 eager-vs-Inductor CPU diff at S = 128.
_TOL = {"bfloat16": (5e-2, 0.6), "float32": (1e-2, 1e-2)}


def _dtype_from_str(name):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(name, torch.float32)


def _build_config(preset, seq_len):
    """The preset's published config at `_LAYERS` blocks; raises if `seq_len` exceeds its positions.

    Eager attention keeps the graph free of the sdpa dispatch, so a failure is
    about the model and not about which attention kernel transformers picked.
    """
    from transformers.models.llama.configuration_llama import LlamaConfig

    p = _PRESETS[preset]
    if seq_len > p["max_position_embeddings"]:
        raise ValueError(f"seq {seq_len} exceeds {preset}'s {p['max_position_embeddings']} positions")
    return LlamaConfig(
        **p,
        num_hidden_layers=_LAYERS,
        rms_norm_eps=1e-5,
        hidden_act="silu",
        attention_bias=False,
        mlp_bias=False,
        bos_token_id=128000,
        eos_token_id=128001,
        use_cache=False,
        attn_implementation="eager",
    )


def _assert_character(cfg, model):
    """Fail if the model did not actually get the feature its config claims.

    Every check here guards against a test that passes without exercising
    anything new.  Rope scaling is folded into inv_freq at construction: if the
    config were ignored the buffer would equal the unscaled one and the run
    would prove nothing about 3.1 or 3.2.
    """
    if cfg.tie_word_embeddings:
        body = model.model if hasattr(model, "model") else model
        head = getattr(model, "lm_head", None)
        assert head is not None, "tie check needs the LM head; run this preset with --part lm"
        assert head.weight.data_ptr() == body.embed_tokens.weight.data_ptr(), \
            "tie_word_embeddings was requested but lm_head does not share the embedding storage"

    if cfg.rope_parameters["rope_type"] != "default":
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

        plain = copy.deepcopy(cfg)
        plain.rope_parameters = {"rope_type": "default", "rope_theta": cfg.rope_parameters["rope_theta"]}
        unscaled = LlamaRotaryEmbedding(config=plain).inv_freq
        scaled = LlamaRotaryEmbedding(config=cfg).inv_freq
        assert not torch.allclose(scaled, unscaled), \
            "rope_scaling was requested but inv_freq is unchanged -- the scaling is not being exercised"
        print(f"rope scaling guard ok: inv_freq max delta "
              f"{(scaled - unscaled).abs().max().item():.6e}")


def _out(output):
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "logits"):
        return output.logits
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, (list, tuple)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Unsupported output type for comparison: {type(output)}")


@torch.no_grad()
def run_llama3x(
    device,
    preset="3.1-8b",
    part="body",
    batch=1,
    seq_len=128,
    dtype="bfloat16",
    compile_model=True,
    rtol=None,
    atol=None,
    decode=False,
):
    """The preset's `part` on a prefill of `seq_len` tokens, or with `decode` (--part layer) one step at
    position seq_len - 1 after a KV cache; CPU against `device`, at `dtype`'s `_TOL` unless given.
    lm and body take test_llama3.py's explicit 4D causal mask, layer the masks the model builds."""
    from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaModel

    torch_dtype = _dtype_from_str(dtype)
    tol = _TOL.get(dtype, _TOL["float32"])
    rtol, atol = tol[0] if rtol is None else rtol, tol[1] if atol is None else atol
    cfg = _build_config(preset, seq_len)

    # Seed before construction: config-random weights otherwise differ per run,
    # so the worst element wanders across the threshold and the test is flaky.
    torch.manual_seed(0)
    cls = LlamaModel if part == "body" else LlamaForCausalLM
    model_cpu = cls(cfg).to(dtype=torch_dtype).eval()

    head_dim = cfg.head_dim
    n_rep = cfg.num_attention_heads // cfg.num_key_value_heads
    print(f"preset={preset} part={part} layers={cfg.num_hidden_layers} "
          f"hidden={cfg.hidden_size} heads={cfg.num_attention_heads}/{cfg.num_key_value_heads} "
          f"head_dim={head_dim} n_rep={n_rep} interm={cfg.intermediate_size} "
          f"vocab={cfg.vocab_size} seq={seq_len} dtype={dtype} "
          f"rope={cfg.rope_parameters['rope_type']}")
    print("model params:", sum(p.numel() for p in model_cpu.parameters()))

    _assert_character(cfg, model_cpu)

    g = torch.Generator().manual_seed(0)
    input_ids = torch.randint(0, cfg.vocab_size, (batch, seq_len), generator=g, dtype=torch.int64)

    if part == "layer":
        cpu_out, npu_out = one_layer.run(model_cpu.model, model_cpu.model.layers[0], cfg, input_ids, device,
                                         decode=decode, compile_model=compile_model)
    else:
        min_dtype = torch.finfo(torch_dtype).min
        causal_mask = torch.full((seq_len, seq_len), fill_value=min_dtype, dtype=torch_dtype)
        if seq_len > 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        attn_mask = causal_mask[None, None, :, :].expand(batch, 1, -1, -1)

        cpu_out = _out(model_cpu(input_ids=input_ids, attention_mask=attn_mask))

        model_npu = copy.deepcopy(model_cpu).to(device).eval()
        if compile_model:
            model_npu = torch.compile(model_npu, dynamic=False)
        npu_out = _out(model_npu(input_ids=input_ids.to(device),
                                 attention_mask=attn_mask.to(device)))

    check_result(f"Llama 3.x {part} ({preset})", npu_out, cpu_out, rtol=rtol, atol=atol)
    print("Max diff > ", torch.max(torch.abs(npu_out.cpu() - cpu_out)))
    print("Llama 3.x Simulation Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llama 3.x end to end on the Triton route")
    # The default is the gate the sweep runs with no arguments: one layer of Llama-3.1-8B plus its lm
    # head, the real 8B width and 3.1's rope scaling in one run.
    parser.add_argument("--preset", type=str, default="3.1-8b", choices=sorted(_PRESETS))
    parser.add_argument("--part", type=str, default="lm", choices=["lm", "body", "layer"],
                        help="lm = LlamaForCausalLM (adds the vocab projection); body = LlamaModel; "
                             "layer = the first decoder layer alone")
    parser.add_argument("--batch", type=int, default=1)
    one_layer.add_grid_args(parser)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--no-compile", dest="compile", action="store_false", default=True)
    parser.add_argument("--rtol", type=float, default=None, help="default: _TOL of --dtype")
    parser.add_argument("--atol", type=float, default=None, help="default: _TOL of --dtype")
    args = parser.parse_args()
    if args.decode is not None and args.part != "layer":
        parser.error("--decode runs with --part layer")

    sys.path.append(os.environ.get("PYTORCHSIM_ROOT_PATH", "/workspace/PyTorchSim"))
    torch.compiler.is_compiling = lambda: True  # FIXME. Same as test_llama.py.

    run_llama3x(
        torch.device("npu:0"),
        preset=args.preset,
        part=args.part,
        batch=args.batch,
        seq_len=args.decode or args.seq,
        decode=args.decode is not None,
        dtype=args.dtype,
        compile_model=args.compile,
        rtol=args.rtol,
        atol=args.atol,
    )
