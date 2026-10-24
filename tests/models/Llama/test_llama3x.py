"""Llama 3.x on the Triton codegen route.

Built from ``LlamaConfig`` with random weights, so it needs no network and no
checkpoint. ``test_llama3.py`` already covers Llama 3's character at a narrow
width; this file exists for the two things that file cannot say:

  * the real 8B width.  1024 hidden is not any shipped Llama, and it puts
    head_dim at 32 -- a quarter of the 128x128 array.  The tile choice, the
    SPAD pressure and the DRAM traffic at 4096 hidden / 128 head_dim are a
    different problem, and passing at 1024 does not predict it.

  * the 3.2 shapes.  3.2-3B is 24 q heads against 8 kv heads, so repeat_kv
    broadcasts by **3** -- every GQA shape tested so far has been a power of
    two.  3.2-1B ties lm_head to the embedding, which is a buffer-aliasing
    question rather than a kernel one.

Depth is not one of them: the layers are the same block again, so a preset
runs one.  What a second layer would add is the block-to-block seam, and
``test_llama3.py``'s model test already crosses it at the narrow width.

3.1 rides along as a flag rather than a preset.  Its rope scaling is computed
in ``LlamaRotaryEmbedding.__init__`` and lands in the ``inv_freq`` buffer, so
the compiled graph is Llama 3's graph with different constants -- see
``_assert_character``, which fails if a preset claims a feature the model did
not actually get.  Without that assert a 3.1 test passes while proving nothing.

    source /workspace/torchsim-env.sh
    python tests/models/Llama/test_llama3x.py --preset 8b

ALL SEVEN CONFIGURATIONS PASS, measured 2026-08-13 against the same model on
CPU, one cleared cache each so the wall clock is what the sweep would pay.
triton-npu pinned to develop 3434608 -- develop-refactor was mid-edit in
p11_select_lane_axis.py and a pass in that state fails in ways that look local:

    preset      layers  hidden  heads/kv  head_dim  interm  params       kern  max diff    time
    tiny          1       256     8/2        32      1024     1,016,576   22   1.5497e-06   87s
    small         1      1024    16/4        64      3584    14,683,136   22   3.4571e-06  100s
    small +3.1    1      1024    16/4        64      3584    14,683,136   22   3.3155e-06   99s
    1b (lm)       1      1024    16/4        64      4096    16,256,000   23   2.3246e-06  106s
    3b            1      1536    12/4       128      6144    36,180,480   22   4.0531e-06  118s
    8b            1      4096    32/8       128     14336   251,670,528   22   6.0797e-06  275s
    8b (lm)       1      4096    32/8       128     14336   285,224,960   23   8.5831e-06  301s

The kernel count does not move with width -- 22 at 256 hidden and 22 at 4096 --
which is the same observation GPT-2 made about depth, one axis over: one block
is every distinct kernel Llama has, and widening it re-tiles the same kernels
rather than adding new ones.  Time grows 2.75x across a 4x width and a 17x
parameter count, so Spike is not paying linearly for the extra work either.

WHAT THIS FOUND: nothing broken.  n_rep=3 and the tied lm_head were the two
shapes expected to fail and both were already handled.  The value of the file
is now the gate, not the bug it did not find.
"""

import argparse
import copy
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from result_check import check_result


# Presets shrink the model, not the shapes' character.  Width is the axis that
# must stay honest: head_dim, the GQA repeat and the SwiGLU ratio are what make
# a shape the shape it is, and all three ride on hidden.  Layers and vocab are
# the cheap axes -- one block covers every distinct kernel, and the vocab only
# has to be large enough that the gather is a gather.
_PRESETS = {
    # 3.0 character at a narrow width, matching test_llama3.py's ratios but with
    # head_dim at 64 rather than 32.  The bisection width, not the gate.
    "small": dict(layers=1, hidden=1024, heads=16, kv_heads=4,
                  intermediate=3584, vocab=1024, seq=32),

    # Smoke.  Small enough to tell a broken test file from a broken backend.
    "tiny": dict(layers=1, hidden=256, heads=8, kv_heads=2,
                 intermediate=1024, vocab=256, seq=16),

    # 3.2-1B character: head_dim 64 and a tied lm_head.
    "1b": dict(layers=1, hidden=1024, heads=16, kv_heads=4,
               intermediate=4096, vocab=1024, seq=32, tie=True),

    # 3.2-3B character: n_rep = 3.  head_dim stays at the real 128.
    "3b": dict(layers=1, hidden=1536, heads=12, kv_heads=4,
               intermediate=6144, vocab=1024, seq=32),

    # 3/3.1 8B, at its real width.  Only vocab and depth are scaled down.
    "8b": dict(layers=1, hidden=4096, heads=32, kv_heads=8,
               intermediate=14336, vocab=8192, seq=32),
}

# 3.1's rope scaling, as shipped on Llama 3.1 8B.  Applied with --rope31.
_ROPE_31 = {
    "rope_type": "llama3",
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}


def _dtype_from_str(name):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(name, torch.float32)


def _build_config(preset, seq_len, rope31):
    from transformers.models.llama.configuration_llama import LlamaConfig

    p = _PRESETS[preset]
    seq_len = seq_len if seq_len is not None else p["seq"]

    return LlamaConfig(
        vocab_size=p["vocab"],
        hidden_size=p["hidden"],
        num_attention_heads=p["heads"],
        num_key_value_heads=p["kv_heads"],
        intermediate_size=p["intermediate"],
        num_hidden_layers=p["layers"],
        max_position_embeddings=131072 if rope31 else 8192,
        rope_theta=500000.0,
        rope_scaling=_ROPE_31 if rope31 else None,
        rms_norm_eps=1e-5,
        hidden_act="silu",
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=p.get("tie", False),
        use_cache=False,
        # Eager keeps the graph free of the sdpa dispatch, so a failure is about
        # the model and not about which attention kernel transformers picked.
        attn_implementation="eager",
    ), seq_len


def _assert_character(preset, cfg, model, rope31):
    """Fail if the model did not actually get the feature the preset claims.

    Every check here guards against a test that passes without exercising
    anything new.  A preset is a claim about shape; this is where the claim is
    cashed.
    """
    n_rep = cfg.num_attention_heads // cfg.num_key_value_heads
    head_dim = cfg.hidden_size // cfg.num_attention_heads

    if preset == "3b":
        assert n_rep == 3, f"3b preset must exercise a non-power-of-two GQA repeat, got n_rep={n_rep}"
    if preset == "8b":
        assert cfg.hidden_size == 4096, "8b preset must run at the real 8B width"
        assert head_dim == 128, f"8b preset must run at the real head_dim, got {head_dim}"
        assert cfg.intermediate_size == 14336, "8b preset must run at the real SwiGLU width"

    if _PRESETS[preset].get("tie", False):
        body = model.model if hasattr(model, "model") else model
        head = getattr(model, "lm_head", None)
        assert head is not None, "tie check needs the LM head; run this preset with --part lm"
        assert head.weight.data_ptr() == body.embed_tokens.weight.data_ptr(), \
            "tie_word_embeddings was requested but lm_head does not share the embedding storage"

    if rope31:
        # The scaling is folded into inv_freq at construction.  If the config
        # were ignored the buffer would be identical to the unscaled one and
        # the run would prove nothing about 3.1.
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

        plain = copy.deepcopy(cfg)
        plain.rope_scaling = None
        # rope_type is cached on the config object by the rope init path.
        if hasattr(plain, "rope_type"):
            plain.rope_type = "default"
        unscaled = LlamaRotaryEmbedding(config=plain).inv_freq
        scaled = LlamaRotaryEmbedding(config=cfg).inv_freq
        assert not torch.allclose(scaled, unscaled), \
            "rope_scaling was requested but inv_freq is unchanged -- 3.1 is not being exercised"
        print(f"rope31 guard ok: inv_freq max delta "
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
    preset="small",
    part="body",
    batch=1,
    seq_len=None,
    dtype="float32",
    rope31=False,
    compile_model=True,
    rtol=1e-2,
    atol=1e-2,
):
    from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaModel

    torch_dtype = _dtype_from_str(dtype)
    cfg, seq_len = _build_config(preset, seq_len, rope31)

    # Seed before construction: config-random weights otherwise differ per run,
    # so the worst element wanders across the threshold and the test is flaky.
    torch.manual_seed(0)
    cls = LlamaForCausalLM if part == "lm" else LlamaModel
    model_cpu = cls(cfg).to(dtype=torch_dtype).eval()

    head_dim = cfg.hidden_size // cfg.num_attention_heads
    n_rep = cfg.num_attention_heads // cfg.num_key_value_heads
    print(f"preset={preset} part={part} layers={cfg.num_hidden_layers} "
          f"hidden={cfg.hidden_size} heads={cfg.num_attention_heads}/{cfg.num_key_value_heads} "
          f"head_dim={head_dim} n_rep={n_rep} interm={cfg.intermediate_size} "
          f"vocab={cfg.vocab_size} seq={seq_len} dtype={dtype} rope31={rope31}")
    print("model params:", sum(p.numel() for p in model_cpu.parameters()))

    _assert_character(preset, cfg, model_cpu, rope31)

    g = torch.Generator().manual_seed(0)
    input_ids = torch.randint(0, cfg.vocab_size, (batch, seq_len), generator=g, dtype=torch.int64)

    # The explicit 4D causal mask is test_llama3.py's, kept so a failure here is
    # comparable to that file's rather than to whatever _update_causal_mask
    # decided to build.
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
    # THE DEFAULT IS THE GATE: the sweep runs this file with no arguments.
    # It is `8b --part lm` -- one layer of the real 8B block, 4096 hidden and
    # head_dim 128 -- because width is the axis this file exists to hold.
    # GPT-2's gate is already at its real width (base is 768 wide), so gating
    # Llama at 1024 would be a narrower promise than the suite's other models
    # make, not an equal one.  It costs 301s against the sweep's 1800s, and the
    # allowlist already carries an 11-minute entry.
    #
    # `small` is the same shapes at a quarter of the width; reach for it when
    # bisecting, not as the thing CI believes.
    parser.add_argument("--preset", type=str, default="8b", choices=sorted(_PRESETS))
    parser.add_argument("--part", type=str, default="lm", choices=["lm", "body"],
                        help="lm = LlamaForCausalLM (adds the vocab projection); body = LlamaModel")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--rope31", action="store_true",
                        help="apply Llama 3.1's rope scaling (guarded by _assert_character)")
    parser.add_argument("--no-compile", dest="compile", action="store_false", default=True)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    args = parser.parse_args()

    sys.path.append(os.environ.get("PYTORCHSIM_ROOT_PATH", "/workspace/PyTorchSim"))
    torch.compiler.is_compiling = lambda: True  # FIXME. Same as test_llama.py.

    run_llama3x(
        torch.device("npu:0"),
        preset=args.preset,
        part=args.part,
        batch=args.batch,
        seq_len=args.seq_len,
        dtype=args.dtype,
        rope31=args.rope31,
        compile_model=args.compile,
        rtol=args.rtol,
        atol=args.atol,
    )
