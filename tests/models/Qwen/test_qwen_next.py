"""The Qwen families transformers 5.x added, on the Triton codegen route.

`test_qwen.py` covers the four families that existed when it was written --
qwen2, qwen2_moe, qwen3, qwen3_moe -- plus the two VL towers in
`test_qwen_vl.py`.  transformers 5.15.0 ships nine more Qwen model types, and
this file is six of them.  Same shape as the others: random weights, no
network, no checkpoint, no remote code.  Each family is one published checkpoint
(`_CONFIGS` names it) at its real width, vocab and expert count, one layer deep.

WHAT EACH ONE ADDS, and why it is not covered by the four already here:

  * qwen3_vl / qwen3_vl_moe   Qwen3's block under a vision tower, with
                              interleaved M-RoPE.  The LANGUAGE half is what
                              this file runs, built from the text config alone;
                              the vision tower needs pixel inputs and is
                              `test_qwen_vl.py`'s shape of problem, not this one's.
  * qwen3_5 / qwen3_5_moe     A LINEAR-ATTENTION line.  The recurrence unrolls
                              per position, which is why the seq axis matters
                              here more than width.  The one layer kept is the
                              first, a linear-attention layer.  It also puts a
                              Conv1d in the block, which needs
                              postgrad.convolution_1d (conv1d as conv2d)
                              or it falls to an extern kernel this device has
                              no implementation for.
  * qwen3_next                Hybrid attention -- linear layers interleaved with
                              full ones -- so its cache is not an Attention
                              cache and `use_cache=False` is required to build
                              one at all.  Its one layer is the first, linear.
  * qwen2_audio               An audio model.  `--audio` feeds a mel spectrogram
                              so the ENCODER runs -- conv stack, attention,
                              projector -- and the result is spliced into the
                              text sequence.  Without it it runs on text alone,
                              which says nothing about the encoder.  See
                              `run_audio` for the knobs that are not free.

WHAT IS NOT COVERED, and it is a large half:

  * THE OMNI PAIR.  qwen2_5_omni and qwen3_omni_moe are a thinker plus a talker
    under one config, and the thinker cannot be constructed on its own: it reads
    `vision_start_token_id`, which no config in the pair carries until the parent
    builds it.  Building the parent means building the talker's vocoder too.
  * qwen3_asr.  Qwen3-ASR's published config is not written down here, and
    transformers' defaults for it are not that checkpoint.
  * DEPTH.  One layer, as everywhere in this suite.

Weights and inputs are bfloat16 by default (``--dtype float32`` for the old
run).  ``--part layer`` runs the text model's first decoder layer alone
(tests/one_layer.py), on ``--seq S`` or on one ``--decode S`` step after a KV
cache; the linear-attention families build one full_attention_interval period
for it, since this transformers' cache cannot hold linear layers alone, and run
only its first, linear, layer.

THE AUDIO PATH IS NOT GATED YET, and the reason is a pin rather than a result.
Its `..._eq_masked_scatter_unsqueeze_22` needs triton-npu's
`fix/p14-cross-loop-registers` -- the pass there sized a loop from its own body
while a value held across it kept registers llc then had to spill, which spike's
sp remap cannot store.  Gating this before the compiler pin carries
that commit turns CI red on a fix that has landed nowhere it can see.

qwen3_vl_moe and qwen3_5_moe lower their `torch.sort` the way Inductor does, to
a bitonic network -- which this machine now holds. The rewrite that used to
stand in for it is gone; see PyTorchSim's commit retiring extension_counting_sort
for what it cost and what would bring it back.
"""

import argparse
import inspect
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from result_check import check_result, check_routed_rows, f32, f32_reference
import one_layer

# (rtol, atol) per dtype; the bf16 atol is 3x the worst bf16 eager-vs-Inductor CPU diff at S = 128.
_TOL = {"bfloat16": (5e-2, 0.5), "float32": (1e-2, 1e-4)}
_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
# Share of token rows a bf16 router near-tie may send to other experts (check_routed_rows); fp32 none.
_ROUTED = {"bfloat16": 0.05, "float32": 0.0}

FAMILIES = ("qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe", "qwen3_next", "qwen2_audio")

#: The one that has an audio tower, and so accepts `--audio`.
AUDIO = ("qwen2_audio",)

#: The gate runs one of each KIND rather than all six: a linear-attention
#: family, its MoE counterpart, and a plain Qwen3 block under another tower.
_GATE = ("qwen3_5", "qwen3_vl_moe")

#: family -> (config model_type, the checkpoint's published fields). `_config_for` cuts each stack to one
#: layer and changes nothing else. The VL and 3.5 families run their TEXT config, the half this file runs.
_CONFIGS = {
    # Qwen/Qwen3-VL-8B-Instruct, text_config
    "qwen3_vl": ("qwen3_vl_text", dict(
        vocab_size=151936, hidden_size=4096, intermediate_size=12288, num_attention_heads=32,
        num_key_value_heads=8, head_dim=128, max_position_embeddings=262144, rms_norm_eps=1e-6,
        tie_word_embeddings=False, attention_bias=False,
        rope_parameters=dict(rope_type="default", rope_theta=5000000.0, mrope_section=[24, 20, 20],
                             mrope_interleaved=True))),
    # Qwen/Qwen3-VL-30B-A3B-Instruct, text_config
    "qwen3_vl_moe": ("qwen3_vl_moe_text", dict(
        vocab_size=151936, hidden_size=2048, intermediate_size=6144, num_attention_heads=32,
        num_key_value_heads=4, head_dim=128, max_position_embeddings=262144, rms_norm_eps=1e-6,
        tie_word_embeddings=False, attention_bias=False, num_experts=128, num_experts_per_tok=8,
        moe_intermediate_size=768, decoder_sparse_step=1, mlp_only_layers=[],
        rope_parameters=dict(rope_type="default", rope_theta=5000000.0, mrope_section=[24, 20, 20],
                             mrope_interleaved=True))),
    # Qwen/Qwen3.5-9B, text_config: the checkpoint transformers' Qwen3_5TextConfig defaults describe
    "qwen3_5": ("qwen3_5_text", dict(
        vocab_size=248320, hidden_size=4096, intermediate_size=12288, num_attention_heads=16,
        num_key_value_heads=4, head_dim=256, max_position_embeddings=262144, rms_norm_eps=1e-6,
        tie_word_embeddings=False, linear_conv_kernel_dim=4, linear_key_head_dim=128,
        linear_value_head_dim=128, linear_num_key_heads=16, linear_num_value_heads=32,
        full_attention_interval=4,
        rope_parameters=dict(rope_type="default", rope_theta=10000000.0, partial_rotary_factor=0.25,
                             mrope_section=[11, 11, 10], mrope_interleaved=True))),
    # Qwen/Qwen3.5-35B-A3B, text_config
    "qwen3_5_moe": ("qwen3_5_moe_text", dict(
        vocab_size=248320, hidden_size=2048, num_attention_heads=16, num_key_value_heads=2, head_dim=256,
        max_position_embeddings=262144, rms_norm_eps=1e-6, tie_word_embeddings=False,
        linear_conv_kernel_dim=4, linear_key_head_dim=128, linear_value_head_dim=128,
        linear_num_key_heads=16, linear_num_value_heads=32, full_attention_interval=4,
        num_experts=256, num_experts_per_tok=8, moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        rope_parameters=dict(rope_type="default", rope_theta=10000000.0, partial_rotary_factor=0.25,
                             mrope_section=[11, 11, 10], mrope_interleaved=True))),
    # Qwen/Qwen3-Next-80B-A3B-Instruct
    "qwen3_next": ("qwen3_next", dict(
        vocab_size=151936, hidden_size=2048, intermediate_size=5120, num_attention_heads=16,
        num_key_value_heads=2, head_dim=256, max_position_embeddings=262144, rms_norm_eps=1e-6,
        tie_word_embeddings=False, linear_conv_kernel_dim=4, linear_key_head_dim=128,
        linear_value_head_dim=128, linear_num_key_heads=16, linear_num_value_heads=32,
        full_attention_interval=4, num_experts=512, num_experts_per_tok=10, moe_intermediate_size=512,
        shared_expert_intermediate_size=512, norm_topk_prob=True, decoder_sparse_step=1,
        mlp_only_layers=[],
        rope_parameters=dict(rope_type="default", rope_theta=10000000.0, partial_rotary_factor=0.25))),
    # Qwen/Qwen2-Audio-7B: a Whisper-large-v3 encoder under a Qwen-7B text model
    "qwen2_audio": ("qwen2_audio", dict(
        audio_token_index=151646,
        text_config=dict(
            model_type="qwen2", vocab_size=156032, hidden_size=4096, intermediate_size=11008,
            num_attention_heads=32, num_key_value_heads=32, max_position_embeddings=8192,
            rms_norm_eps=1e-5, tie_word_embeddings=False,
            rope_parameters=dict(rope_type="default", rope_theta=10000.0)),
        audio_config=dict(
            d_model=1280, encoder_attention_heads=20, encoder_ffn_dim=5120, num_mel_bins=128,
            max_source_positions=1500, activation_function="gelu", scale_embedding=False))),
}


def _config_for(family, layers=1):
    """This family's published config with each text stack cut to `layers` and the audio encoder to one.

    Linear-attention families keep their published `full_attention_interval`, so the first layer is a
    linear-attention one.
    """
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    model_type, published = _CONFIGS[family]
    if family in AUDIO:
        return CONFIG_MAPPING[model_type](
            **{k: v for k, v in published.items() if k not in ("text_config", "audio_config")},
            text_config=dict(published["text_config"], num_hidden_layers=layers),
            audio_config=dict(published["audio_config"], encoder_layers=1))
    return CONFIG_MAPPING[model_type](**published, num_hidden_layers=layers)


def build(family, layers=1):
    """The model, and which Auto class accepted it."""
    from transformers import AutoModel, AutoModelForCausalLM
    cfg = _config_for(family, layers)
    for maker in (AutoModelForCausalLM, AutoModel):
        try:
            return cfg, maker.from_config(cfg), maker.__name__
        except Exception:
            continue
    raise RuntimeError(f"no Auto class accepted {family}")


def run_family(family, seq_len=128, dtype="bfloat16", rtol=None, atol=None, compile_it=True, part="model",
               decode=False, device="npu"):
    """One layer of the family's text model on `seq_len` random tokens, or with part "layer" that layer
    alone, on that prefill or with `decode` one step at position seq_len - 1 after a KV cache."""
    tol = _TOL.get(dtype, _TOL["float32"])
    rtol, atol = tol[0] if rtol is None else rtol, tol[1] if atol is None else atol
    text = getattr(_config_for(family), "text_config", None) or _config_for(family)
    if seq_len > text.max_position_embeddings:
        raise ValueError(f"seq {seq_len} exceeds {family}'s {text.max_position_embeddings} positions")
    torch.manual_seed(0)
    layers = _CONFIGS[family][1].get("full_attention_interval", 1) if part == "layer" else 1
    cfg, model, how = build(family, layers)
    model.to(_DTYPES[dtype]).eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"family={family} {type(model).__name__} via {how} "
          f"params={n/1e6:.2f}M seq={seq_len}")

    ids = torch.randint(0, 64, (1, seq_len))
    if part == "layer":
        body = model.model if hasattr(model, "model") else model
        cpu_out, npu_out = one_layer.run(body, body.layers[0], cfg, ids, torch.device(device),
                                         decode=decode, compile_model=compile_it)
        row_share = _ROUTED.get(dtype, 0.0) if getattr(text, "num_experts", 0) else 0.0
        check_routed_rows(f"Qwen {family} layer", npu_out.cpu(), cpu_out, rtol, atol, row_share)
        return
    sig = set(inspect.signature(model.forward).parameters)
    kw = {"input_ids": ids}
    if "attention_mask" in sig:
        kw["attention_mask"] = torch.ones_like(ids)
    if "use_cache" in sig:
        # qwen3_next's cache is not an Attention cache, and asking it for a
        # sequence length raises rather than returning zero.
        kw["use_cache"] = False

    def first(out):
        return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

    with torch.no_grad():
        cpu_out = first(f32_reference(model)(**f32(kw)))

    model_npu = model.to(device)
    kw_npu = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in kw.items()}
    runner = torch.compile(model_npu, dynamic=False) if compile_it else model_npu
    with torch.no_grad():
        npu_out = first(runner(**kw_npu)).cpu()

    check_result(f"Qwen {family}", npu_out, cpu_out, rtol=rtol, atol=atol)


def run_audio(family, dtype="bfloat16", rtol=None, atol=None, compile_it=True):
    """The AUDIO tower: a mel spectrogram in, spliced into the text sequence.

    TWO THINGS ARE NOT FREE HERE, and each one cost a run:

      * THE FEATURE LENGTH is validated, not accepted.  qwen2_audio wants
        exactly `max_source_positions * conv1.stride * conv2.stride` frames,
        3000 at the published 1500 positions -- 30 s of audio.
      * THE PLACEHOLDER COUNT must equal the number of audio embeddings, and
        `frames // 2` is not that number.  The tower carries the arithmetic as
        a method; ask it rather than guess -- a guess put 50 tokens against 13
        features.
    """
    tol = _TOL.get(dtype, _TOL["float32"])
    rtol, atol = tol[0] if rtol is None else rtol, tol[1] if atol is None else atol
    torch.manual_seed(0)
    cfg, model, _ = build(family)
    model.to(_DTYPES[dtype]).eval()
    tower = model.audio_tower
    ac = cfg.audio_config
    frames = ac.max_source_positions * 2
    feats = torch.randn(1, ac.num_mel_bins, frames).to(_DTYPES[dtype])
    _, got = tower._get_feat_extract_output_lengths(torch.tensor([frames]))
    n_audio = int(got.flatten()[0])

    tok = cfg.audio_token_index
    ids = torch.cat([torch.full((1, n_audio), tok, dtype=torch.long),
                     torch.randint(0, 16, (1, 8))], dim=1)
    kw = {"input_ids": ids, "input_features": feats,
          "feature_attention_mask": torch.ones(1, frames, dtype=torch.long),
          "attention_mask": torch.ones_like(ids), "use_cache": False}

    print(f"family={family} {type(model).__name__} audio tower "
          f"{type(tower).__name__} mel={ac.num_mel_bins}x{frames} "
          f"audio_tokens={n_audio} at id {tok}")

    def first(out):
        return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

    with torch.no_grad():
        cpu_out = first(f32_reference(model)(**f32(kw)))

    model_npu = model.to("npu")
    kw_npu = {k: (v.to("npu") if torch.is_tensor(v) else v) for k, v in kw.items()}
    runner = torch.compile(model_npu, dynamic=False) if compile_it else model_npu
    with torch.no_grad():
        npu_out = first(runner(**kw_npu)).cpu()

    check_result(f"Qwen {family} audio", npu_out, cpu_out, rtol=rtol, atol=atol)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="The Qwen families transformers 5.x added, on the Triton route")
    parser.add_argument("--family", type=str, default=None, choices=sorted(FAMILIES),
                        help=f"default: the gate, {' and '.join(_GATE)}")
    parser.add_argument("--part", type=str, default="model", choices=["model", "layer"],
                        help="model = the one-layer text model; layer = its decoder layer alone")
    one_layer.add_grid_args(parser)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=sorted(_DTYPES))
    parser.add_argument("--rtol", type=float, default=None, help="default: _TOL of --dtype")
    parser.add_argument("--atol", type=float, default=None, help="default: _TOL of --dtype")
    parser.add_argument("--no-compile", dest="compile", action="store_false", default=True)
    parser.add_argument("--audio", action="store_true",
                        help=f"run the audio tower, for {' and '.join(AUDIO)}")
    args = parser.parse_args()
    if args.decode is not None and args.part != "layer":
        parser.error("--decode runs with --part layer")

    if args.audio:
        for fam in ([args.family] if args.family else AUDIO):
            if fam not in AUDIO:
                raise SystemExit(f"{fam} has no audio tower")
            run_audio(fam, dtype=args.dtype, rtol=args.rtol, atol=args.atol, compile_it=args.compile)
    else:
        for fam in ([args.family] if args.family else _GATE):
            run_family(fam, seq_len=args.decode or args.seq, dtype=args.dtype, rtol=args.rtol, atol=args.atol,
                       compile_it=args.compile, part=args.part, decode=args.decode is not None)
