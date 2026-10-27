"""Qwen2-VL and Qwen2.5-VL on the Triton codegen route.

Built from the ``transformers`` configs with random weights: no checkpoint, no
network, and no image file -- the pixel input is the patch tensor the processor
would have produced, which is what the model actually consumes.

A VL model is not the text model with an encoder bolted on the front, and the
three things that make it different are the three this file exists for:

  * THE PATCHES ARE A FLAT SEQUENCE, not a grid.  ``pixel_values`` arrives as
    (num_patches, channels * temporal_patch * patch * patch) with the layout
    carried separately in ``image_grid_thw``, so the tower's first op is a
    3D convolution over a sequence rather than a 2D one over an image.

  * M-ROPE.  ``rope_scaling={"mrope_section": [...]}`` splits head_dim/2 into
    three bands -- temporal, height, width -- and each band takes its position
    from a different axis.  Inductor lowers the band selection to
    ``triton_helpers.select_one``, which the compiler takes from torch with the
    kernel's module.

  * MASKED_SCATTER.  The image embeddings are spliced into the text sequence
    at the image-token positions.  Inductor lowers that to a running index --
    a ``tl.associative_scan`` carried across the reduction loop in int64 --
    used as a gather offset.  THAT is what blocked this file: triton_shared's
    ``tts.get_structured_state`` accepted only i32 tensors while the prepass
    that builds it wraps every int tensor, so the whole kernel was refused.
    Fixed in triton_shared (fix/structured-state-i64-offsets); with the old
    binary this test stops at stage 3 and says so.

``_assert_character`` cashes each of those against the built model, and the
image is proved to REACH the output rather than assumed to: the same model is
run twice with different pixels and the logits must differ.  Without that a
tower that quietly contributed nothing would pass.

    source /workspace/torchsim-env.sh
    python tests/models/Qwen/test_qwen_vl.py

Each version is its published checkpoint's config at one text layer and one vision layer:
Qwen2-VL-2B-Instruct and Qwen2.5-VL-3B-Instruct, real vocab, widths, heads, M-RoPE and tower.
The image is a 1x4x4 patch grid (4 visual tokens) inside ``--seq`` tokens of text, batch 1, prefill.

WHAT THE 1280-WIDE TOWER NEEDED, since none of it is in this file.  Three
toolchain changes, in the order they were hit:

  triton_shared   `tts.get_structured_state` accepted i32 offset tensors while
                  the prepass that builds it wraps every int tensor, so the
                  splice's int64 running index failed the whole kernel.
                  Widened to the i32-or-i64 constraint that file already has.

  triton-npu      `_replicate_computed_row` chose `replicate_axes` by matching
                  shapes, and a rank-2 row against a rank-4 tile matched none
                  of its three arms -- so transformers' image-token COUNT (not
                  the model's arithmetic: the check that placeholders match
                  embeddings) was refused by bank_vectorize with "one operand
                  ... is in a single bank (ONE_LANE) while another is banked
                  across the lanes on iteration dim 0 of extent 128".  The
                  correspondence was in the operand's map; it reads it there
                  now.  Qwen2-Audio's audio-token count is the same shape.

  PyTorchSim      at the real tower width Inductor lowered the splice's cumsum as
                  a DECOUPLED-LOOKBACK split scan, where one block spins on
                  another's published state.  This route compiles one ELF and
                  walks the grid in its own wrapper, so there is no such
                  channel, and the kernel was refused: "kernel uses
                  triton_helpers.{exclusive_scan_decoupled_lookback_64}, which
                  lives in torch and the tnpu venv has no torch".  It is
                  answered where Inductor asks -- NPUChoices.reduction_split_
                  factor pins the split at 1, because splitting buys
                  parallelism across blocks that run at the same time.  The
                  scan is then the ordinary looped form, associative_scan plus
                  a carry, which this route already runs.

THERE USED TO BE AN --eager-splice FLAG HERE and it is gone.  It declined
MASKED_SCATTER_WITH_INDEX so the splice fell back to aten -- right values, no
kernel, and nothing simulated.  A CPU fallback is a defect (see CLAUDE.md), and
keeping an escape hatch to one is keeping the defect.
"""

import argparse
import copy
import inspect
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from result_check import check_result, f32, f32_reference


#: Published config per version: per-layer sizes as shipped, depth cut to `_LAYERS` on both sides.
_MODELS = {
    # Qwen/Qwen2-VL-2B-Instruct
    "2": dict(
        text=dict(vocab_size=151936, hidden_size=1536, num_attention_heads=12, num_key_value_heads=2,
                  intermediate_size=8960, hidden_act="silu", max_position_embeddings=32768,
                  rms_norm_eps=1e-6, use_sliding_window=False, attention_dropout=0.0),
        rope=dict(rope_theta=1000000.0, mrope_section=[16, 24, 24]),
        vision=dict(embed_dim=1280, hidden_size=1536, mlp_ratio=4, num_heads=16, hidden_act="quick_gelu",
                    in_chans=3, patch_size=14, spatial_merge_size=2, temporal_patch_size=2),
        tie_word_embeddings=True),
    # Qwen/Qwen2.5-VL-3B-Instruct
    "2.5": dict(
        text=dict(vocab_size=151936, hidden_size=2048, num_attention_heads=16, num_key_value_heads=2,
                  intermediate_size=11008, hidden_act="silu", max_position_embeddings=128000,
                  rms_norm_eps=1e-6, use_sliding_window=False, attention_dropout=0.0),
        rope=dict(rope_theta=1000000.0, mrope_section=[16, 24, 24]),
        vision=dict(hidden_size=1280, intermediate_size=3420, num_heads=16, hidden_act="silu",
                    in_chans=3, patch_size=14, spatial_merge_size=2, temporal_patch_size=2,
                    out_hidden_size=2048, window_size=112, fullatt_block_indexes=[7, 15, 23, 31],
                    tokens_per_second=2),
        tie_word_embeddings=True),
}

_LAYERS = 1
_GRID = (1, 4, 4)

# (rtol, atol) per dtype; the bf16 atol is 3x the worst bf16 eager-vs-Inductor CPU diff at S = 128.
_TOL = {"bfloat16": (5e-2, 0.2), "float32": (1e-2, 1e-2)}

_IMAGE_TOKEN, _VIDEO_TOKEN, _VISION_START, _VISION_END = 151655, 151656, 151652, 151653


def _dtype_from_str(name):
    return {"float32": torch.float32, "float16": torch.float16,
            "bfloat16": torch.bfloat16}.get(name, torch.float32)




#: Which model class a version builds, so `_inputs` can ask what its forward
#: takes without building it twice.  Set by `_build`; the two are always used together.
_FORWARD = {}


def _forward_of(version):
    return _FORWARD.get(version, lambda **_: None)


def _text_of(cfg):
    """The config that carries the text fields, in either spelling.

    4.x puts them on the VL config itself; 5.x moves them into `text_config`.
    Everything that asks "how wide is the text side" goes through here so the
    question is asked once.
    """
    return getattr(cfg, "text_config", cfg)


def _rope_of(cfg):
    """The rope dict, `rope_parameters` (5.x) or `rope_scaling` (4.x)."""
    t = _text_of(cfg)
    for name in ("rope_parameters", "rope_scaling"):
        got = getattr(t, name, None)
        if got:
            return got
    return {}


def _build(version, dtype):
    """The version's published config at `_LAYERS` text and vision layers, with random weights.

    2.5 windows the tower's attention on all but fullatt_block_indexes [7, 15, 23, 31], so the one
    kept vision layer (index 0) is a windowed block, as in the shipped model.

    THE TEXT HALF IS SPELLED THREE WAYS, AND THEY MOVED ONE AT A TIME.  4.51 keeps the text fields
    on the VL config with a `rope_scaling` dict; 4.57 splits out `text_config` but the dict is still
    `rope_scaling`; 5.x renames it `rope_parameters`.  Each question is asked of the signature that
    answers it -- does the VL config take a text_config, and which name does the TEXT config take --
    which covers the middle spelling a single version test does not: 4.57 built with rope_parameters
    read `self.rope_scaling["mrope_section"]` off a None.

    THE INNER KEY MOVED WITH THE OUTER ONE.  4.x spells the kind `type` and its validator
    normalises "mrope" to "default" while keeping the sections; 5.x spells it `rope_type`, has an
    mrope entry of its own, and REJECTS the 4.x spelling.  Measured on 4.57.6: `rope_type="mrope"`
    raises `KeyError: 'mrope'` out of ROPE_INIT_FUNCTIONS, `type="mrope"` is taken.
    """
    m = _MODELS[version]
    if version == "2":
        from transformers.models.qwen2_vl.configuration_qwen2_vl import (
            Qwen2VLConfig as Config, Qwen2VLVisionConfig as VisionConfig)
        from transformers.models.qwen2_vl.modeling_qwen2_vl import (
            Qwen2VLForConditionalGeneration as Model)
    else:
        from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
            Qwen2_5_VLConfig as Config, Qwen2_5_VLVisionConfig as VisionConfig)
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VLForConditionalGeneration as Model)
    vision = VisionConfig(depth=_LAYERS, **m["vision"])

    text = dict(m["text"], num_hidden_layers=_LAYERS, use_cache=False)
    theta, mrope = m["rope"]["rope_theta"], list(m["rope"]["mrope_section"])
    tokens = dict(image_token_id=_IMAGE_TOKEN, video_token_id=_VIDEO_TOKEN,
                  vision_start_token_id=_VISION_START, vision_end_token_id=_VISION_END)
    if "text_config" in inspect.signature(Config.__init__).parameters:
        TextConfig = type(Config().text_config)
        names = inspect.signature(TextConfig.__init__).parameters
        if "rope_parameters" in names:
            text["rope_parameters"] = {"rope_type": "mrope", "mrope_section": mrope,
                                       "rope_theta": theta}
        else:
            text["rope_scaling"] = {"type": "mrope", "mrope_section": mrope}
            text["rope_theta"] = theta
        cfg = Config(text_config=text, vision_config=vision.to_dict(),
                     attn_implementation="eager", tie_word_embeddings=m["tie_word_embeddings"],
                     **tokens)
    else:
        cfg = Config(
            vision_config=vision.to_dict(), attn_implementation="eager",
            rope_scaling={"type": "mrope", "mrope_section": mrope}, rope_theta=theta,
            tie_word_embeddings=m["tie_word_embeddings"], **tokens, **text)
    torch.manual_seed(0)
    _FORWARD[version] = Model.forward
    return cfg, Model(cfg).to(dtype=_dtype_from_str(dtype)).eval()


def _inputs(version, seq, dtype, seed=0):
    """Tokens and patches, shaped the way the processor would hand them over.

    `seq` tokens in all: vision start, the image tokens (each 2x2 patch block of the grid merges
    into one), vision end, then text.  transformers 5.x asks the caller which tokens are image
    (`mm_token_type_ids`: text 0, image 1, video 2) and refuses M-RoPE without it; 4.x derives it
    and takes no such argument, so it is passed only where the signature has it.
    """
    m = _MODELS[version]
    vis = m["vision"]
    t, h, w = _GRID
    n_patches = t * h * w
    n_tokens = n_patches // vis["spatial_merge_size"] ** 2
    if seq < n_tokens + 3:
        raise ValueError(f"--seq {seq} is too short for {n_tokens} image tokens plus vision start/end "
                         "and one text token")
    if seq > m["text"]["max_position_embeddings"]:
        raise ValueError(f"--seq {seq} exceeds max_position_embeddings "
                         f"{m['text']['max_position_embeddings']}")
    g = torch.Generator().manual_seed(seed)
    patch = vis["in_chans"] * vis["temporal_patch_size"] * vis["patch_size"] ** 2
    pixel = torch.randn(n_patches, patch, generator=g, dtype=_dtype_from_str(dtype))
    text_ids = torch.randint(0, _VISION_START, (seq - n_tokens - 2,),
                             generator=torch.Generator().manual_seed(0))
    ids = torch.cat([torch.tensor([_VISION_START] + [_IMAGE_TOKEN] * n_tokens + [_VISION_END]),
                     text_ids]).unsqueeze(0)
    got = dict(input_ids=ids, pixel_values=pixel,
               image_grid_thw=torch.tensor([[t, h, w]]))
    if "mm_token_type_ids" in inspect.signature(_forward_of(version)).parameters:
        got["mm_token_type_ids"] = (ids == _IMAGE_TOKEN).to(torch.int32)
    return got


def _assert_character(version, cfg, model):
    """Fail if this is not actually the VL model the version claims."""
    m = _MODELS[version]
    vision = cfg.vision_config

    tower = model.visual if hasattr(model, "visual") else model.model.visual
    assert tower is not None, "a VL model without a vision tower is a text model"
    assert vision.spatial_merge_size == 2, "the 2x2 spatial merge is part of the shape"
    assert getattr(vision, "temporal_patch_size", None) == 2, \
        "patches are 3D (temporal_patch_size 2); a 2D patchifier is a different model"

    text = _text_of(cfg)
    sections = _rope_of(cfg)["mrope_section"]
    head_dim = text.hidden_size // text.num_attention_heads
    assert len(sections) == 3, f"M-RoPE has three bands (t, h, w); got {sections}"
    assert sum(sections) == head_dim // 2, \
        (f"the M-RoPE bands must cover head_dim/2 = {head_dim // 2}, got {sum(sections)} "
         "-- a mismatch here silently changes which positions rope sees")

    for name, want in m["text"].items():
        assert getattr(text, name) == want, f"text {name}: {getattr(text, name)} != published {want}"
    width = vision.embed_dim if version == "2" else vision.hidden_size
    assert width == 1280, f"the tower runs the real 1280 width, got {width}"
    assert vision.patch_size == 14, "the real tower is patch 14"


def _logits(out):
    return out.logits if hasattr(out, "logits") else out[0]


@torch.no_grad()
def run_qwen_vl(device, version="2", seq=128, dtype="bfloat16",
                compile_model=True, rtol=None, atol=None):
    """One forward of the version on `_GRID`'s image and `seq` tokens, CPU against `device`, at `_TOL`
    unless given; weights and inputs in `dtype`."""
    tol = _TOL.get(dtype, _TOL["float32"])
    rtol, atol = tol[0] if rtol is None else rtol, tol[1] if atol is None else atol
    cfg, model_cpu = _build(version, dtype)
    kwargs = _inputs(version, seq, dtype)

    text = _text_of(cfg)
    vis_width = cfg.vision_config.embed_dim if version == "2" else cfg.vision_config.hidden_size
    print(f"version=Qwen{'2' if version == '2' else '2.5'}-VL seq={seq} "
          f"text_hidden={text.hidden_size} heads={text.num_attention_heads}/{text.num_key_value_heads} "
          f"interm={text.intermediate_size} vision={vis_width}x{cfg.vision_config.num_heads}heads "
          f"patches={kwargs['pixel_values'].shape[0]} grid={_GRID} "
          f"mrope={_rope_of(cfg)['mrope_section']} vocab={text.vocab_size} dtype={dtype}")
    print("model params:", sum(x.numel() for x in model_cpu.parameters()))

    _assert_character(version, cfg, model_cpu)

    model_ref = f32_reference(model_cpu)
    cpu_out = _logits(model_ref(**f32(kwargs)))

    # THE IMAGE MUST REACH THE OUTPUT. A tower whose result never lands in the
    # text sequence would agree with the NPU perfectly and prove nothing, and
    # the splice is exactly the part this file is here for.
    other = _inputs(version, seq, dtype, seed=1)
    delta = (cpu_out - _logits(model_ref(**f32(other)))).abs().max().item()
    assert delta > 1e-4, \
        (f"changing the pixels moved the logits by {delta:.3e} -- the image is not "
         "reaching the output, so this run would not test the splice")
    print(f"image-reaches-output guard ok: {delta:.4e} on a pixel change")

    model_npu = copy.deepcopy(model_cpu).to(device).eval()
    if compile_model:
        model_npu = torch.compile(model_npu, dynamic=False)
    npu_kwargs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in kwargs.items()}
    npu_out = _logits(model_npu(**npu_kwargs))

    check_result(f"Qwen{'2' if version == '2' else '2.5'}-VL (seq {seq})",
                npu_out, cpu_out, rtol=rtol, atol=atol)
    print("Max diff > ", torch.max(torch.abs(npu_out.cpu() - cpu_out)))
    print("Qwen-VL Simulation Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen2-VL / Qwen2.5-VL on the Triton route")
    # The default runs BOTH versions: their towers differ (2.5 windows its attention and carries
    # its own MLP width), so running one and claiming the other would be a vacuous pass.
    parser.add_argument("--version", type=str, default=None, choices=sorted(_MODELS))
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--no-compile", dest="compile", action="store_false", default=True)
    parser.add_argument("--rtol", type=float, default=None, help="default: _TOL of --dtype")
    parser.add_argument("--atol", type=float, default=None, help="default: _TOL of --dtype")
    args = parser.parse_args()

    sys.path.append(os.environ.get("PYTORCHSIM_ROOT_PATH", "/workspace/PyTorchSim"))
    torch.compiler.is_compiling = lambda: True  # FIXME. Same as test_llama.py.

    for version in ([args.version] if args.version else ["2", "2.5"]):
        run_qwen_vl(
            torch.device("npu:0"),
            version=version,
            seq=args.seq,
            dtype=args.dtype,
            compile_model=args.compile,
            rtol=args.rtol,
            atol=args.atol,
        )
