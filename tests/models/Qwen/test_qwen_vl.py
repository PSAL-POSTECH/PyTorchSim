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

MEASURED 2026-08-13.  The `tiny` rows are triton-npu develop 3434608; the `2b`
row also needs the triton-npu fix below (fix/p13-replicate-one-lane).  Both need
triton_shared with the i64 fix:

    preset  version     text hidden  vision  patches  params        max diff
    tiny    Qwen2-VL      256         128      16      1,825,536    1.2517e-06
    tiny    Qwen2.5-VL    256         128      16      1,792,384    1.0729e-06
    2b      Qwen2-VL     1536        1280      64  114,652,928    4.2021e-06
    2b      Qwen2.5-VL   1536        1280      64  114,676,408    4.3809e-06

WHAT THE SIZED PRESETS NEEDED, since none of it is in this file.  Three
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

  PyTorchSim      at the sized presets Inductor lowered the splice's cumsum as
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

`tiny` is the default because it is the smaller claim, not the easier one: its
scan is one block either way, and it exercises every structural claim above.
`7b` is the same shapes at the 7B text width and is not measured here.
"""

import argparse
import copy
import inspect
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from result_check import check_result


# Depth is one layer on BOTH sides, for the reason the text presets give: the
# tower is the same block 32 times.  Width is what stays honest -- the vision
# tower's real 1280 embed and patch 14 on the sized presets, and the text side
# at the width the shipped model pairs with it.
_PRESETS = {
    # Smoke: small enough to tell a broken test file from a broken backend.
    "tiny": dict(hidden=256, heads=8, kv_heads=2, intermediate=512, vocab=1024,
                 vis_hidden=128, vis_heads=2, vis_interm=256,
                 grid=(1, 4, 4), mrope=(8, 4, 4)),

    # Qwen2-VL-2B / Qwen2.5-VL-3B character: the real vision tower width (1280,
    # 16 heads, patch 14) against the 1536-wide text block.
    "2b": dict(hidden=1536, heads=12, kv_heads=2, intermediate=8960, vocab=4096,
               vis_hidden=1280, vis_heads=16, vis_interm=3420,
               grid=(1, 8, 8), mrope=(16, 24, 24)),

    # Qwen2-VL-7B / Qwen2.5-VL-7B: the real 7B text width under the same tower.
    "7b": dict(hidden=3584, heads=28, kv_heads=4, intermediate=18944, vocab=8192,
               vis_hidden=1280, vis_heads=16, vis_interm=3420,
               grid=(1, 8, 8), mrope=(16, 24, 24)),
}

_IMAGE_TOKEN, _VIDEO_TOKEN, _VISION_START = 1001, 1002, 1003


def _dtype_from_str(name):
    return {"float32": torch.float32, "float16": torch.float16,
            "bfloat16": torch.bfloat16}.get(name, torch.float32)




#: Which model class a preset builds, so `_inputs` can ask what its forward
#: takes without building it twice.  Set by `_build`; the two are always used
#: together and a preset's class does not depend on anything else.
_FORWARD = {}


def _forward_of(preset):
    return _FORWARD.get(preset, lambda **_: None)


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


def _build(version, preset, dtype):
    p = _PRESETS[preset]
    if version == "2":
        from transformers.models.qwen2_vl.configuration_qwen2_vl import (
            Qwen2VLConfig as Config, Qwen2VLVisionConfig as VisionConfig)
        from transformers.models.qwen2_vl.modeling_qwen2_vl import (
            Qwen2VLForConditionalGeneration as Model)
        vision = VisionConfig(depth=1, embed_dim=p["vis_hidden"], num_heads=p["vis_heads"],
                              in_chans=3, patch_size=14, spatial_merge_size=2,
                              temporal_patch_size=2, hidden_size=p["hidden"])
    else:
        from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
            Qwen2_5_VLConfig as Config, Qwen2_5_VLVisionConfig as VisionConfig)
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VLForConditionalGeneration as Model)
        # 2.5 replaces the tower's uniform full attention with windowed
        # attention on all but fullatt_block_indexes.  With one layer, index 0
        # is that full-attention block -- windowed attention over a single
        # window would be the same thing computed the long way round.
        vision = VisionConfig(depth=1, hidden_size=p["vis_hidden"], num_heads=p["vis_heads"],
                              in_chans=3, patch_size=14, spatial_merge_size=2,
                              temporal_patch_size=2, out_hidden_size=p["hidden"],
                              intermediate_size=p["vis_interm"], fullatt_block_indexes=[0])

    # THE TEXT HALF IS SPELLED TWO WAYS AND THIS FILE HAS TO BUILD BOTH.  Up to
    # transformers 4.x a VL config carries the text fields at the top level and
    # a `rope_scaling` dict; from 5.x they live in a `text_config` of their own
    # and the dict is `rope_parameters`.  Same model, same numbers -- so the
    # values below are written once and placed by which spelling the installed
    # version takes, rather than by a version number, which would go stale.
    text = dict(vocab_size=p["vocab"], hidden_size=p["hidden"],
                num_attention_heads=p["heads"], num_key_value_heads=p["kv_heads"],
                intermediate_size=p["intermediate"], num_hidden_layers=1,
                use_cache=False)
    # THREE SPELLINGS, NOT TWO, AND THEY MOVED ONE AT A TIME. 4.51 keeps the
    # text fields on the VL config with a `rope_scaling` dict; 4.57 splits out
    # `text_config` but the dict is still `rope_scaling`; 5.x renames it
    # `rope_parameters`. Asking each question of the signature that answers it
    # -- does the VL config take a text_config, and which name does the TEXT
    # config take -- covers the middle spelling, which a single version test
    # does not: 4.57 was built with rope_parameters and its attention then read
    # `self.rope_scaling["mrope_section"]` off a None.
    # THE INNER KEY MOVED WITH THE OUTER ONE. 4.x spells the kind `type` and its
    # validator normalises "mrope" to "default" while keeping the sections;
    # 5.x spells it `rope_type` and has an mrope entry of its own, and REJECTS
    # the 4.x spelling. Measured on 4.57.6: `rope_type="mrope"` raises
    # `KeyError: 'mrope'` out of ROPE_INIT_FUNCTIONS, `type="mrope"` is taken.
    mrope = list(p["mrope"])
    if "text_config" in inspect.signature(Config.__init__).parameters:
        TextConfig = type(Config().text_config)
        names = inspect.signature(TextConfig.__init__).parameters
        if "rope_parameters" in names:
            text["rope_parameters"] = {"rope_type": "mrope", "mrope_section": mrope,
                                       "rope_theta": 1000000.0}
        else:
            text["rope_scaling"] = {"type": "mrope", "mrope_section": mrope}
        cfg = Config(text_config=text, vision_config=vision.to_dict(),
                     attn_implementation="eager",
                     image_token_id=_IMAGE_TOKEN, video_token_id=_VIDEO_TOKEN,
                     vision_start_token_id=_VISION_START)
    else:
        cfg = Config(
            vision_config=vision.to_dict(), attn_implementation="eager",
            rope_scaling={"type": "mrope", "mrope_section": mrope},
            image_token_id=_IMAGE_TOKEN, video_token_id=_VIDEO_TOKEN,
            vision_start_token_id=_VISION_START, **text)
    torch.manual_seed(0)
    _FORWARD[preset] = Model.forward
    return cfg, Model(cfg).to(dtype=_dtype_from_str(dtype)).eval()


def _inputs(preset, dtype, seed=0):
    """Tokens and patches, shaped the way the processor would hand them over."""
    p = _PRESETS[preset]
    t, h, w = p["grid"]
    n_patches = t * h * w
    # spatial_merge_size 2 merges each 2x2 patch block into one visual token.
    n_tokens = n_patches // 4
    g = torch.Generator().manual_seed(seed)
    pixel = torch.randn(n_patches, 3 * 2 * 14 * 14, generator=g,
                        dtype=_dtype_from_str(dtype))
    ids = torch.tensor([[5, 6, _VISION_START] + [_IMAGE_TOKEN] * n_tokens + [7]])
    got = dict(input_ids=ids, pixel_values=pixel,
               image_grid_thw=torch.tensor([[t, h, w]]))
    # WHICH TOKENS ARE IMAGE, SPELLED OUT.  transformers 5.x asks the caller for
    # it -- text 0, image 1, video 2 -- rather than recovering it from
    # `input_ids == image_token_id`, and refuses M-RoPE without it; 4.x derives
    # it and takes no such argument.  Passed only where the signature has it, so
    # this file builds the same model on both.
    if "mm_token_type_ids" in inspect.signature(_forward_of(preset)).parameters:
        got["mm_token_type_ids"] = (ids == _IMAGE_TOKEN).to(torch.int32)
    return got


def _assert_character(version, cfg, model, preset):
    """Fail if this is not actually the VL model the preset claims."""
    p = _PRESETS[preset]
    vision = cfg.vision_config

    tower = model.visual if hasattr(model, "visual") else model.model.visual
    assert tower is not None, "a VL preset without a vision tower is a text model"
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

    if preset in ("2b", "7b"):
        assert vision.hidden_size if version != "2" else vision.embed_dim
        width = vision.embed_dim if version == "2" else vision.hidden_size
        assert width == 1280, f"the sized presets run the real tower width, got {width}"
        assert vision.patch_size == 14, "the real tower is patch 14"
    if preset == "7b":
        assert text.hidden_size == 3584 and text.intermediate_size == 18944, \
            "7b preset must run at the real 7B text width"


def _logits(out):
    return out.logits if hasattr(out, "logits") else out[0]


@torch.no_grad()
def run_qwen_vl(device, version="2", preset="2b", dtype="float32",
                compile_model=True, rtol=1e-2, atol=1e-2):
    cfg, model_cpu = _build(version, preset, dtype)
    kwargs = _inputs(preset, dtype)

    p = _PRESETS[preset]
    text = _text_of(cfg)
    vis_width = cfg.vision_config.embed_dim if version == "2" else cfg.vision_config.hidden_size
    print(f"version=Qwen{'2' if version == '2' else '2.5'}-VL preset={preset} "
          f"text_hidden={text.hidden_size} heads={text.num_attention_heads}/{text.num_key_value_heads} "
          f"interm={text.intermediate_size} vision={vis_width}x{cfg.vision_config.num_heads}heads "
          f"patches={kwargs['pixel_values'].shape[0]} grid={p['grid']} "
          f"mrope={p['mrope']} vocab={text.vocab_size} dtype={dtype}")
    print("model params:", sum(x.numel() for x in model_cpu.parameters()))

    _assert_character(version, cfg, model_cpu, preset)

    cpu_out = _logits(model_cpu(**kwargs))

    # THE IMAGE MUST REACH THE OUTPUT. A tower whose result never lands in the
    # text sequence would agree with the NPU perfectly and prove nothing, and
    # the splice is exactly the part this file is here for.
    other = _inputs(preset, dtype, seed=1)
    delta = (cpu_out - _logits(model_cpu(**other))).abs().max().item()
    assert delta > 1e-4, \
        (f"changing the pixels moved the logits by {delta:.3e} -- the image is not "
         "reaching the output, so this run would not test the splice")
    print(f"image-reaches-output guard ok: {delta:.4e} on a pixel change")

    model_npu = copy.deepcopy(model_cpu).to(device).eval()
    if compile_model:
        model_npu = torch.compile(model_npu, dynamic=False)
    npu_kwargs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in kwargs.items()}
    npu_out = _logits(model_npu(**npu_kwargs))

    check_result(f"Qwen{'2' if version == '2' else '2.5'}-VL ({preset})",
                npu_out, cpu_out, rtol=rtol, atol=atol)
    print("Max diff > ", torch.max(torch.abs(npu_out.cpu() - cpu_out)))
    print("Qwen-VL Simulation Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen2-VL / Qwen2.5-VL on the Triton route")
    # The default runs BOTH versions at `2b`. They differ in the tower -- 2.5
    # windows its attention and carries its own MLP width -- and the text side
    # they feed is the same, so running one and claiming the other would be the
    # kind of vacuous pass _assert_character exists to prevent.
    parser.add_argument("--version", type=str, default=None, choices=["2", "2.5"])
    # tiny, not 2b: the default is the cheaper claim, not the only passing one.
    # 2b runs the real 1280 vision width and takes about six times as long.
    parser.add_argument("--preset", type=str, default="tiny", choices=sorted(_PRESETS))
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--no-compile", dest="compile", action="store_false", default=True)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    args = parser.parse_args()

    sys.path.append(os.environ.get("TORCHSIM_DIR", "/workspace/PyTorchSim"))
    torch.compiler.is_compiling = lambda: True  # FIXME. Same as test_llama.py.

    for version in ([args.version] if args.version else ["2", "2.5"]):
        run_qwen_vl(
            torch.device("npu:0"),
            version=version,
            preset=args.preset,
            dtype=args.dtype,
            compile_model=args.compile,
            rtol=args.rtol,
            atol=args.atol,
        )
