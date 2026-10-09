"""Llama 4 on the Triton codegen route.

Built from ``Llama4TextConfig`` with random weights, so it needs no network and
no checkpoint. Needs transformers 4.51.3 -- the module does not exist at 4.43.

WHAT IS NEW, and it is four things rather than one. Llama 4 is the first break
in the family since 3, and every one of these is absent from test_llama3x:

  MoE in EVERY layer    interleave_moe_layer_step defaults to 1, so unlike
                        Mixtral or Qwen-MoE there is no dense layer to fall back
                        on. Scout routes top-1 of 16 experts and adds a SHARED
                        expert whose output is summed in unconditionally.

  NoPE layers           no_rope_layers is [1,1,1,0,...] -- every fourth layer
                        applies no rotary embedding at all. A decoder where the
                        position information is present in some layers and
                        absent in others has no precedent in this suite, and it
                        is a graph difference rather than a constant one.

  QK norm               use_qk_norm puts an RMSNorm on q and k after the rotary,
                        per head rather than per hidden.

  chunked attention     attention_chunk_size bounds how far back a token may
                        attend, which is a second mask shape beside the causal
                        one.

The presets keep all four. Experts are scaled DOWN -- 4 rather than Scout's 16 --
on the same argument tests/models/Qwen/test_qwen.py makes for its MoE presets:
what is unproven is the routing and the combine, and those do not change with
the expert count. Layers stay at 4 because that is the shortest run that
contains a NoPE layer; `scout` names its own no_rope_layers to get one in two.

    source /workspace/torchsim-env.sh
    python tests/models/Llama/test_llama4.py --preset small

MEASURED 2026-08-14 on transformers 5.15.0, tnpu 983eee4, --preset small:
147,874,816 parameters, 31 kernels, 4.7684e-06, 8m00s. All four guards green --
4/4 MoE layers, no_rope_layers [1,1,1,0], experts=4 top-1, qk_norm, chunk=32.

Fifteen ops run eager: 11 `fill_` and 4 `topk`. The complex arithmetic used to
be the bulk of that list -- 7 `view_as_complex`, 7 `mul.out`, 6 `view_as_real`,
twenty calls leaving the simulator -- and `rewrite_fx_graph` (84ad277)
now keeps all of it in the graph. That pass was written and measured against
DeepSeek-V2's rope, and it covers this one unchanged: Llama 4's complex path
uses no op outside the set it already knew. The kernel count went 30 -> 31 as
the complex work became a kernel rather than a fallback.

SCOPE, AND IT IS A REAL LIMIT rather than a shrug. The gate runs `small`, 1024
hidden. `--preset scout` builds Scout's real width (5120 hidden, 40/8 heads,
1,394,672,640 parameters), passes all four guards, compiles 30 kernels, and
then DIES IN SPIKE at kernel 19:

    Kernel store segfault @ 0x0000000000000000        a0 0000000000000000
                                                      s4 0000000050000000

That is not this model's arithmetic. s4 is 0x50000000 = 1,342,177,280, exactly
the byte size of the MoE expert stack (4 x 5120 x 8192 x 2 f32), and a0 is the
null it was told to store through. tnpu's wrapper.py sizes every buffer

    padded = ((nbytes + 63) // 64) * 64 * 2

-- doubled, to keep a DMA tail write off the heap -- so the 1.25 GiB argument is
requested as 2.5 GiB, `calloc` returns NULL, and nothing checks it. The DMA tail
is bounded by a tile, not by the tensor, so the factor is the thing to look at
first; the missing NULL check is why this reads as a segfault instead of "1.25
GiB allocation failed". Until that moves, Llama 4's claim here is a shape claim,
not a width claim -- unlike test_llama3x.py, which gates the real 8B block.

THREE WALLS, IN THE ORDER THEY FELL, because none of them was visible until the
one before it moved.

  atomic_add      On transformers 4.51.3 this stopped at kernel 28,
                  ..._scatter_add_view_28, with "unexpected op in ptr sequence"
                  -- triton_shared's PtrAnalysis meeting `tt.atomic_rmw`, an op
                  it has no arm for. No coverage kernel in triton-npu calls
                  `tl.atomic_*` at all, and the two that look like they do
                  (dl/moe_combine, dl/embedding_grad_scatter_add) both AVOID
                  atomics on purpose, one with a read-modify-write and one by
                  resolving the collision as a predicate. 5.15.0's MoE combine
                  no longer emits it.

  polar           The complex-valued rotary. Llama 4 builds its frequencies as
                  complex numbers, which no other model in this suite does, and
                  the strides arrived transposed against what the op expects:
                  "expected size 32==32, stride 64==1 at dim=1". Fixed in
                  PyTorchSim 3fdcab7, which lowers `polar` to a view over a real
                  pair so it never leaves the graph. Adopting torch's own
                  decomposition verbatim does NOT work -- its body mutates
                  views and trips assert_functional_graph.

  the lost base   With polar out of the way the model compiled and answered
                  WRONG, max abs diff 1.31 against a 4.77e-06 tolerance. Not the
                  complex rewrite: with that pass off the diff is 1.3095997
                  against 1.3095998, the same answer with complex computed on
                  the CPU. It was the MoE router. It fills with -inf and
                  scatters the top-k logits in, and `_roles` classified that
                  scattered-into buffer `out` rather than `inout`, so the kernel
                  got a fresh buffer and every unselected expert arrived at 0
                  instead of -inf. sigmoid turned those into 0.5 and top-1
                  routing became a blend of all four. Fixed in PyTorchSim
                  8d8e2ed; tests/ops/misc/test_inplace_partial_write.py pins it.

The third one is the reason this file says what the model does rather than only
what it builds. A router that blends four experts instead of choosing one still
produces a plausible tensor of the right shape, and only a numeric comparison
against the reference calls it.
"""

import argparse
import copy
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from _pytorchsim_utils import test_result


# Experts scaled down, everything that decides the SHAPE of the routing kept.
# `layers` is 4 so no_rope_layers' period lands a NoPE layer inside the run.
_PRESETS = {
    "tiny": dict(layers=4, hidden=256, heads=8, kv_heads=2, intermediate=512,
                 experts=4, vocab=256, seq=16, chunk=32),

    "small": dict(layers=4, hidden=1024, heads=16, kv_heads=4, intermediate=2048,
                  experts=4, vocab=1024, seq=32, chunk=32),

    # Scout's real width and head ratio, two layers so one of them is NoPE, and
    # four experts rather than sixteen.
    "scout": dict(layers=2, hidden=5120, heads=40, kv_heads=8, intermediate=8192,
                  experts=4, vocab=2048, seq=32, chunk=32,
                  no_rope=[1, 0]),

    # MAVERICK IS SCOUT'S MIRROR IMAGE HERE: its real EXPERT COUNT at a reduced
    # width, where `scout` is the real width at a reduced expert count. Between
    # them the two axes are covered once each.
    #
    # Maverick differs from Scout in exactly one config field that reaches the
    # graph -- num_local_experts, 128 against 16. Measured on three builds:
    # raising the count moves only the leading dim of the expert weight
    # (4, 256, 1024) -> (16, 256, 1024) and nothing about the routing or the
    # combine, and `intermediate_size_mlp` moves NOTHING at all (identical
    # parameter count, identical expert tensor) because it sizes a dense MLP and
    # interleave_moe_layer_step = 1 leaves no dense layer to size.
    #
    # So the width is what has to give. At Maverick's real 5120 the expert stack
    # is 128 x 5120 x 8192 x 2 f32 = 40 GiB, which is not a simulator question.
    "maverick": dict(layers=2, hidden=512, heads=8, kv_heads=2, intermediate=1024,
                     experts=128, vocab=1024, seq=32, chunk=32,
                     no_rope=[1, 0]),
}


def _dtype_from_str(name):
    return {"float32": torch.float32, "float16": torch.float16,
            "bfloat16": torch.bfloat16}.get(name, torch.float32)


def _build_config(preset, seq_len):
    from transformers.models.llama4.configuration_llama4 import Llama4TextConfig

    p = _PRESETS[preset]
    seq_len = seq_len if seq_len is not None else p["seq"]
    kw = dict(
        vocab_size=p["vocab"], hidden_size=p["hidden"],
        num_attention_heads=p["heads"], num_key_value_heads=p["kv_heads"],
        intermediate_size=p["intermediate"], intermediate_size_mlp=p["intermediate"],
        num_hidden_layers=p["layers"],
        num_local_experts=p["experts"], num_experts_per_tok=1,
        interleave_moe_layer_step=1,
        max_position_embeddings=max(seq_len, 512), rope_theta=500000.0,
        use_qk_norm=True, attention_chunk_size=p["chunk"],
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
        use_cache=False, attn_implementation="eager",
    )
    if "no_rope" in p:
        kw["no_rope_layers"] = p["no_rope"]
    return Llama4TextConfig(**kw), seq_len


def _assert_character(cfg, model):
    """Fail if the model did not get the four things this file is about."""
    from transformers.models.llama4.modeling_llama4 import Llama4TextMoe

    kinds = [type(l.feed_forward).__name__ for l in model.layers]
    n_moe = sum(isinstance(l.feed_forward, Llama4TextMoe) for l in model.layers)
    assert n_moe == len(model.layers), \
        f"every Llama 4 layer is MoE; got {n_moe} of {len(model.layers)}: {kinds}"

    # A run with no NoPE layer is a Llama 3 decoder with extra steps.
    assert 0 in list(cfg.no_rope_layers), \
        f"preset must contain a NoPE layer; no_rope_layers={list(cfg.no_rope_layers)}"

    assert cfg.use_qk_norm, "preset must keep the q/k norm"
    assert cfg.num_experts_per_tok < cfg.num_local_experts, \
        "top-k must be smaller than the expert count or the routing is dense"
    print(f"guards ok: {n_moe}/{len(model.layers)} MoE layers, "
          f"no_rope_layers={list(cfg.no_rope_layers)}, "
          f"experts={cfg.num_local_experts} top-{cfg.num_experts_per_tok}, "
          f"qk_norm={cfg.use_qk_norm}, chunk={cfg.attention_chunk_size}")


@torch.no_grad()
def run_llama4(device, preset="small", batch=1, seq_len=None, dtype="float32",
               compile_model=True, rtol=1e-2, atol=1e-2):
    from transformers.models.llama4.modeling_llama4 import Llama4TextModel

    torch_dtype = _dtype_from_str(dtype)
    cfg, seq_len = _build_config(preset, seq_len)

    torch.manual_seed(0)
    model_cpu = Llama4TextModel(cfg).to(dtype=torch_dtype).eval()

    print(f"preset={preset} layers={cfg.num_hidden_layers} hidden={cfg.hidden_size} "
          f"heads={cfg.num_attention_heads}/{cfg.num_key_value_heads} "
          f"interm={cfg.intermediate_size} vocab={cfg.vocab_size} seq={seq_len}")
    print("model params:", sum(p.numel() for p in model_cpu.parameters()))
    _assert_character(cfg, model_cpu)

    g = torch.Generator().manual_seed(0)
    input_ids = torch.randint(3, cfg.vocab_size, (batch, seq_len), generator=g,
                              dtype=torch.long)

    cpu_out = model_cpu(input_ids=input_ids).last_hidden_state

    model_npu = copy.deepcopy(model_cpu).to(device).eval()
    if compile_model:
        model_npu = torch.compile(model_npu, dynamic=False)
    npu_out = model_npu(input_ids=input_ids.to(device)).last_hidden_state

    test_result(f"Llama 4 text ({preset})", npu_out, cpu_out, rtol=rtol, atol=atol)
    print("Max diff > ", torch.max(torch.abs(npu_out.cpu() - cpu_out)))
    print("Llama 4 Simulation Done")


# `intermediate` IS NOT FREE: Llama4VisionMLP2's fc1 is
# Linear(intermediate_size, projector_input_dim) and its input is the
# pixel-shuffled tower output, whose channel count is hidden / ratio**2. At the
# shipped ratio of 0.5 that is hidden * 4, so intermediate MUST be hidden * 4 or
# the projector cannot be applied at all:
#
#     RuntimeError: mat1 and mat2 shapes cannot be multiplied
#                   (256x3072 and 5632x4096)
#
# Llama4VisionConfig's own defaults do not satisfy it -- hidden_size 768 against
# intermediate_size 5632 -- because 5632 is Scout's, and Scout's vision hidden is
# 1408. Anything built from the class defaults is a tower that cannot run.
_VISION_PRESETS = {
    "tiny": dict(hidden=256, heads=8, layers=2, intermediate=1024,
                 image=112, patch=14, projector=1024),

    # Scout's real vision width and patch grid: 336/14 = 24, so 576 patches and
    # a 12x12 grid after the shuffle.
    "real": dict(hidden=1408, heads=16, layers=2, intermediate=5632,
                 image=336, patch=14, projector=4096),
}


def _assert_vision_character(cfg, model):
    """Fail if the vision tower did not get the two things this part is about."""
    from transformers.models.llama4.modeling_llama4 import (
        Llama4VisionPixelShuffleMLP, Llama4VisionRotaryEmbedding)

    assert isinstance(model.vision_adapter, Llama4VisionPixelShuffleMLP), \
        "the pixel-shuffle projector is the point of this part"
    assert cfg.pixel_shuffle_ratio != 1.0, \
        f"a shuffle ratio of 1 folds nothing; got {cfg.pixel_shuffle_ratio}"
    assert isinstance(model.rotary_embedding, Llama4VisionRotaryEmbedding), \
        "the 2-D vision rope is the other half of this part"

    grid = cfg.image_size // cfg.patch_size
    assert grid * grid == (model.rotary_embedding.freqs_ci.shape[0] - 1), \
        "the rope table must carry one entry per patch plus the CLS token"
    assert model.rotary_embedding.freqs_ci.is_complex(), \
        "the vision rope is complex; a real table means this is not Llama 4's"
    print(f"guards ok: {grid}x{grid} patches, shuffle={cfg.pixel_shuffle_ratio}, "
          f"rope table {tuple(model.rotary_embedding.freqs_ci.shape)} complex")


@torch.no_grad()
def run_llama4_vision(device, preset="tiny", batch=1, dtype="float32",
                      compile_model=True, rtol=1e-2, atol=1e-2):
    from transformers.models.llama4.configuration_llama4 import Llama4VisionConfig
    from transformers.models.llama4.modeling_llama4 import Llama4VisionModel

    p = _VISION_PRESETS[preset]
    torch_dtype = _dtype_from_str(dtype)
    cfg = Llama4VisionConfig(
        hidden_size=p["hidden"], num_hidden_layers=p["layers"],
        num_attention_heads=p["heads"], intermediate_size=p["intermediate"],
        image_size=p["image"], patch_size=p["patch"],
        projector_input_dim=p["projector"], projector_output_dim=p["projector"],
        vision_output_dim=p["projector"],
    )

    torch.manual_seed(0)
    # NOT `.to(dtype=...)`. Module.to casts complex tensors too, and this
    # tower's rope table is a complex64 BUFFER (freqs_ci, one entry per patch
    # plus the CLS token). Casting it to float32 discards the imaginary half
    # and the rope silently becomes a real multiply -- caught by the guard
    # below, which is what that guard is for.
    model_cpu = Llama4VisionModel(cfg).eval()
    if torch_dtype != torch.float32:
        for mod in model_cpu.modules():
            for name, t in list(mod.named_parameters(recurse=False)):
                if t.is_floating_point():
                    setattr(mod, name, torch.nn.Parameter(t.to(torch_dtype)))
            for name, t in list(mod.named_buffers(recurse=False)):
                if t is not None and t.is_floating_point():
                    setattr(mod, name, t.to(torch_dtype))

    print(f"vision preset={preset} layers={cfg.num_hidden_layers} "
          f"hidden={cfg.hidden_size} heads={cfg.num_attention_heads} "
          f"image={cfg.image_size} patch={cfg.patch_size}")
    print("model params:", sum(p.numel() for p in model_cpu.parameters()))
    _assert_vision_character(cfg, model_cpu)

    g = torch.Generator().manual_seed(0)
    px = torch.randn(batch, cfg.num_channels, cfg.image_size, cfg.image_size,
                     generator=g).to(dtype=torch_dtype)

    cpu_out = model_cpu(px).last_hidden_state

    model_npu = copy.deepcopy(model_cpu).to(device).eval()
    if compile_model:
        model_npu = torch.compile(model_npu, dynamic=False)
    npu_out = model_npu(px.to(device)).last_hidden_state

    test_result(f"Llama 4 vision ({preset})", npu_out, cpu_out, rtol=rtol, atol=atol)
    print("Max diff > ", torch.max(torch.abs(npu_out.cpu() - cpu_out)))
    print("Llama 4 Vision Simulation Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llama 4 on the Triton route")
    parser.add_argument("--part", type=str, default="text",
                        choices=["text", "vision"])
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--no-compile", dest="compile", action="store_false", default=True)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    args = parser.parse_args()

    sys.path.append(os.environ.get("PYTORCHSIM_ROOT_PATH", "/workspace/PyTorchSim"))
    torch.compiler.is_compiling = lambda: True  # FIXME. Same as test_llama.py.

    if args.part == "vision":
        run_llama4_vision(torch.device("npu:0"),
                          preset=args.preset or "tiny", batch=args.batch,
                          dtype=args.dtype, compile_model=args.compile,
                          rtol=args.rtol, atol=args.atol)
    else:
        run_llama4(torch.device("npu:0"), preset=args.preset or "small",
                   batch=args.batch, seq_len=args.seq_len, dtype=args.dtype,
                   compile_model=args.compile, rtol=args.rtol, atol=args.atol)
