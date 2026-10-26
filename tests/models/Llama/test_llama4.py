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

The text part is one config: Scout's published text_config with the layer count
cut from 48 to 4, the shortest run that contains a NoPE layer. Every per-layer size
is Scout's -- 5120 hidden, 40/8 heads, head_dim 128, 16 experts top-1 plus the shared
expert, vocab 202048, rope 500000 with llama3 scaling. attention_chunk_size is the
real 8192, so at --seq <= 8192 no token crosses a chunk boundary and the chunked
mask equals the causal one; the chunked layers are still built and traced. The
vision part is Scout's vision tower, two of its 34 layers.

    source /workspace/torchsim-env.sh
    python tests/models/Llama/test_llama4.py --seq 128

SCOPE, AND IT IS AN OPEN LIMIT. A 2-layer, 4-expert build at this width compiled
30 kernels and then DIED IN SPIKE at kernel 19:

    Kernel store segfault @ 0x0000000000000000        a0 0000000000000000
                                                      s4 0000000050000000

s4 is 0x50000000 = 1,342,177,280, exactly the byte size of that MoE expert stack
(4 x 5120 x 8192 x 2 f32), and a0 is the null it was told to store through.
tnpu's wrapper.py sizes every buffer

    padded = ((nbytes + 63) // 64) * 64 * 2

-- doubled, to keep a DMA tail write off the heap -- so the 1.25 GiB argument is
requested as 2.5 GiB, `calloc` returns NULL, and nothing checks it. The real
16-expert stack is four times that (5 GiB, 10 GiB once doubled), so this default
meets the same wrapper limit until the factor or the missing NULL check moves.

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
from result_check import check_result


# meta-llama/Llama-4-Scout-17B-16E text_config; only num_hidden_layers is cut (48 -> 4, one NoPE period).
SCOUT_TEXT = dict(
    vocab_size=202048, hidden_size=5120, num_attention_heads=40, num_key_value_heads=8, head_dim=128,
    intermediate_size=8192, intermediate_size_mlp=16384, num_hidden_layers=4,
    num_local_experts=16, num_experts_per_tok=1, interleave_moe_layer_step=1,
    max_position_embeddings=10485760, rms_norm_eps=1e-5,
    rope_parameters=dict(rope_type="llama3", rope_theta=500000.0, factor=16.0, low_freq_factor=1.0,
                         high_freq_factor=1.0, original_max_position_embeddings=8192),
    no_rope_layer_interval=4, attention_chunk_size=8192, use_qk_norm=True,
    attn_temperature_tuning=True, floor_scale=8192, attn_scale=0.1, attention_bias=False,
    pad_token_id=200018, bos_token_id=200000, eos_token_id=[200001, 200007, 200008],
)


def _dtype_from_str(name):
    return {"float32": torch.float32, "float16": torch.float16,
            "bfloat16": torch.bfloat16}.get(name, torch.float32)


def _build_config(seq_len):
    """Scout's text config; raises if seq_len exceeds its max positions."""
    from transformers.models.llama4.configuration_llama4 import Llama4TextConfig

    if seq_len > SCOUT_TEXT["max_position_embeddings"]:
        raise ValueError(f"seq {seq_len} exceeds max_position_embeddings "
                         f"{SCOUT_TEXT['max_position_embeddings']}")
    return Llama4TextConfig(use_cache=False, attn_implementation="eager", **SCOUT_TEXT)


def _assert_character(cfg, model):
    """Fail if the model did not get the four things this file is about."""
    from transformers.models.llama4.modeling_llama4 import Llama4TextMoe

    kinds = [type(l.feed_forward).__name__ for l in model.layers]
    n_moe = sum(isinstance(l.feed_forward, Llama4TextMoe) for l in model.layers)
    assert n_moe == len(model.layers), \
        f"every Llama 4 layer is MoE; got {n_moe} of {len(model.layers)}: {kinds}"

    # A run with no NoPE layer is a Llama 3 decoder with extra steps.
    assert 0 in list(cfg.no_rope_layers), \
        f"the run must contain a NoPE layer; no_rope_layers={list(cfg.no_rope_layers)}"

    assert cfg.use_qk_norm, "Scout keeps the q/k norm"
    assert cfg.num_experts_per_tok < cfg.num_local_experts, \
        "top-k must be smaller than the expert count or the routing is dense"
    print(f"guards ok: {n_moe}/{len(model.layers)} MoE layers, "
          f"no_rope_layers={list(cfg.no_rope_layers)}, "
          f"experts={cfg.num_local_experts} top-{cfg.num_experts_per_tok}, "
          f"qk_norm={cfg.use_qk_norm}, chunk={cfg.attention_chunk_size}")


@torch.no_grad()
def run_llama4(device, batch=1, seq_len=128, dtype="float32",
               compile_model=True, rtol=1e-2, atol=1e-2):
    from transformers.models.llama4.modeling_llama4 import Llama4TextModel

    torch_dtype = _dtype_from_str(dtype)
    cfg = _build_config(seq_len)

    torch.manual_seed(0)
    model_cpu = Llama4TextModel(cfg).to(dtype=torch_dtype).eval()

    print(f"layers={cfg.num_hidden_layers} hidden={cfg.hidden_size} "
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

    check_result("Llama 4 text (scout)", npu_out, cpu_out, rtol=rtol, atol=atol)
    print("Max diff > ", torch.max(torch.abs(npu_out.cpu() - cpu_out)))
    print("Llama 4 Simulation Done")


# meta-llama/Llama-4-Scout-17B-16E vision_config; only num_hidden_layers is cut (34 -> 2).
SCOUT_VISION = dict(
    hidden_size=1408, num_hidden_layers=2, num_attention_heads=16, intermediate_size=5632,
    image_size=336, patch_size=14, num_channels=3, norm_eps=1e-5, pixel_shuffle_ratio=0.5,
    projector_input_dim=4096, projector_output_dim=4096, vision_output_dim=4096,
    multi_modal_projector_bias=False, rope_parameters=dict(rope_type="default", rope_theta=10000.0),
)


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
def run_llama4_vision(device, batch=1, dtype="float32",
                      compile_model=True, rtol=1e-2, atol=1e-2):
    from transformers.models.llama4.configuration_llama4 import Llama4VisionConfig
    from transformers.models.llama4.modeling_llama4 import Llama4VisionModel

    torch_dtype = _dtype_from_str(dtype)
    cfg = Llama4VisionConfig(**SCOUT_VISION)

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

    print(f"vision layers={cfg.num_hidden_layers} "
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

    check_result("Llama 4 vision (scout)", npu_out, cpu_out, rtol=rtol, atol=atol)
    print("Max diff > ", torch.max(torch.abs(npu_out.cpu() - cpu_out)))
    print("Llama 4 Vision Simulation Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llama 4 on the Triton route")
    parser.add_argument("--part", type=str, default="text",
                        choices=["text", "vision"])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--no-compile", dest="compile", action="store_false", default=True)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    args = parser.parse_args()

    sys.path.append(os.environ.get("PYTORCHSIM_ROOT_PATH", "/workspace/PyTorchSim"))
    torch.compiler.is_compiling = lambda: True  # FIXME. Same as test_llama.py.

    if args.part == "vision":
        run_llama4_vision(torch.device("npu:0"), batch=args.batch,
                          dtype=args.dtype, compile_model=args.compile,
                          rtol=args.rtol, atol=args.atol)
    else:
        run_llama4(torch.device("npu:0"),
                   batch=args.batch, seq_len=args.seq, dtype=args.dtype,
                   compile_model=args.compile, rtol=args.rtol, atol=args.atol)
