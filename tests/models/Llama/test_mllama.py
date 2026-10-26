"""Llama 3.2 Vision (Mllama) on the Triton codegen route.

Built from ``MllamaVisionConfig`` / ``MllamaTextConfig`` with random weights, so
it needs no network and no checkpoint. Needs transformers 4.51.3 -- the module
does not exist at 4.43.

WHY THIS IS NOT test_llama3x WITH A VISION TOWER. Mllama is a different model
class, and the two things it adds are the two things no Llama test reaches:

  cross-attention layers   MllamaCrossAttentionDecoderLayer replaces the self
                           attention one at config.cross_attention_layers -- on
                           the real 11B that is every fifth layer of forty. K
                           and V come from the vision tower rather than from
                           the sequence, so the bmm's two operands have
                           unrelated lengths (--seq text against 4 x 1601 vision here).

  a second mask            full_text_row_masked_out_mask zeroes whole text rows
                           that have no image, alongside the usual causal mask.
                           Two mask tensors of different rank meeting in one
                           layer is the part with no precedent in this suite,
                           and the causal mask alone has already taken out a
                           model here once (GPT-2).

The vision tower adds tiles and a local/global layer split on top of a ViT.

Both towers are meta-llama/Llama-3.2-11B-Vision's published configs with only
the layer counts cut: vision 2 local + 1 global layers (of 32 + 8), text 4
layers (of 40) with 2 cross-attention layers. The input is one image of 4 tiles
at 560 px (1601 patches a tile) and --seq text tokens. The vision tower keeps
the real five intermediate taps, each clamped to the kept depth, so its output
stays vision_output_dim = 7680 wide. The text tower's cross layers sit at 1 and
3, where the real model has every fifth layer from 3.

    source /workspace/torchsim-env.sh
    python tests/models/Llama/test_mllama.py --part vision
    python tests/models/Llama/test_mllama.py --part text --seq 128

WHAT THE TEXT TOWER NEEDED, and it was this suite's kind of defect rather than
anything about Mllama. The cross-attention layer leaves a unit middle axis --
tensor<32x1x1024xf32> -- on a single-input elementwise generic whose two maps
are both the identity. Absorbing the move drops that axis, and with no third
operand naming the dim, linalg can no longer infer that loop's bound, so
triton-npu's absorb_layout refused:

    _Fatal: indexing map cannot take the move

The bound it could not infer is 1. Fixed in triton-npu 2988424, which had to
teach that to two places at once: dropping the last bare mention of a dim is
exactly the condition that blinds the extent lookup, so the refusal and the
cleanup below it went blind together. Relaxing only the refusal left
iterator_types three long against maps naming two dims -- IR that does not
verify, and that one-shot-bufferize segfaults on rather than declining.

NOT the tanh gate the failing kernel's name advertised (..._select_tanh_24): a
sibling kernel carries the same scalar gate and compiles through every pass.
Measured by deletion, not read off the name.
"""

import argparse
import copy
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from result_check import check_result


# meta-llama/Llama-3.2-11B-Vision vision_config; layers cut 32 + 8 -> 2 + 1, taps clamped to the kept depth.
VISION = dict(
    hidden_size=1280,
    num_hidden_layers=2,
    num_global_layers=1,
    attention_heads=16,
    intermediate_size=5120,
    image_size=560,
    patch_size=14,
    num_channels=3,
    max_num_tiles=4,
    norm_eps=1e-5,
    intermediate_layers_indices=[min(i, 1) for i in (3, 7, 15, 23, 30)],
    vision_output_dim=7680,
)

# meta-llama/Llama-3.2-11B-Vision text_config; layers cut 40 -> 4, cross-attention at 1 and 3.
TEXT = dict(
    vocab_size=128256,
    hidden_size=4096,
    num_attention_heads=32,
    num_key_value_heads=8,
    intermediate_size=14336,
    num_hidden_layers=4,
    cross_attention_layers=[1, 3],
    max_position_embeddings=131072,
    rms_norm_eps=1e-5,
    rope_parameters=dict(rope_type="llama3", rope_theta=500000.0, factor=8.0, low_freq_factor=1.0,
                         high_freq_factor=4.0, original_max_position_embeddings=8192),
    pad_token_id=128004,
    bos_token_id=128000,
    eos_token_id=128001,
)


def _dtype_from_str(name):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(name, torch.float32)


def _vision_config():
    from transformers.models.mllama.configuration_mllama import MllamaVisionConfig
    return MllamaVisionConfig(_attn_implementation="eager", **VISION)


def _text_config():
    from transformers.models.mllama.configuration_mllama import MllamaTextConfig
    return MllamaTextConfig(use_cache=False, _attn_implementation="eager", **TEXT)


@torch.no_grad()
def run_vision(device, batch=1, dtype="float32", compile_model=True,
               rtol=1e-2, atol=1e-2):
    from transformers.models.mllama.modeling_mllama import MllamaVisionModel

    torch_dtype = _dtype_from_str(dtype)
    cfg = _vision_config()
    torch.manual_seed(0)
    model_cpu = MllamaVisionModel(cfg).to(dtype=torch_dtype).eval()

    tiles = cfg.max_num_tiles
    patches = (cfg.image_size // cfg.patch_size) ** 2
    print(f"[vision] hidden={cfg.hidden_size} local={cfg.num_hidden_layers} "
          f"global={cfg.num_global_layers} tiles={tiles} patches/tile={patches} "
          f"image={cfg.image_size} patch={cfg.patch_size}")
    print("vision params:", sum(p.numel() for p in model_cpu.parameters()))

    assert tiles > 1, "vision preset must keep more than one tile"
    assert cfg.num_global_layers > 0, "vision preset must keep the global layers"

    g = torch.Generator().manual_seed(0)
    pixel_values = torch.randn(batch, 1, tiles, 3, cfg.image_size, cfg.image_size,
                               generator=g, dtype=torch_dtype)
    aspect_ratio_ids = torch.tensor([[1]] * batch, dtype=torch.long)
    aspect_ratio_mask = torch.ones(batch, 1, tiles, dtype=torch.long)

    cpu_out = model_cpu(pixel_values=pixel_values,
                        aspect_ratio_ids=aspect_ratio_ids,
                        aspect_ratio_mask=aspect_ratio_mask).last_hidden_state

    model_npu = copy.deepcopy(model_cpu).to(device).eval()
    if compile_model:
        model_npu = torch.compile(model_npu, dynamic=False)
    npu_out = model_npu(pixel_values=pixel_values.to(device),
                        aspect_ratio_ids=aspect_ratio_ids.to(device),
                        aspect_ratio_mask=aspect_ratio_mask.to(device)).last_hidden_state

    check_result("Mllama vision tower", npu_out, cpu_out, rtol=rtol, atol=atol)
    print("Max diff > ", torch.max(torch.abs(npu_out.cpu() - cpu_out)))


@torch.no_grad()
def run_text(device, batch=1, seq_len=128, dtype="float32", compile_model=True,
             rtol=1e-2, atol=1e-2):
    """Run the cross-attention text tower; random states stand in for the vision tower's output."""
    from transformers.models.mllama.modeling_mllama import (
        MllamaTextModel, MllamaCrossAttentionDecoderLayer, _prepare_cross_attention_mask,
    )

    torch_dtype = _dtype_from_str(dtype)
    cfg = _text_config()
    if seq_len > cfg.max_position_embeddings:
        raise ValueError(f"seq {seq_len} exceeds max_position_embeddings {cfg.max_position_embeddings}")
    torch.manual_seed(0)
    model_cpu = MllamaTextModel(cfg).to(dtype=torch_dtype).eval()

    kinds = [type(l).__name__ for l in model_cpu.layers]
    print(f"[text] hidden={cfg.hidden_size} heads={cfg.num_attention_heads}/"
          f"{cfg.num_key_value_heads} layers={cfg.num_hidden_layers} "
          f"cross_at={cfg.cross_attention_layers} seq={seq_len}")
    print("layers:", [k.replace("Mllama", "").replace("DecoderLayer", "") for k in kinds])
    print("text params:", sum(p.numel() for p in model_cpu.parameters()))

    n_cross = sum(isinstance(l, MllamaCrossAttentionDecoderLayer) for l in model_cpu.layers)
    assert n_cross > 0, "no cross-attention layer was built -- this is not testing Mllama"
    print(f"cross-attention layers built: {n_cross}")

    tiles = VISION["max_num_tiles"]
    vision_tokens = (VISION["image_size"] // VISION["patch_size"]) ** 2 + 1

    g = torch.Generator().manual_seed(0)
    input_ids = torch.randint(3, cfg.vocab_size, (batch, seq_len), generator=g,
                              dtype=torch.long)
    cross_states = torch.randn(batch, tiles * vision_tokens, cfg.hidden_size,
                               generator=g, dtype=torch_dtype)
    cross_mask_in = torch.ones(batch, seq_len, 1, tiles, dtype=torch.long)
    cross_mask, full_row_mask = _prepare_cross_attention_mask(
        cross_mask_in, num_vision_tokens=vision_tokens, dtype=torch_dtype)
    print(f"cross_attention_mask={tuple(cross_mask.shape)} "
          f"full_text_row_masked_out_mask={tuple(full_row_mask.shape)}")

    def call(m, dev):
        return m(input_ids=input_ids.to(dev),
                 cross_attention_states=cross_states.to(dev),
                 cross_attention_mask=cross_mask.to(dev),
                 full_text_row_masked_out_mask=full_row_mask.to(dev)).last_hidden_state

    cpu_out = call(model_cpu, "cpu")

    model_npu = copy.deepcopy(model_cpu).to(device).eval()
    if compile_model:
        model_npu = torch.compile(model_npu, dynamic=False)
    npu_out = call(model_npu, device)

    check_result("Mllama text tower (cross-attention)", npu_out, cpu_out,
                rtol=rtol, atol=atol)
    print("Max diff > ", torch.max(torch.abs(npu_out.cpu() - cpu_out)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llama 3.2 Vision on the Triton route")
    parser.add_argument("--part", type=str, default="all",
                        choices=["vision", "text", "all"],
                        help="vision = the tower; text = the cross-attention decoder")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--no-compile", dest="compile", action="store_false", default=True)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    args = parser.parse_args()

    sys.path.append(os.environ.get("PYTORCHSIM_ROOT_PATH", "/workspace/PyTorchSim"))
    device = torch.device("npu:0")
    torch.compiler.is_compiling = lambda: True  # FIXME. Same as test_llama.py.

    if args.part in ("vision", "all"):
        run_vision(device, batch=args.batch, dtype=args.dtype,
                   compile_model=args.compile, rtol=args.rtol, atol=args.atol)
    if args.part in ("text", "all"):
        run_text(device, batch=args.batch, seq_len=args.seq, dtype=args.dtype,
                 compile_model=args.compile, rtol=args.rtol, atol=args.atol)
    print("Mllama Simulation Done")
