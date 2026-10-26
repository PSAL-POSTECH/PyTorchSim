"""The Kimi line, end to end on the Triton codegen route, oldest rung first.

Moonshot's open models are one architecture that grew, so this file is a LADDER
rather than a model: each rung is the previous one plus something this backend
has not had to do before, and a rung is only worth climbing once the one below
it passes.

    rung          model                                   what it adds here
    ----          -----                                   -----------------
    moonlight     Moonlight-16B-A3B-Instruct   2025-02    MLA (kv_lora q/kv down-
                                                          projections, split
                                                          nope/rope head dims) +
                                                          sigmoid-routed MoE
    kimi-vl       Kimi-VL-A3B-Instruct         2025-04    MoonViT tower in front
                                                          of that same LM
    kimi-k2       Kimi-K2-Instruct             2025-07    the same MLA/MoE at
                                                          7168 hidden, 384
                                                          experts, noaux_tc gate
    kimi-linear   Kimi-Linear-48B-A3B          2025-10    KDA linear attention
                                                          3:1 against MLA

`moonlight` is the LM half of the first Kimi-named release: Kimi-VL's language
model IS this network, and both load DeepSeek's own remote code
(`modeling_deepseek.py`, model_type deepseek_v3). Starting a rung below the
first Kimi is deliberate -- it separates "MLA + MoE on this backend" from
"MoonViT on this backend", and the vision rung cannot be read until the text one
does.

Weights are random from the config, so no checkpoint is downloaded -- only the config and, for the
remote rungs, the modeling file. Every rung runs at its published per-layer size and only the layer
count is cut, by `--layers` (default 2). Two layers is the minimum that means anything on this
ladder: `first_k_dense_replace` makes layer 0 a plain MLP, so the MoE block only exists from layer
1 on. Moonlight's config is the cached -Instruct snapshot, whose architecture fields are the base
Moonlight-16B-A3B's. The input is batch 1, one prefill of `--seq` tokens (default 128).

The vision rungs cut MoonViT to the same `--layers` and feed one 224 px image: a 16x16 grid of
14 px patches, which the 2x2 patch merger folds into 64 image tokens inside the prompt.

Judgement is spike's, against the same model on CPU. Run it with the Triton
route on and timing off:

    source /workspace/kimi-env.sh
    python tests/models/Kimi/test_kimi.py --rung moonlight

The tolerance here is 1e-3. It is deliberately NOT the 3e-1 the DeepSeek test carries: that number
was picked for fp16 and it would hide two orders of magnitude of drift on this fp32 path.

What the moonlight rung crosses, and what therefore needs no separate claim: MLA's
asymmetric head (q is nope+rope wide, v is not), the kv_lora down-projections,
the RoPE apply kernel (`cat`/`index`/`neg`/`slice` -- the shape that stops
DeepSeek-V3 in Spike), the sigmoid gate's topk, and the routed-expert loop.
"""

import argparse
import copy
import importlib.abc
import importlib.util
import os
import sys
import types

import torch

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from result_check import check_result


#: Rung -> (model id, kind, remote). Release order; climb it in this order.
#:
#: `remote` IS WHERE THE MODEL CODE COMES FROM, and on transformers 5 it decides
#: whether the rung runs at all. The Hub files these repos ship are 4.x code:
#: they import `is_torch_fx_available` from transformers.utils.import_utils,
#: which v5 removed, so `trust_remote_code=True` fails at import before a config
#: is built. Where transformers implements the architecture itself, the
#: maintained implementation is the one to use -- and it is the same network:
#: Moonlight's config says model_type deepseek_v3, which transformers has had
#: since 4.51.
#:
#:     kimi_k25       native   (Kimi K2.5, and it is the newest rung here)
#:     deepseek_v3    native   (Moonlight, and Kimi-VL's language half)
#:     kimi_vl        remote   MoonViT has no transformers implementation
#:     kimi_k2        remote   but the architecture IS deepseek_v3
#:     kimi_linear    remote   KDA has no transformers implementation
_RUNGS = {
    "moonlight":   ("moonshotai/Moonlight-16B-A3B-Instruct",   "causal-lm",  "native"),
    "kimi-vl":     ("moonshotai/Kimi-VL-A3B-Instruct",         "vision2seq", "remote"),
    "kimi-k2":     ("moonshotai/Kimi-K2-Instruct",             "causal-lm",  "deepseek_v3"),
    "kimi-linear": ("moonshotai/Kimi-Linear-48B-A3B-Instruct", "causal-lm",  "remote"),
    # K2.5 IS A VISION-LANGUAGE MODEL and transformers registers it under
    # AutoModelForImageTextToText, not AutoModelForCausalLM. This rung runs its
    # TEXT half only for now -- no pixel_values -- so what it adds over kimi-k2
    # is the 2026 config, not the tower.
    "kimi-k25":    ("moonshotai/Kimi-K2.5",                    "image-text", "kimi_k25"),
}


def _load_config(model_id, source):
    """The rung's config, from whichever implementation can build it.

    THREE SOURCES, AND THE THIRD IS THE INTERESTING ONE.

      native        the model_type is one transformers implements
                    (Moonlight says deepseek_v3, which it has had since 4.51)
      remote        the Hub file, for an architecture transformers does not
                    have -- MoonViT, KDA. On transformers 5 these repos'
                    files are 4.x code and fail at import.
      <model_type>  REBUILT on a native config class. Kimi K2 is DeepSeek-V3
                    with different numbers -- same key names, same meanings,
                    kv_lora_rank / qk_nope_head_dim / n_routed_experts and the
                    rest -- so its config.json is read as data and handed to
                    DeepseekV3Config. That keeps what the rung is about (7168
                    wide, 384 experts, noaux_tc) without running 4.x-only code
                    to get it.

    The rebuild drops the keys that name the remote implementation --
    `auto_map`, `architectures`, `model_type` -- and keeps every other value.
    """
    from transformers import AutoConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    if source in ("native", "remote"):
        return AutoConfig.from_pretrained(model_id,
                                          trust_remote_code=(source == "remote"))

    import json
    from transformers.utils import cached_file

    with open(cached_file(model_id, "config.json")) as fh:
        raw = json.load(fh)
    for key in ("auto_map", "architectures", "model_type", "_name_or_path"):
        raw.pop(key, None)
    return CONFIG_MAPPING[source](**raw)

#: One 224 px image as MoonViT sees it: a 16x16 grid of 14 px patches, 64 tokens after the 2x2 merger.
_IMAGE_GRID = (16, 16)


# DeepSeek's remote modeling file imports flash_attn, and transformers' static
# check_imports requires it to be installed even though this backend never runs
# flash attention (math-only SDPA). This image is CPU-only and has none, so
# satisfy the import check without providing package metadata: with no metadata
# is_flash_attn_2_available() stays False and the model takes the eager path.
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


# The MoE gate breaks the graph, and what follows the break has to be compiled
# too or it runs eager and this measures nothing.
torch.npu.register_eager_to_compile([
    "aten::zero_",
    "aten::sum.IntList_out",
    "aten::mul.out",
    "aten::floor_divide",
    "aten::floor_divide.Tensor",
    "aten::floor_divide.Scalar",
    "aten::cat.out",
    "aten::sort.values_stable",
])


def _dtype_from_str(name):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(name, torch.float32)


def _cut_layers(config, layers):
    """Keep the first `layers` layers; every per-layer size stays the published one."""
    config.num_hidden_layers = min(int(layers), int(config.num_hidden_layers))
    return config


def _build_inputs(batch, seq_len, vocab_size, device):
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, vocab_size, (batch, seq_len), generator=g, dtype=torch.int64)
    return ids.to(device)


def _vision_inputs(config, ids, grid_h, grid_w):
    """One image, and the placeholder tokens that reserve room for it.

    The image arrives already cut into patches -- `pixel_values` is
    (patches, 3, patch, patch) and `image_grid_hws` says how those patches are
    laid out -- because the HF image processor, not the model, does the cutting.
    The 2x2 patch merger then turns four patches into one image token, so the
    prompt has to hold exactly grid_h*grid_w/4 placeholders: the model asserts
    that count against the features it produced.
    """
    ps = config.vision_config.patch_size
    mh, mw = config.vision_config.merge_kernel_size
    n_patches = grid_h * grid_w
    n_tokens = n_patches // (mh * mw)

    g = torch.Generator().manual_seed(1)
    pixel_values = torch.randn(n_patches, 3, ps, ps, generator=g)
    image_grid_hws = torch.tensor([[grid_h, grid_w]], dtype=torch.int64)

    ids = ids.clone()
    if ids.numel() < n_tokens + 1:
        raise ValueError(f"seq_len {ids.numel()} is too short for {n_tokens} image tokens")
    ids[:, 1:1 + n_tokens] = config.media_placeholder_token_id
    return ids, {"pixel_values": pixel_values, "image_grid_hws": image_grid_hws}


def _vision_inputs_thw(config, ids, grid_h, grid_w):
    """The same image, in transformers' own spelling for this family.

    `image_grid_thw` is (images, 3) with a time axis in front -- one frame for a
    still image -- and the token count is prod(thw) over the merge kernel, which
    `get_image_features` computes exactly that way. The placeholder is
    `config.image_token_id` rather than a media_placeholder attribute.
    """
    vision = config.vision_config
    ps = vision.patch_size
    mh, mw = vision.merge_kernel_size
    n_patches = grid_h * grid_w
    n_tokens = n_patches // (mh * mw)

    g = torch.Generator().manual_seed(1)
    pixel_values = torch.randn(n_patches, 3, ps, ps, generator=g)
    grid_thw = torch.tensor([[1, grid_h, grid_w]], dtype=torch.int64)

    ids = ids.clone()
    if ids.numel() < n_tokens + 1:
        raise ValueError(f"seq_len {ids.numel()} is too short for {n_tokens} image tokens")
    ids[:, 1:1 + n_tokens] = config.image_token_id
    return ids, {"pixel_values": pixel_values, "image_grid_thw": grid_thw}


def _logits(output):
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, (list, tuple)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Unsupported output type for comparison: {type(output)}")


@torch.no_grad()
def run_rung(rung, device, dtype="float32", batch=1, seq_len=128, layers=2,
             compile_model=True, rtol=1e-3, atol=1e-3):
    """Build one rung at its published per-layer size, `layers` deep, and compare it with CPU.

    The text tower and, for the vision rungs, MoonViT are both cut to `layers`. Kimi-K2.5's
    tower is MoonViT maintained inside transformers (kimi_k25), so it takes the same image with
    the grid spelt thw rather than hw. MoonViT offers eager/sdpa/flash_attention_2 and this image
    has no flash_attn, so the Kimi-VL rung asks for eager rather than let the default pick.
    """
    from transformers import AutoModelForCausalLM

    model_id, kind, source = _RUNGS[rung]

    grid = None
    config = _load_config(model_id, source)
    text_config = getattr(config, "text_config", config)
    if seq_len > text_config.max_position_embeddings:
        raise ValueError(f"seq {seq_len} exceeds {model_id}'s "
                         f"{text_config.max_position_embeddings} positions")
    _cut_layers(text_config, layers)
    if kind in ("image-text", "vision2seq"):
        _cut_layers(config.vision_config, layers)
        grid = _IMAGE_GRID
    if kind == "vision2seq":
        config._attn_implementation = "eager"
        config.vision_config._attn_implementation = "eager"
        config.text_config._attn_implementation = "eager"
    config.use_cache = False

    # Random weights are the model here, so seed before building: without this
    # every run is a different network and the worst-element error wanders
    # across the threshold.
    torch.manual_seed(0)
    auto_class = AutoModelForCausalLM
    if kind == "image-text":
        from transformers import AutoModelForImageTextToText
        auto_class = AutoModelForImageTextToText
    model = auto_class.from_config(
        config, trust_remote_code=(source == "remote")).eval()
    model = model.to(dtype=_dtype_from_str(dtype))

    vocab = text_config.vocab_size
    print(f"rung {rung}: {model_id}")
    print("  hidden_size:", text_config.hidden_size,
          " layers:", text_config.num_hidden_layers,
          " heads:", text_config.num_attention_heads,
          " routed experts:", getattr(text_config, "n_routed_experts", "n/a"))
    if grid is not None:
        print("  vision:", config.vision_config.hidden_size, "wide,",
              config.vision_config.num_hidden_layers, "layers, grid", grid)
    print("  params:", sum(p.numel() for p in model.parameters()))

    cpu_ids = _build_inputs(batch, seq_len, vocab, torch.device("cpu"))
    extra = {}
    if grid is not None and kind == "image-text":
        cpu_ids, extra = _vision_inputs_thw(config, cpu_ids, *grid)
    elif grid is not None:
        cpu_ids, extra = _vision_inputs(config, cpu_ids, *grid)

    # BOTH COPIES ARE TAKEN BEFORE EITHER RUNS. MoonViT's 2D rope caches its
    # cis table in a plain attribute (`Rope2DPosEmb.freqs_cis`, not a buffer)
    # on the device of the first call, so copying the npu model out of the
    # already-run cpu model hands it a CPU complex64 table that `.to(device)`
    # does not move -- and the multiply inside apply_rope then fails device
    # inference before a kernel is ever emitted.
    model_cpu = copy.deepcopy(model).cpu().eval()
    model_npu = copy.deepcopy(model).to(device).eval()

    cpu_out = _logits(model_cpu(cpu_ids, **extra))

    if compile_model:
        model_npu = torch.compile(model_npu, dynamic=False)
    npu_out = _logits(model_npu(cpu_ids.to(device),
                                **{k: v.to(device) for k, v in extra.items()}))

    print(f"  max abs diff: {(npu_out.cpu() - cpu_out).abs().max().item():.6g}")
    check_result(f"Kimi {rung}", npu_out, cpu_out, rtol=rtol, atol=atol)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The Kimi ladder on the Triton route")
    parser.add_argument("--rung", default="moonlight", choices=list(_RUNGS))
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    run_rung(
        args.rung,
        torch.device("npu:0"),
        dtype=args.dtype,
        batch=args.batch,
        seq_len=args.seq,
        layers=args.layers,
        compile_model=not args.no_compile,
    )
