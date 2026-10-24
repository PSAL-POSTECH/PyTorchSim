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

Weights are random from the config (`--init-mode config-random`), so no
checkpoint is downloaded -- only the config and the remote modeling file. The
presets shrink hidden_size, layer count and expert count; they do NOT shrink the
MLA head dims (qk_nope 128 / qk_rope 64 / v 128, kv_lora_rank 512), because
those are the shapes the layout work is about.

Judgement is spike's, against the same model on CPU. Run it with the Triton
route on and timing off:

    source /workspace/kimi-env.sh
    python tests/models/Kimi/test_kimi.py --rung moonlight --preset tiny

WHERE THE LADDER STANDS. The environment moved mid-bringup -- transformers
4.43.4 -> 4.51.3 and triton-npu's p01/p11/p13/p16 refactors -- so the rows say
which side of that they were measured on. The new baseline is the one to
extend; the old rows stay until each is re-measured, because a number whose
compiler is gone is a number about nothing.

  BASELINE: transformers 4.51.3, triton-npu develop-e2e-kimi at the develop
  merge (b28808b..), PyTorchSim develop-e2e-kimi ac39991.

    rung       preset  kernels  goldens  max diff      time
    moonlight  tiny        51      430   7.1526e-07    ~5 min

  BEFORE IT: transformers 4.43.4, triton-npu 3434608, PyTorchSim 204ec14.

    rung       preset  kernels  goldens  max diff      time
    moonlight  tiny        49      408   8.0466e-07    ~5 min
    moonlight  small       58      416   1.0729e-06    ~9 min
    moonlight  medium     157        -   2.0266e-06    ~43 min
    kimi-vl    tiny        93      909   8.3447e-07    ~8 min

The two kernels and 22 goldens tiny gained across that move are not attributed:
both the compiler and transformers changed between the runs, and one run cannot
tell them apart. Pin TORCHSIM_COMPILE_DIR at a pre-merge worktree to split them if it ever
matters.

`goldens` is with `pytorchsim_functional_verify_per_kernel` on, so it is every
realized Spike buffer compared against a CPU golden at 1e-4 and not just the
logits -- 408 of them across 17 graphs, none divergent.

The tolerance here is 1e-3 against a measured 8e-07. It is deliberately NOT the
3e-1 the DeepSeek test carries: that number was picked for fp16 and it would
hide two orders of magnitude of drift on this fp32 path.

What tiny already crosses, and what therefore needs no separate claim: MLA's
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

#: preset -> (scale, layers, batch, seq_len). `scale` multiplies the widths and
#: the expert counts; layers is an absolute cap. Two layers is the minimum that
#: means anything on this ladder: `first_k_dense_replace` makes layer 0 a plain
#: MLP, so the MoE block only exists from layer 1 on.
_PRESETS = {
    "tiny":   (0.06, 2, 1, 16),
    "small":  (0.12, 2, 1, 32),
    "medium": (0.25, 4, 1, 32),
    # THE REAL WIDTHS, TWO LAYERS -- the same gate BERT and GPT-2 use. Nothing
    # is scaled here: hidden 2048, 16 heads, 64 routed experts, moe 1408. Two
    # layers is what makes it runnable, and it is enough because layer 0 is the
    # dense MLP and layer 1 is the MoE block, so both kinds are present. The
    # 27-layer model is the same two blocks repeated.
    "block":  (1.00, 2, 1, 32),
    "full":   (1.00, None, 1, 32),
}

#: preset -> (hidden, heads, layers, intermediate, grid_h, grid_w) for MoonViT.
#: The vision tower is sized OUTRIGHT rather than by `scale`, because two of its
#: numbers are not free: the 2D rope splits each head into an h half and a w
#: half, so head_dim has to stay a multiple of 4, and the patch merger folds a
#: 2x2 block, so the grid has to stay even. A proportional scale lands on
#: head_dim 69 and the rope asserts. `patch_size` stays 14 -- it is the shape of
#: the conv, and shrinking it would make the patch embed a different op.
_VISION_PRESETS = {
    "tiny":   (128, 4, 2, 256,  4, 4),
    "small":  (256, 8, 2, 512,  4, 4),
    "medium": (384, 8, 4, 768,  6, 6),
    "full":   (1152, 16, 27, 4304, 8, 8),
}


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


def _scaled(value, scale, min_value=1):
    return max(min_value, int(round(float(value) * float(scale))))


def _scale_config(config, scale, max_layers):
    """Shrink widths and counts, leave the MLA head dims alone.

    What is NOT scaled is the point: `qk_nope_head_dim`, `qk_rope_head_dim`,
    `v_head_dim` and `kv_lora_rank` stay at the real model's values, so a
    preset still has MLA's asymmetric head (q is nope+rope wide, v is not) and
    still splits the rope half off with a slice. Scaling those would turn the
    rung into a generic attention test.
    """
    for name in ("hidden_size", "intermediate_size", "num_attention_heads"):
        if hasattr(config, name):
            setattr(config, name, _scaled(getattr(config, name), scale))

    if hasattr(config, "num_key_value_heads"):
        config.num_key_value_heads = min(
            _scaled(config.num_key_value_heads, scale),
            config.num_attention_heads,
        )

    for name in ("n_routed_experts", "n_shared_experts", "num_experts",
                 "num_local_experts", "num_experts_per_tok",
                 "num_experts_per_token", "moe_intermediate_size",
                 "shared_expert_intermediate_size"):
        if hasattr(config, name) and getattr(config, name) is not None:
            setattr(config, name, _scaled(getattr(config, name), scale))

    # WIDTHS A FUSED EXPERT MATMUL CAN ADDRESS. transformers 5.x runs the routed
    # experts through `torch.nn.functional.grouped_mm` -- one batched matmul over
    # the expert dimension instead of a loop of MLPs -- and that kernel wants
    # strides that are a multiple of 16 bytes ("strides should be multiple of 16
    # bytes"). Sixteen ELEMENTS covers both dtypes with one number. 4.51.3
    # looped over per-expert MLPs and never asked. Measured and reasoned out in
    # PyTorchSim 44814bd on the DeepSeek test, which is the same architecture.
    for name in ("intermediate_size", "moe_intermediate_size",
                 "shared_expert_intermediate_size"):
        if hasattr(config, name) and getattr(config, name):
            setattr(config, name, max(16, (int(getattr(config, name)) // 16) * 16))

    # The gate groups the experts, so the count has to stay a multiple of the
    # group count -- AND EVERY GROUP NEEDS TWO EXPERTS IN IT, because the group
    # score is the sum of the top TWO experts in that group. A group of one asks
    # topk for a second element that is not there ("selected index k out of
    # range"). The group COUNT is what gives: shrinking it keeps both halves of
    # the structure (several groups, a choice inside each) and leaves the expert
    # count where the preset put it.
    n_group = getattr(config, "n_group", None) or getattr(config, "num_expert_group", None)
    if n_group and getattr(config, "n_routed_experts", None):
        g = max(1, min(int(n_group), max(1, int(config.n_routed_experts) // 2)))
        config.n_routed_experts = max(2 * g, ((config.n_routed_experts + g - 1) // g) * g)
        if hasattr(config, "n_group"):
            config.n_group = g
        if hasattr(config, "topk_group"):
            config.topk_group = max(1, min(int(config.topk_group), g))

    if max_layers is not None and hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = max(1, min(int(max_layers), int(config.num_hidden_layers)))

    # A MULTIPLE OF THE HEAD COUNT **AND** OF 16 ELEMENTS. The head count is for
    # the dense projections (MLA's own heads are sized by the head dims above);
    # the 16 is the same grouped_mm stride rule as the widths. Asked of the two
    # together rather than of either alone, so both hold at once.
    if getattr(config, "num_attention_heads", 0):
        step = int(config.num_attention_heads)
        while step % 16:
            step *= 2
        config.hidden_size = max(step, (config.hidden_size // step) * step)
    return config


def _build_inputs(batch, seq_len, vocab_size, device):
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, vocab_size, (batch, seq_len), generator=g, dtype=torch.int64)
    return ids.to(device)


def _size_vision(vision, preset):
    """MoonViT is sized outright. See `_VISION_PRESETS` for why, and note that
    `patch_size` and `merge_kernel_size` are left at the real model's values."""
    hidden, heads, layers, inter, grid_h, grid_w = _VISION_PRESETS[preset]
    vision.hidden_size = hidden
    vision.num_attention_heads = heads
    vision.num_hidden_layers = layers
    vision.intermediate_size = inter
    return grid_h, grid_w


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
def run_rung(rung, device, preset="tiny", dtype="float32", batch=None,
             seq_len=None, layers=None, compile_model=True, rtol=1e-3, atol=1e-3):
    from transformers import AutoConfig, AutoModelForCausalLM

    model_id, kind, source = _RUNGS[rung]
    scale, max_layers, preset_batch, preset_seq = _PRESETS[preset]
    batch = batch if batch is not None else preset_batch
    seq_len = seq_len if seq_len is not None else preset_seq
    max_layers = layers if layers is not None else max_layers

    grid = None
    config = _load_config(model_id, source)
    if kind == "image-text":
        # ITS VISION TOWER **IS** MoonViT, maintained inside transformers: the
        # native kimi_k25 vision config carries Kimi-VL's numbers exactly --
        # hidden 1152, 27 layers, 16 heads, intermediate 4304, patch 14, merge
        # (2, 2). So the same sizing applies, and the same two constraints
        # (head_dim a multiple of 4 for the 2D rope, an even grid for the
        # merger). pixel_values has the same (patches, 3, patch, patch) layout;
        # only the grid spelling differs -- thw here, hw in the Hub's Kimi-VL.
        _scale_config(getattr(config, "text_config", config), scale, max_layers)
        grid = _size_vision(config.vision_config, preset)
        # THE WIDTH IS STATED TWICE. `Kimi_K25MultimodalProjection` sizes its
        # in_proj from `vision_config.hidden_size` but its pre_norm from the
        # top-level `projection_hidden_size`, which carries the same 1152. Move
        # one and the norm rejects the tower's output:
        #   normalized_shape=[1152], but got input of size [4, 4, 128]
        if hasattr(config, "projection_hidden_size"):
            config.projection_hidden_size = config.vision_config.hidden_size
    elif kind == "vision2seq":
        # The vision rung carries two configs, and the text half IS the rung
        # below -- same DeepseekV3Config, same remote code.
        _scale_config(config.text_config, scale, max_layers)
        grid = _size_vision(config.vision_config, preset)
        # MoonViT offers eager/sdpa/flash_attention_2 and this image has no
        # flash_attn, so say eager rather than let the default pick.
        config._attn_implementation = "eager"
        config.vision_config._attn_implementation = "eager"
        config.text_config._attn_implementation = "eager"
    else:
        _scale_config(config, scale, max_layers)
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

    text_config = getattr(config, "text_config", config)
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

    # Print the worst element whether or not it passes: the threshold says
    # pass/fail, this says how much room is left, and it is the number the
    # ladder note below is written from.
    print(f"  max abs diff: {(npu_out.cpu() - cpu_out).abs().max().item():.6g}")
    check_result(f"Kimi {rung} ({preset})", npu_out, cpu_out, rtol=rtol, atol=atol)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The Kimi ladder on the Triton route")
    parser.add_argument("--rung", default="moonlight", choices=list(_RUNGS))
    parser.add_argument("--preset", default="tiny", choices=list(_PRESETS))
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    run_rung(
        args.rung,
        torch.device("npu:0"),
        preset=args.preset,
        dtype=args.dtype,
        batch=args.batch,
        seq_len=args.seq_len,
        layers=args.layers,
        compile_model=not args.no_compile,
    )
