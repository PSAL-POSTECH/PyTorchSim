"""Gemma 1, 2 and 3 (text) on the Triton codegen route.

Built from the ``transformers`` configs with random weights, so it needs no
network and no checkpoint.  The models are ``transformers``' own
``GemmaForCausalLM`` / ``Gemma2ForCausalLM`` / ``Gemma3ForCausalLM`` -- nothing
here reimplements a block, so what the test exercises is the shipped module,
not our reading of it.  One file covers all three generations because they are
deltas on one block, the way the Qwen file covers its family.

HOW DEEP A PRESET RUNS IS DECIDED BY THE REPEATING UNIT, not by taste:

  * Gemma 1's unit is ONE layer -- 18 of them at 2B, 28 at 7B, all identical.
    A preset runs one.  What depth would add is the block-to-block seam, which
    the Llama and GPT-2 tests already cross.
  * Gemma 2 alternates local and global attention, ``is_sliding = not
    (layer_idx % 2)``, so its unit is TWO layers.  A one-layer preset would run
    the local variant and never the global one.
  * Gemma 3 runs FIVE local layers to one global, ``is_sliding =
    (layer_idx + 1) % sliding_window_pattern`` with the pattern at 6, so its
    unit is SIX layers.  That is affordable only because Gemma 3's smallest
    shipped text model is narrow (1152 hidden), which is why ``3-1b`` is the
    preset that carries the pattern and ``3-4b`` carries width alone.

WHAT EACH GENERATION ADDS, and why it needs its own preset:

  * 1  the baseline block -- see the Gemma-1 section below.
  * 2  LOCAL/GLOBAL ALTERNATION and LOGIT SOFT-CAPPING.  Attention logits are
       capped at 50 and the final logits at 30, each as
       ``tanh(x / cap) * cap`` -- a tanh on the [batch, heads, seq, seq] score
       tensor and again on [batch, seq, vocab].  Nothing else in this suite
       caps.  Gemma 2 also adds a SECOND PAIR of RMSNorms per layer
       (``pre_feedforward_layernorm`` / ``post_feedforward_layernorm``, on top
       of input/post_attention), so the block carries four norms where Llama
       and Gemma 1 carry two.
  * 3  TWO ROPE BASES IN ONE MODEL.  Local layers use ``rope_local_base_freq``
       (10000) and global layers ``rope_theta`` (1000000), materialised as two
       separate rotary modules (``rotary_emb`` and ``rotary_emb_local``) whose
       cos/sin the layers pick between.  Every other model here has exactly one
       inv_freq.  Soft-capping is gone again (both caps are None).  Gemma 3
       also applies q_norm/k_norm over head_dim -- that is NOT new, Qwen 3
       contributes it -- so the presets do not claim it.

THE SLIDING WINDOW IS SCALED DOWN WITH seq, AND THAT IS LOAD-BEARING.  Both
shipped configs use a 4096-token window; at seq 32 ``torch.tril(..., diagonal=
-4096)`` selects NOTHING, so the local layers become byte-identical to the
global ones and the alternation the presets exist to test evaporates.  Measured:
window 4096 at seq 32 masks 0 elements, window 8 masks 300.  So window is a
preset field, ``_assert_character`` refuses any preset whose window is not
strictly inside its seq, and a "Gemma 2 passes" that never built a band mask
cannot happen quietly.

WHAT GEMMA 1 ADDS over the Llama/Qwen blocks already in this suite:

  * RMSNorm SCALED BY (1 + w).  ``GemmaRMSNorm`` holds a zero-initialised
    weight and applies ``x * (1 + w)``, upcasting to float32 first and casting
    back after -- every other RMSNorm in this suite is ``x * w``.  The offset
    is not cosmetic: with w near zero the Llama form collapses the activation
    and the Gemma form passes it through, so a backend that folded the two
    together would fail here and nowhere else.
  * EMBEDDING SCALED BY sqrt(hidden_size).  The embedding output is multiplied
    by a scalar (55.4 at 7B) before the first block.  A broadcast multiply of a
    Python-side constant against the full [batch, seq, hidden] activation.
  * head_dim 256 -- TWICE the largest in this suite, which tops out at 128
    (Llama 3, Qwen 2.5/3).  It doubles the reduction length inside attention
    and the tile the systolic array sees for q/k/v.
  * MQA at 2B: ``num_key_value_heads=1``, so n_rep = 8 and ONE kv head is
    broadcast across every query head.  The suite's GQA repeats are 1, 2, 3, 4,
    6, 7 and 8-into-8; a single kv head is the degenerate end none of them
    reach, and it makes k_proj/v_proj narrow (2048 -> 256) where q_proj is
    square.
  * GeGLU.  The gated MLP uses ``gelu_pytorch_tanh``; every gated MLP in this
    suite is SwiGLU (silu).  GPT-2 contributes a gelu, but an ungated one.
  * vocab 256000 with a tied lm_head -- the largest vocab in the suite.  The
    presets scale vocab down (it is a cost axis, not a character axis) but keep
    the tie, which is what makes the vocab projection share the embedding.

WHY 2B AND 7B ARE BOTH HERE.  They are not the same block at two widths:

  * 2B is MQA (kv_heads 1) and heads * head_dim == hidden (8 * 256 == 2048).
  * 7B is MHA (kv_heads 16) and heads * head_dim != hidden (16 * 256 == 4096
    against 3072 hidden), so q_proj is not square and o_proj shrinks 4096 ->
    3072.

Dropping either one drops a claim.  ``_assert_character`` cashes every claim
above against the built model, so a preset cannot pass while proving nothing.

    source /workspace/torchsim-env.sh
    python tests/models/Gemma/test_gemma.py --preset 7b

MEASURED ON TRANSFORMERS 5.15.0, 2026-08-14, triton-npu develop c6e9d7f,
functional-only config (Spike decides correctness; no cycles are reported).
Every preset ran with its OWN TORCHSIM_DUMP_PATH: the kernel cache is keyed on
Triton source text alone, and a bmm template's text is identical for two presets
that differ only in head COUNT, so a shared dump lets one preset's kernel answer
for another's.

    preset         gen  layers  kernels   max diff    time   (4.51.3 was)
    tiny            1      1    15 / 18  9.5367e-07   101s   same, same
    2b              1      1    16 / 18  5.7220e-06   231s   same, same
    7b              1      1    15 / 17  7.6294e-06   348s   same, same
    2-27b-scalar    2      1    17 / 19  3.8147e-06   108s   20/22, same
    2-2b            2      2    20 / 35  4.0717e-06   323s   24/38, 5.7220e-06
    3-4b            3      1    20 / 21  7.6294e-06   240s   23/24, 5.7220e-06
    3-1b            3      6    21 / 96  3.6955e-06   500s   25/100, 4.5896e-06
    3-mm            3      1    25 / 38  FAILS        169s   44/61, 1.0133e-06
    pali            1      1    25 / 38  FAILS        187s   42/59, 1.2517e-06
    rg-2b           G      3    24 / 45  1.2159e-05   441s   did not reach the backend

`rg-2b` is Griffin, marked G: a linear recurrence with a temporal conv1d where
the other families put attention, and two layers in three are that recurrence.
Its diff is a decade above the rest of the file, which is where a scan over the
sequence would put it, but that is an observation and not a measured account.
It reaches the backend at all because of two fixes made the same day -- torch's
own conv1d_to_conv2d registered so the conv is lowered rather than sent to an
extern kernel, and triton-npu d846ff4 so the fused block's 24 arguments fit the
2048 bytes riscv-pk gives a guest's argv. Either one missing and it stops before
a kernel runs.

GEMMA 1 IS BIT-IDENTICAL ACROSS THE BUMP -- same kernels, same last digit, at
all three widths.  Gemma 2 and 3 keep passing but fuse differently, and where a
fusion boundary moves the accumulation order moves with it, so the last digit
drifts in both directions.  The tolerance is 1e-2 and these are float32 noise;
none of it is a regression.

THE TWO MULTIMODAL PRESETS ARE.  They passed at 4.51.3 and fail here, and the
cause is not the version-agnostic accessors below.  5.15.0 stops fusing the
splice into its neighbours: what was
``..._embedding_eq_expand_masked_scatter_unsqueeze_42`` at 4.51.3 is a
standalone ``masked_scatter`` kernel here, holding two <32 x i64> index vectors
at once, and llc answers that with a ``vs2r.v`` -- a whole register GROUP on the
stack.  The tnpu pipeline refuses a spilled kernel because there is no working
spill path behind it; Spike would die at the next stage on an invalid spad
address.  Both families reach the identical 25/38 despite different towers and
different text widths, which is what says the stop belongs to the shared splice
path rather than to either model.  The index range is 32 x 256, so i32 would
carry it -- narrowing it is the open lead and is NOT done here.

"kernels" is unique / executed.  Three things the counts say:

  * GEMMA 2 COSTS KERNELS, NOT WIDTH.  2-27b-scalar compiles 20 unique kernels
    at 11.5M parameters where Gemma 1's 302M 7b compiles 15.  The extra work is
    per-layer and shape-independent: two more RMSNorms and two tanh soft-caps.
  * LOCAL AND GLOBAL ATTENTION COMPILE SEPARATELY.  2-2b runs two layers and
    lands at 24 unique against 38 executed -- the pair does not share one
    attention kernel, which is the observable consequence of the alternation.
  * DEPTH REUSES.  3-1b executes 100 kernels for 25 unique across six layers:
    the five local layers share, and the global layer's different rope base
    earns it its own.  That is the argument for running exactly one period and
    not two.

2b compiles one MORE kernel than 7b despite being under half the size: MQA
broadcasts its single kv head where 7B's MHA does not, and that broadcast does
not fuse into either neighbour.

Nothing here was blocked when it was written -- all three generations ran end to
end on the first attempt against this pin, with no backend change.  These
numbers are a floor to regress against, not a fix.
"""

import argparse
import copy
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from result_check import check_result


# Presets shrink cost, not character.  Width carries head_dim, the GQA repeat
# and the GeGLU ratio, so width stays honest wherever it is affordable; depth
# and vocab are the cheap axes and are scaled down everywhere.  Both shipped
# Gemma 1 configs tie word embeddings, so every preset does too.
_PRESETS = {
    # Smoke.  Small enough to tell a broken test file from a broken backend.
    # Keeps MQA so the cheapest preset still covers the single-kv-head path.
    "tiny": dict(family="gemma", layers=1, hidden=256, heads=4, kv_heads=1,
                 head_dim=64, intermediate=1024, vocab=256, seq=16, tie=True),

    # Gemma-2B at its real width: MQA (n_rep = 8), head_dim 256, GeGLU 16384.
    "2b": dict(family="gemma", layers=1, hidden=2048, heads=8, kv_heads=1,
               head_dim=256, intermediate=16384, vocab=8192, seq=32, tie=True),

    # Gemma-7B at its real width: MHA, head_dim 256 decoupled from 3072 hidden,
    # GeGLU 24576.
    "7b": dict(family="gemma", layers=1, hidden=3072, heads=16, kv_heads=16,
               head_dim=256, intermediate=24576, vocab=8192, seq=32, tie=True),

    # Gemma2-2B at its real width, TWO layers -- one local, one global, which is
    # Gemma 2's whole repeating unit.  Soft-capping on at its shipped values.
    "2-2b": dict(family="gemma2", layers=2, hidden=2304, heads=8, kv_heads=4,
                 head_dim=256, intermediate=9216, vocab=8192, seq=32, tie=True,
                 window=8, attn_softcap=50.0, final_softcap=30.0, qpas=256),

    # Gemma2-27B's attention scaling, at a width that costs nothing: 27B is the
    # one Gemma whose query_pre_attn_scalar (144) is NOT head_dim (128), so the
    # pre-attention scale is not head_dim**-0.5.  One layer -- this preset is a
    # claim about a scalar, and the alternation is 2-2b's job.
    "2-27b-scalar": dict(family="gemma2", layers=1, hidden=1024, heads=8, kv_heads=4,
                         head_dim=128, intermediate=2048, vocab=2048, seq=32, tie=True,
                         window=8, attn_softcap=50.0, final_softcap=30.0, qpas=144),

    # Gemma3-1B at its real width, SIX layers -- one full 5:1 period, so five
    # local layers and the single global one all run.  Affordable only because
    # 1B is narrow: 1152 hidden with MQA keeps six layers near 170M.
    "3-1b": dict(family="gemma3", layers=6, hidden=1152, heads=4, kv_heads=1,
                 head_dim=256, intermediate=6912, vocab=8192, seq=32, tie=True,
                 window=8, rope_local=10000.0, rope_global=1000000.0, pattern=6, qpas=256),

    # Gemma3-4B's width, one layer.  A width claim only: at pattern 6 layer 0 is
    # local, so this preset says nothing about the period -- 3-1b covers that.
    "3-4b": dict(family="gemma3", layers=1, hidden=2560, heads=8, kv_heads=4,
                 head_dim=256, intermediate=10240, vocab=8192, seq=32, tie=True,
                 window=8, rope_local=10000.0, rope_global=1000000.0, pattern=6, qpas=256),

    # -- multimodal.  These two claim the TOWER and the SPLICE, not width ------
    # The text side is deliberately small: 3-1b already gates the Gemma 3
    # decoder, and repeating it here would only make the preset slower without
    # making it say more.  head_dim stays 256 so the shared assertion below
    # still holds.
    #
    # What they add is `masked_scatter`: the image embeddings are spliced into
    # the text sequence at the placeholder positions, which is the op every
    # VLM in this suite stops at.  A SigLIP tower and a projector come with it.
    #
    # WIDTH 256 WAS A CEILING ON 4.51.3, bisected rather than guessed: above it
    # Inductor lowered the splice's scan as a MULTI-BLOCK one --
    # `triton_helpers.exclusive_scan_decoupled_lookback_64`, which publishes
    # each block's sum and spins on `tl.atomic_xchg` until its predecessors
    # publish theirs.  This backend compiles one binary per kernel and its C
    # wrapper walks the grid as a sequential loop, so that channel does not
    # exist.  Measured there, everything else held fixed:
    #
    #     hidden   pali        3-mm
    #     128      1.0133e-06  7.1526e-07
    #     256      1.2517e-06  1.0133e-06     <- the presets
    #     512      exclusive_scan_decoupled_lookback_64, both
    #
    # ON 5.15.0 THE WIDTH IS NOT WHAT STOPS THEM, and reading it as the same
    # wall cost a wrong diagnosis.  Those logs contain no decoupled-lookback
    # scan at any width: the splice simply stops being fused, and the standalone
    # kernel spills.  See the module docstring.  The width is left at 256
    # because that is where both were last known to pass, not because 5.15.0
    # would accept it and 512 would not -- neither compiles there.
    "3-mm": dict(family="gemma3_mm", layers=1, hidden=256, heads=2, kv_heads=1,
                 head_dim=256, intermediate=512, vocab=1024, seq=32, tie=True,
                 window=8, rope_local=10000.0, rope_global=1000000.0, pattern=2,
                 qpas=256,
                 vision=dict(hidden=384, intermediate=768, layers=2, heads=6,
                             image=64, patch=16),
                 # 4x4 patches pooled to 2x2: Gemma 3 avg-pools the tower's
                 # output down to mm_tokens_per_image, which PaliGemma does not.
                 mm_tokens=4),

    # PaliGemma is the same SigLIP tower in front of a GEMMA 1 text tower, so
    # against 3-mm it separates the tower from Gemma 3's own decoder.  It emits
    # one token per patch, unpooled -- 4x4 = 16.
    # RecurrentGemma-2B at its real width, THREE layers -- block_types cycles
    # ("recurrent", "recurrent", "attention"), so three is one full period and
    # anything less runs the recurrence without ever reaching the attention it
    # alternates with.  The same rule that put Gemma 2 at two layers and Gemma 3
    # at six.
    #
    # IT IS NOT A GEMMA BLOCK.  It is Griffin: a linear recurrence with a
    # temporal conv1d where the other families put attention.  What it shares is
    # the RMSNorm's (1 + w) offset and the GeGLU, which is why it belongs in
    # this file at all, and the shared assertions cash both.  The conv1d is the
    # reason it needs `conv1d_to_conv2d` registered; the fused block is what
    # overran pk's argv buffer.
    # 3n's period is 5 (four sliding, one full).  Six layers runs one whole
    # period and one layer past it, which is what leaves a KV-shared layer with
    # a same-typed donor in front of it: the sharing index is resolved by
    # searching layer_types[:num_layers - num_kv_shared] backwards for this
    # layer's own type, so num_kv_shared_layers=1 is the largest value a
    # six-layer preset can carry without that search raising.
    "3n": dict(family="gemma3n", layers=6, hidden=2048, heads=8, kv_heads=2,
               head_dim=256, intermediate=8192, vocab=8192, seq=32, tie=True,
               window=8, pattern=5, hidden_per_layer=256, vocab_per_layer=8192,
               altup=4, laurel=64, kv_shared=1, sparsity=(0.95, 0.95)),
    "rg-2b": dict(family="recurrentgemma", layers=3, hidden=2560, heads=10,
                  kv_heads=10, head_dim=256, intermediate=7680, vocab=8192,
                  seq=32, tie=True, window=8, lru_width=2560,
                  block_types=("recurrent", "recurrent", "attention")),

    "pali": dict(family="paligemma", layers=1, hidden=256, heads=2, kv_heads=1,
                 head_dim=256, intermediate=512, vocab=1024, seq=32, tie=True,
                 vision=dict(hidden=384, intermediate=768, layers=2, heads=6,
                             image=64, patch=16),
                 mm_tokens=16),
}

_FAMILIES = {
    "gemma": ("transformers.models.gemma.configuration_gemma", "GemmaConfig",
              "transformers.models.gemma.modeling_gemma", "GemmaModel", "GemmaForCausalLM"),
    "gemma2": ("transformers.models.gemma2.configuration_gemma2", "Gemma2Config",
               "transformers.models.gemma2.modeling_gemma2", "Gemma2Model", "Gemma2ForCausalLM"),
    "gemma3": ("transformers.models.gemma3.configuration_gemma3", "Gemma3TextConfig",
               "transformers.models.gemma3.modeling_gemma3", "Gemma3TextModel", "Gemma3ForCausalLM"),
    # A conditional-generation class is both the body and the LM head, so
    # --part has nothing to choose between for these two.
    "gemma3_mm": ("transformers.models.gemma3.configuration_gemma3", "Gemma3Config",
                  "transformers.models.gemma3.modeling_gemma3",
                  "Gemma3ForConditionalGeneration", "Gemma3ForConditionalGeneration"),
    "gemma3n": ("transformers.models.gemma3n.configuration_gemma3n", "Gemma3nTextConfig",
                "transformers.models.gemma3n.modeling_gemma3n",
                "Gemma3nTextModel", "Gemma3nForCausalLM"),
    "recurrentgemma": ("transformers.models.recurrent_gemma.configuration_recurrent_gemma",
                       "RecurrentGemmaConfig",
                       "transformers.models.recurrent_gemma.modeling_recurrent_gemma",
                       "RecurrentGemmaModel", "RecurrentGemmaForCausalLM"),
    "paligemma": ("transformers.models.paligemma.configuration_paligemma", "PaliGemmaConfig",
                  "transformers.models.paligemma.modeling_paligemma",
                  "PaliGemmaForConditionalGeneration", "PaliGemmaForConditionalGeneration"),
}


def _is_vlm(preset):
    return "vision" in _PRESETS[preset]


def _text_cfg(cfg):
    """The text half of a config, whichever kind it is."""
    return getattr(cfg, "text_config", cfg)


def _text_model(model):
    """The causal-LM half of a model, whichever kind it is."""
    return getattr(model, "language_model", model)


def _descend(model, attr, depth=4):
    """Walk `.model` / `.language_model` until a module has `attr`.

    The wrappers moved between transformers versions and they moved in
    OPPOSITE directions for the two things this file reads:

        4.51.3   ForConditionalGeneration -> language_model (a ForCausalLM)
                                          -> model (the layer stack)
        5.15.0   ForConditionalGeneration -> model (a Gemma3Model)
                                          -> language_model (the layer stack)

    and `lm_head` sits on the inner ForCausalLM in the first and on the OUTER
    module in the second. Naming the attribute wanted and walking to it is the
    one formulation that holds for both, and for the plain text models where
    there is no wrapper at all.
    """
    seen = model
    for _ in range(depth):
        if hasattr(seen, attr):
            return seen
        nxt = getattr(seen, "language_model", None)
        if nxt is None or nxt is seen:
            nxt = getattr(seen, "model", None)
        if nxt is None or nxt is seen:
            break
        seen = nxt
    return None


def _lm_head(model):
    holder = _descend(model, "lm_head")
    return getattr(holder, "lm_head", None) if holder is not None else None


def _vision_tower(model):
    holder = _descend(model, "vision_tower")
    return getattr(holder, "vision_tower", None) if holder is not None else None


def _projector(model):
    holder = _descend(model, "multi_modal_projector")
    return getattr(holder, "multi_modal_projector", None) if holder is not None else None


def _layer_is_sliding(body, idx, cfg):
    """Does layer `idx` attend locally, in either spelling?

    4.51.3 decides per layer and stores the answer on the decoder layer
    (`layer.is_sliding`). 5.15.0 makes it declarative: `config.layer_types` is a
    list of "sliding_attention" / "full_attention", mirrored onto the attention
    as `layer_type`. The claim is the same -- which layers see a band mask --
    and the presets are written against the claim, not the spelling.
    """
    layer = body.layers[idx]
    flag = getattr(layer, "is_sliding", None)
    if flag is not None:
        return bool(flag)
    kind = getattr(getattr(layer, "self_attn", None), "layer_type", None)
    if kind is None:
        types = getattr(cfg, "layer_types", None)
        kind = types[idx] if types else None
    return kind == "sliding_attention"


def _rope_bases(cfg, body):
    """(local, global) rope bases, in either spelling, or None if not split.

    4.51.3 carries `rope_theta` and `rope_local_base_freq` on the config and
    builds TWO rotary modules (`rotary_emb`, `rotary_emb_local`). 5.15.0 keeps
    one rotary module and moves both bases into `rope_parameters`, keyed by the
    same layer-type names `layer_types` uses. Reading the config works for both
    and does not depend on how many modules got built.
    """
    params = getattr(cfg, "rope_parameters", None)
    if isinstance(params, dict) and "sliding_attention" in params:
        return (params["sliding_attention"].get("rope_theta"),
                params["full_attention"].get("rope_theta"))
    local = getattr(cfg, "rope_local_base_freq", None)
    glob = getattr(cfg, "rope_theta", None)
    if local is None or glob is None:
        return None
    return (local, glob)


def _dtype_from_str(name):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(name, torch.float32)


def _import(module, name):
    import importlib

    return getattr(importlib.import_module(module), name)


#: The token that stands in for an image.  Random text tokens are drawn ABOVE
#: it so a draw cannot accidentally become a placeholder and change the count
#: the vision tower has to match.
_IMAGE_TOKEN = 7

#: A multimodal family's text tower is one of the three text families, and the
#: config branches below are about the tower, not the wrapper.
_TEXT_FAMILY = {"gemma3_mm": "gemma3", "paligemma": "gemma"}


def _build_config(preset, seq_len):
    p = _PRESETS[preset]
    cfg_mod, cfg_name, _, _, _ = _FAMILIES[p["family"]]
    Config = _import(cfg_mod, cfg_name)
    seq_len = seq_len if seq_len is not None else p["seq"]
    text_family = _TEXT_FAMILY.get(p["family"], p["family"])

    kwargs = dict(
        vocab_size=p["vocab"],
        hidden_size=p["hidden"],
        num_attention_heads=p["heads"],
        num_key_value_heads=p["kv_heads"],
        head_dim=p["head_dim"],
        intermediate_size=p["intermediate"],
        num_hidden_layers=p["layers"],
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        # Gemma 1's MLP reads hidden_act; 2 and 3 read hidden_activation.  Both
        # are set so the gate cannot depend on which name a given transformers
        # version happens to consult.
        hidden_act="gelu_pytorch_tanh",
        hidden_activation="gelu_pytorch_tanh",
        attention_bias=False,
        tie_word_embeddings=p["tie"],
        use_cache=False,
        # Eager keeps the graph free of the sdpa dispatch, so a failure is about
        # the model rather than about which attention kernel transformers picked.
        attn_implementation="eager",
    )

    if text_family == "recurrentgemma":
        # Griffin's own knobs. attention_window_size is the band on the ONE
        # attention layer in each period, and it is shrunk to fit inside seq for
        # the same reason Gemma 2/3's sliding_window is: shipped it is 2048, and
        # at seq 32 a 2048-wide band is the whole row, so the local attention
        # would be indistinguishable from full and the preset would prove
        # nothing about it.
        kwargs["attention_window_size"] = p["window"]
        kwargs["lru_width"] = p["lru_width"]
        kwargs["block_types"] = p["block_types"]
        kwargs.pop("attention_bias", None)
    elif text_family == "gemma":
        kwargs["rope_theta"] = 10000.0
    elif text_family == "gemma3n":
        # 3n's own machinery, all of it on: the per-layer input embedding and
        # its projection, altup's parallel residual streams, laurel's low-rank
        # branch, activation sparsity on the leading layers, and KV sharing on
        # the trailing one.  A preset with these off is a Gemma 3 block wearing
        # a different config class.
        kwargs["sliding_window"] = p["window"]
        kwargs["hidden_size_per_layer_input"] = p["hidden_per_layer"]
        kwargs["vocab_size_per_layer_input"] = p["vocab_per_layer"]
        kwargs["altup_num_inputs"] = p["altup"]
        kwargs["laurel_rank"] = p["laurel"]
        kwargs["num_kv_shared_layers"] = p["kv_shared"]
        # Per layer, and shorter than the model on purpose: the shipped E2B is
        # sparse for its leading layers and dense after, so a flat pattern would
        # not exercise the branch that skips the threshold entirely.
        kwargs["activation_sparsity_pattern"] = list(p["sparsity"]) + \
            [0.0] * (p["layers"] - len(p["sparsity"]))
        # 3n keeps the final cap Gemma 3 dropped and has no attention cap at
        # all, so the shipped 30.0 stays rather than being pinned to None the
        # way the gemma3 branch does.
    else:
        # The window is shrunk to fit inside seq on purpose; see the module
        # docstring.  _assert_character refuses a preset where it does not.
        kwargs["sliding_window"] = p["window"]
        kwargs["query_pre_attn_scalar"] = p["qpas"]
        # Both 2 and 3 default this to "hybrid", which pairs a static cache with
        # the window for generate().  use_cache is off and this test never
        # generates, so leaving it set only produces a warning that does not
        # describe the run.
        kwargs["cache_implementation"] = None

    if text_family == "gemma2":
        kwargs["rope_theta"] = 10000.0
        kwargs["attn_logit_softcapping"] = p["attn_softcap"]
        kwargs["final_logit_softcapping"] = p["final_softcap"]

    if text_family == "gemma3":
        kwargs["rope_theta"] = p["rope_global"]
        kwargs["rope_local_base_freq"] = p["rope_local"]
        kwargs["sliding_window_pattern"] = p["pattern"]
        # Gemma 3 dropped both caps; pinning them to None keeps a future
        # transformers default from quietly reintroducing a tanh.
        kwargs["attn_logit_softcapping"] = None
        kwargs["final_logit_softcapping"] = None

    if not _is_vlm(preset):
        return Config(**kwargs), seq_len

    # Multimodal: the text kwargs above become the nested text config, and the
    # tower is sized outright.  The tower is SMALL on purpose -- this preset
    # claims that a SigLIP tower, a projector and the masked_scatter splice run
    # and agree with CPU, not that they do so at a shipped vision width.
    v = p["vision"]
    composite = dict(
        text_config=kwargs,
        vision_config=dict(hidden_size=v["hidden"], intermediate_size=v["intermediate"],
                           num_hidden_layers=v["layers"], num_attention_heads=v["heads"],
                           image_size=v["image"], patch_size=v["patch"]),
        image_token_index=_IMAGE_TOKEN,
    )
    if p["family"] == "gemma3_mm":
        composite["mm_tokens_per_image"] = p["mm_tokens"]
    else:
        composite["projection_dim"] = p["hidden"]

    cfg = Config(**composite)
    # The composite constructor does not forward attn_implementation into the
    # towers, and an sdpa dispatch here would make a failure about which kernel
    # transformers picked rather than about this backend.
    cfg._attn_implementation = "eager"
    return cfg, seq_len


def _layer_norm0(layer):
    """The norm before the layer's first block, in either family.

    Gemma 1/2/3 call it `input_layernorm`; RecurrentGemma splits the block into
    a temporal half and a channel half and calls the first one
    `temporal_pre_norm`. The claim being checked is the same either way -- that
    it is a Gemma RMSNorm with the (1 + w) offset.
    """
    for name in ("input_layernorm", "temporal_pre_norm"):
        if hasattr(layer, name):
            return getattr(layer, name)
    raise AttributeError(f"no leading norm on {type(layer).__name__}")


def _layer_mixer(layer):
    """What mixes across positions: attention, or Griffin's recurrence.

    `self_attn` in the Gemma families. RecurrentGemma names it `temporal_block`
    and it is a RecurrentGemmaRecurrentBlock on two layers out of three and a
    RecurrentGemmaAttention on the third, which is the whole point of that
    model.
    """
    for name in ("self_attn", "temporal_block"):
        if hasattr(layer, name):
            return getattr(layer, name)
    raise AttributeError(f"no mixer on {type(layer).__name__}")


def _layer_mlp(layer):
    for name in ("mlp", "mlp_block"):
        if hasattr(layer, name):
            return getattr(layer, name)
    raise AttributeError(f"no mlp on {type(layer).__name__}")


def _text_body(model):
    """The stack of decoder layers, under whichever wrappers this model has."""
    body = _descend(model, "layers")
    assert body is not None, f"no layer stack under {type(model).__name__}"
    return body


def _layer0(model):
    return _text_body(model).layers[0]


def _assert_character(preset, cfg, model, seq_len):
    """Fail if the model did not actually get the feature the preset claims.

    A preset is a claim about shape and about which Gemma the block is; this is
    where the claim is cashed.  Every check guards against a run that passes
    without exercising anything the suite does not already cover.
    """
    p = _PRESETS[preset]
    family = _TEXT_FAMILY.get(p["family"], p["family"])
    # Everything below the multimodal section is a claim about the TEXT tower,
    # so it reads the text config and the text body whichever kind of model
    # this is.  `full_cfg` keeps the composite for the multimodal checks.
    full_cfg, cfg = cfg, _text_cfg(cfg)
    body = _text_body(model)
    layer = _layer0(model)
    attn = _layer_mixer(layer)
    if not hasattr(attn, "q_proj"):
        # Griffin's layer 0 is the recurrence, which has no projections to
        # check. The attention claims below belong to the ONE layer in the
        # period that has one, so find it rather than reading layer 0 twice.
        attn = next((m for m in (_layer_mixer(l) for l in body.layers)
                     if hasattr(m, "q_proj")), attn)
    n_rep = cfg.num_attention_heads // cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads

    # -- what makes any Gemma block not a Llama block -------------------------
    # The (1 + w) offset, shared by all three generations.  Checked on the norm
    # module rather than on the config, because the config cannot express it: a
    # zero weight that still passes the activation through is only visible in
    # the module's own arithmetic.
    norm = _layer_norm0(layer)
    norm_name = type(norm).__name__
    # "Gemma" anywhere, not as a prefix: RecurrentGemmaRMSNorm carries the same
    # (1 + w) offset and is the reason that family can share these assertions.
    assert "Gemma" in norm_name and norm_name.endswith("RMSNorm"), \
        f"expected a *Gemma*RMSNorm, got {norm_name}; the (1 + w) offset is the point of this test"
    probe = torch.ones(1, 1, cfg.hidden_size)
    with torch.no_grad():
        scaled = norm(probe)
    assert torch.allclose(scaled, probe, atol=1e-5), \
        (f"{norm_name} with a zero weight must be the identity on a unit vector; "
         "a Llama-style x*w would return zeros here")

    assert attn.q_proj.bias is None, "Gemma has attention_bias=False; a bias here makes this a Qwen2 block"
    mlp = _layer_mlp(layer)
    assert type(mlp).__name__.endswith(("MLP", "Mlp"))
    for name in ("gate_proj", "up_proj", "down_proj"):
        assert hasattr(mlp, name), f"GeGLU needs {name}; this MLP is not gated"
    act = type(mlp.act_fn).__name__.lower()
    assert "gelu" in act or "tanh" in act, \
        f"the gated MLP must use a gelu, got {type(mlp.act_fn).__name__}; silu here means this is a SwiGLU block"

    # head_dim 256 is the reason the attention tiles differ from the rest of the
    # suite, so a preset that quietly fell back to hidden/heads proves nothing.
    # 2-27b-scalar is the deliberate exception: 27B is the one Gemma at 128.
    if preset not in ("tiny", "2-27b-scalar"):
        assert head_dim == 256, f"this preset runs head_dim 256, got {head_dim}"
        assert head_dim == 2 * 128, "the suite's previous maximum was 128; this preset exists to double it"

    # -- Gemma 1 is the plain block: no window, no caps -----------------------
    if family == "gemma":
        assert getattr(cfg, "sliding_window", None) is None, \
            "Gemma 1 has no sliding window; a window here means the wrong config class was built"
        assert getattr(cfg, "attn_logit_softcapping", None) is None, \
            "Gemma 1 does not soft-cap; capping here means this is a Gemma 2 block"

    # -- the window must actually bite, or the alternation proves nothing -----
    if family in ("gemma2", "gemma3", "gemma3n"):
        window = cfg.sliding_window
        assert window is not None and window < seq_len, \
            (f"sliding_window ({window}) must be strictly inside seq ({seq_len}); "
             "at the shipped 4096 against seq 32 the band mask selects nothing and "
             "the local layers become byte-identical to the global ones")
        band = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=-window)
        assert band.any(), \
            f"window {window} at seq {seq_len} masks no element; this preset would test causal attention twice"

        # is_sliding lives on the decoder layer in both generations; Gemma 3
        # also mirrors it onto the attention, Gemma 2 does not.
        sliding = [_layer_is_sliding(body, i, cfg)
                   for i in range(cfg.num_hidden_layers)]
        assert len(sliding) == cfg.num_hidden_layers
        if family != "gemma3n":
            assert cfg.query_pre_attn_scalar == p["qpas"], \
                f"query_pre_attn_scalar must be {p['qpas']}, got {cfg.query_pre_attn_scalar}"

    # -- Gemma 2: alternation, soft-capping, four norms per layer -------------
    if family == "gemma2":
        assert cfg.attn_logit_softcapping == p["attn_softcap"] and \
            cfg.final_logit_softcapping == p["final_softcap"], \
            "Gemma 2 soft-caps attention and final logits; without both this is a Gemma 1 block with a window"
        for name in ("input_layernorm", "post_attention_layernorm",
                     "pre_feedforward_layernorm", "post_feedforward_layernorm"):
            assert hasattr(layer, name), \
                f"Gemma 2 carries four RMSNorms per layer; {name} is missing"
        if cfg.num_hidden_layers >= 2:
            assert sliding[:2] == [True, False], \
                f"Gemma 2 alternates local then global; got {sliding[:2]}"

    # -- Gemma 3: two rope bases, 5:1 period, no caps -------------------------
    if family == "gemma3":
        assert cfg.attn_logit_softcapping is None and cfg.final_logit_softcapping is None, \
            "Gemma 3 dropped soft-capping; a cap here means the preset is really a Gemma 2"
        bases = _rope_bases(cfg, body)
        assert bases is not None, "Gemma 3 must carry a local AND a global rope base"
        assert bases[0] != bases[1], \
            ("the two rope bases must differ -- that is the whole point of the Gemma 3 "
             f"preset; got local={bases[0]} global={bases[1]}")
        if cfg.num_hidden_layers >= p["pattern"]:
            assert sliding.count(False) == cfg.num_hidden_layers // p["pattern"], \
                f"one layer in {p['pattern']} must be global; got {sliding}"
            assert sliding[p["pattern"] - 1] is False, \
                f"the global layer sits last in the period; got {sliding}"
        else:
            assert all(sliding), \
                ("a preset shorter than the period runs local layers only; "
                 "if a global one appeared here the period changed")

    # -- Gemma 3n: four things Gemma 3 does not have, and all of them ON ------
    if family == "gemma3n":
        assert cfg.final_logit_softcapping == 30.0, \
            ("3n kept the final cap Gemma 3 dropped; without it this preset is a "
             f"Gemma 3 block in a 3n config class, got {cfg.final_logit_softcapping}")
        for name in ("altup", "laurel", "per_layer_input_gate",
                     "per_layer_projection", "post_per_layer_input_norm"):
            assert hasattr(layer, name), \
                f"3n carries {name} per layer; without it the preset proves nothing new"
        assert cfg.altup_num_inputs == p["altup"] and cfg.altup_num_inputs > 1, \
            (f"altup needs more than one residual stream to be altup, got "
             f"{cfg.altup_num_inputs}")
        assert cfg.hidden_size_per_layer_input == p["hidden_per_layer"], \
            "the per-layer input embedding is what feeds per_layer_projection"

        # ACTIVATION SPARSITY, and it must be partial. A flat pattern would run
        # one branch of the MLP for every layer; the shipped E2B is sparse for
        # its leading layers and dense after, which is both branches in one run.
        sparse = [_layer_mlp(l).activation_sparsity for l in body.layers]
        assert any(s > 0.0 for s in sparse) and any(s == 0.0 for s in sparse), \
            (f"3n's preset must run sparse AND dense layers to cover both MLP "
             f"branches; got {sparse}")

        # KV SHARING, and it must actually fire on a layer whose donor exists.
        shared = [bool(getattr(_layer_mixer(l), "is_kv_shared_layer", False))
                  for l in body.layers]
        assert shared.count(True) == p["kv_shared"], \
            f"expected {p['kv_shared']} KV-shared layer(s), got {shared}"
        assert shared[-1] and not shared[0], \
            f"sharing is on the TRAILING layers; got {shared}"

        assert sliding.count(False) == cfg.num_hidden_layers // p["pattern"], \
            f"one layer in {p['pattern']} must be full attention; got {sliding}"

    if preset == "2b":
        assert cfg.hidden_size == 2048 and cfg.intermediate_size == 16384, \
            "2b preset must run at the real Gemma-2B width"
        assert cfg.num_key_value_heads == 1, \
            f"Gemma-2B is MQA; kv_heads={cfg.num_key_value_heads} makes this an ordinary GQA preset"
        assert n_rep == 8, f"2b must broadcast one kv head across 8 query heads, got n_rep={n_rep}"
        assert attn.k_proj.out_features == head_dim, \
            f"MQA k_proj projects to a single head ({head_dim}), got {attn.k_proj.out_features}"
        assert cfg.num_attention_heads * head_dim == cfg.hidden_size, \
            "2B couples heads*head_dim to hidden; that coupling is what 7b breaks"

    if preset == "7b":
        assert cfg.hidden_size == 3072 and cfg.intermediate_size == 24576, \
            "7b preset must run at the real Gemma-7B width"
        assert n_rep == 1, f"Gemma-7B is MHA; got n_rep={n_rep}"
        assert cfg.num_attention_heads * head_dim != cfg.hidden_size, \
            ("this preset exists to decouple head_dim from hidden_size; "
             f"heads*head_dim == hidden ({cfg.hidden_size}) means it proves nothing new")
        assert attn.o_proj.in_features == 4096 and attn.o_proj.out_features == 3072, \
            "7b's o_proj must shrink 4096 -> 3072"

    if preset == "2-2b":
        assert cfg.hidden_size == 2304 and cfg.intermediate_size == 9216, \
            "2-2b preset must run at the real Gemma2-2B width"
        assert n_rep == 2, f"Gemma2-2B is GQA 8/4; got n_rep={n_rep}"
        assert cfg.num_hidden_layers == 2, \
            ("Gemma 2's repeating unit is a local/global PAIR; one layer here would "
             "run the local variant and never the global one")

    if preset == "2-27b-scalar":
        assert cfg.query_pre_attn_scalar != head_dim, \
            ("this preset exists because 27B scales queries by 144 while its head_dim is 128; "
             f"scalar == head_dim ({head_dim}) means it proves nothing the other presets do not")

    if preset == "3-1b":
        assert cfg.hidden_size == 1152 and cfg.intermediate_size == 6912, \
            "3-1b preset must run at the real Gemma3-1B width"
        assert cfg.num_key_value_heads == 1 and n_rep == 4, \
            f"Gemma3-1B is MQA with 4 query heads; got kv={cfg.num_key_value_heads} n_rep={n_rep}"
        assert cfg.num_hidden_layers == p["pattern"], \
            (f"3-1b runs one full {p['pattern']}-layer period so the single global layer runs; "
             f"got {cfg.num_hidden_layers} layers")

    if preset == "3-4b":
        assert cfg.hidden_size == 2560 and cfg.intermediate_size == 10240, \
            "3-4b preset must run at the real Gemma3-4B width"
        assert cfg.num_hidden_layers < p["pattern"], \
            "3-4b is a width claim; if it ran a full period it would duplicate 3-1b's job at higher cost"

    # -- Griffin: the recurrence, and the one attention it alternates with -----
    if family == "recurrentgemma":
        kinds = [type(_layer_mixer(l)).__name__ for l in body.layers]
        assert cfg.num_hidden_layers == len(p["block_types"]), \
            (f"block_types cycles every {len(p['block_types'])} layers, so that is "
             f"the period; got {cfg.num_hidden_layers} layers")
        assert any("Recurrent" in k for k in kinds), \
            f"a RecurrentGemma preset must run the recurrence; got {kinds}"
        assert any("Attention" in k for k in kinds), \
            ("the period exists because the recurrence ALTERNATES with attention; "
             f"without one this preset is a recurrence-only model: {kinds}")
        # The band on that attention layer, same guard as Gemma 2/3's window.
        win = cfg.attention_window_size
        assert win < seq_len, \
            (f"attention_window_size ({win}) must be strictly inside seq ({seq_len}); "
             "the shipped 2048 at seq 32 is the whole row and tests nothing")
        # The temporal conv1d is what needs conv1d_to_conv2d registered. It is
        # the reason this model reached the backend at all, so it is asserted
        # rather than assumed.
        rec = next(m for m, k in zip((_layer_mixer(l) for l in body.layers), kinds)
                   if "Recurrent" in k)
        assert hasattr(rec, "conv_1d"), \
            "Griffin's recurrent block convolves along time; no conv_1d here"

    # -- multimodal: the tower, the projector, and the splice ------------------
    if _is_vlm(preset):
        tower, proj = _vision_tower(model), _projector(model)
        assert tower is not None and proj is not None, \
            "a multimodal preset must carry a vision tower and a projector"
        assert type(tower).__name__.startswith("Siglip"), \
            (f"both Gemma VLMs use a SigLIP tower; got {type(tower).__name__}")

        v = p["vision"]
        patches = (v["image"] // v["patch"]) ** 2
        assert full_cfg.image_token_index == _IMAGE_TOKEN

        # The placeholder count must equal what the tower emits, or the splice
        # has nothing coherent to scatter into and transformers refuses the
        # forward.  The two families differ here and that is the point of
        # running both: Gemma 3 POOLS the tower's patches down to
        # mm_tokens_per_image, PaliGemma emits one token per patch.
        if p["family"] == "gemma3_mm":
            assert full_cfg.mm_tokens_per_image == p["mm_tokens"]
            assert p["mm_tokens"] < patches, \
                (f"Gemma 3 pools {patches} patches down to mm_tokens_per_image; "
                 f"{p['mm_tokens']} == {patches} would leave the pooling untested")
        else:
            assert p["mm_tokens"] == patches, \
                (f"PaliGemma emits one token per patch, so mm_tokens must be {patches}, "
                 f"got {p['mm_tokens']}")

    if p["tie"]:
        head = _lm_head(model)
        assert head is not None, "tie check needs the LM head; run this preset with --part lm"
        assert head.weight.data_ptr() == body.embed_tokens.weight.data_ptr(), \
            "tie_word_embeddings was requested but lm_head does not share the embedding storage"


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
def run_gemma(
    device,
    preset="tiny",
    part="lm",
    batch=1,
    seq_len=None,
    dtype="float32",
    compile_model=True,
    rtol=1e-2,
    atol=1e-2,
):
    p = _PRESETS[preset]
    _, _, mod_mod, body_name, lm_name = _FAMILIES[p["family"]]
    cls = _import(mod_mod, lm_name if part == "lm" else body_name)

    torch_dtype = _dtype_from_str(dtype)
    cfg, seq_len = _build_config(preset, seq_len)

    # Seed before construction: config-random weights otherwise differ per run,
    # so the worst element wanders across the threshold and the test is flaky.
    torch.manual_seed(0)
    model_cpu = cls(cfg).to(dtype=torch_dtype).eval()

    tcfg = _text_cfg(cfg)
    text_family = _TEXT_FAMILY.get(p["family"], p["family"])
    head_dim = getattr(tcfg, "head_dim", None) or tcfg.hidden_size // tcfg.num_attention_heads
    n_rep = tcfg.num_attention_heads // tcfg.num_key_value_heads
    print(f"preset={preset} family={p['family']} part={part} layers={tcfg.num_hidden_layers} "
          f"hidden={tcfg.hidden_size} heads={tcfg.num_attention_heads}/{tcfg.num_key_value_heads} "
          f"head_dim={head_dim} n_rep={n_rep} interm={tcfg.intermediate_size} "
          f"vocab={tcfg.vocab_size} seq={seq_len} dtype={dtype}")
    act = getattr(tcfg, "hidden_activation", None) or getattr(tcfg, "hidden_act", None)
    print(f"  gemma: act={act} embed_scale=sqrt({tcfg.hidden_size})={tcfg.hidden_size ** 0.5:.4f} "
          f"tie={tcfg.tie_word_embeddings}")
    if text_family in ("gemma2", "gemma3"):
        _body = _text_body(model_cpu)
        sliding = ["local" if _layer_is_sliding(_body, i, tcfg) else "GLOBAL"
                   for i in range(tcfg.num_hidden_layers)]
        print(f"  window={tcfg.sliding_window} (seq={seq_len}) qpas={tcfg.query_pre_attn_scalar} "
              f"layers={'/'.join(sliding)}")
        if text_family == "gemma2":
            print(f"  softcap: attn={tcfg.attn_logit_softcapping} final={tcfg.final_logit_softcapping}")
        else:
            _b = _rope_bases(tcfg, _body)
            print(f"  rope: local={_b[0]} global={_b[1]} "
                  f"pattern={getattr(tcfg, 'sliding_window_pattern', None)}")
    if _is_vlm(preset):
        v = p["vision"]
        print(f"  vision: {type(_vision_tower(model_cpu)).__name__} hidden={v['hidden']} "
              f"layers={v['layers']} image={v['image']} patch={v['patch']} "
              f"patches={(v['image'] // v['patch']) ** 2} image_tokens={p['mm_tokens']}")
    print("model params:", sum(x.numel() for x in model_cpu.parameters()))

    _assert_character(preset, cfg, model_cpu, seq_len)

    g = torch.Generator().manual_seed(0)

    if _is_vlm(preset):
        # Tokens are drawn ABOVE the placeholder so a random draw cannot become
        # one and break the count the tower has to match.  No explicit mask:
        # a VLM's mask depends on which positions are image, and building one
        # here by hand would test our arithmetic rather than the model's.
        input_ids = torch.randint(_IMAGE_TOKEN + 1, tcfg.vocab_size, (batch, seq_len),
                                  generator=g, dtype=torch.int64)
        input_ids[:, :p["mm_tokens"]] = _IMAGE_TOKEN
        v = p["vision"]
        pixel_values = torch.randn(batch, 3, v["image"], v["image"],
                                   generator=g, dtype=torch.float32).to(torch_dtype)
        cpu_inputs = {"input_ids": input_ids, "pixel_values": pixel_values}
    else:
        input_ids = torch.randint(0, tcfg.vocab_size, (batch, seq_len),
                                  generator=g, dtype=torch.int64)
        # The explicit 4D causal mask is test_llama3x.py's, kept so a failure
        # here is comparable to that file's rather than to whatever
        # _update_causal_mask decided to build.
        min_dtype = torch.finfo(torch_dtype).min
        causal_mask = torch.full((seq_len, seq_len), fill_value=min_dtype, dtype=torch_dtype)
        if seq_len > 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        attn_mask = causal_mask[None, None, :, :].expand(batch, 1, -1, -1)
        cpu_inputs = {"input_ids": input_ids, "attention_mask": attn_mask}

    cpu_out = _out(model_cpu(**cpu_inputs))

    model_npu = copy.deepcopy(model_cpu).to(device).eval()
    if compile_model:
        model_npu = torch.compile(model_npu, dynamic=False)
    npu_out = _out(model_npu(**{k: v.to(device) for k, v in cpu_inputs.items()}))

    check_result(f"Gemma {part} ({preset})", npu_out, cpu_out, rtol=rtol, atol=atol)
    print("Max diff > ", torch.max(torch.abs(npu_out.cpu() - cpu_out)))
    print("Gemma Simulation Done")


# One preset per generation, because a generation is what this file adds and a
# gate that skipped one would leave it claimed-but-unrun.  7b is Gemma 1 at its
# real width, 2-2b is the Gemma 2 local/global pair, 3-1b is one full Gemma 3
# period.
#
# MEASURED AS CI RUNS IT -- all three in ONE process, one dump path: 1137s and
# 64 unique kernels, against the sweep's 1800s timeout.  Run separately they sum
# to 1198s, so the three presets share a little codegen rather than colliding in
# it: each preset's max diff is identical to its isolated run (7.6294e-06,
# 5.7220e-06, 4.5896e-06), which is what rules out the shared-dump hazard that
# the per-preset numbers in the docstring were measured to avoid.
_GATE = ("7b", "2-2b", "3-1b")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemma 1/2/3 end to end on the Triton route")
    parser.add_argument("--preset", type=str, default=None, choices=sorted(_PRESETS),
                        help=f"one preset; the default runs the gate ({', '.join(_GATE)})")
    parser.add_argument("--part", type=str, default="lm", choices=["lm", "body"],
                        help="lm = ...ForCausalLM (adds the vocab projection); body = ...Model")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--no-compile", dest="compile", action="store_false", default=True)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    args = parser.parse_args()

    sys.path.append(os.environ.get("TORCHSIM_DIR", "/workspace/PyTorchSim"))
    torch.compiler.is_compiling = lambda: True  # FIXME. Same as test_llama.py.

    for preset in ([args.preset] if args.preset else _GATE):
        run_gemma(
            torch.device("npu:0"),
            preset=preset,
            part=args.part,
            batch=args.batch,
            seq_len=args.seq_len,
            dtype=args.dtype,
            compile_model=args.compile,
            rtol=args.rtol,
            atol=args.atol,
        )
