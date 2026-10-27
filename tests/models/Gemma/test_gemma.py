"""Gemma 1, 2 and 3 (text) on the Triton codegen route.

Built from the ``transformers`` configs with random weights, so it needs no
network and no checkpoint.  The models are ``transformers``' own
``GemmaForCausalLM`` / ``Gemma2ForCausalLM`` / ``Gemma3ForCausalLM`` -- nothing
here reimplements a block, so what the test exercises is the shipped module,
not our reading of it.  One file covers all three generations because they are
deltas on one block, the way the Qwen file covers its family.

EVERY PRESET IS A SHIPPED CONFIG.  Hidden, heads, kv heads, head_dim,
intermediate, vocab, max positions, rope bases and scaling, sliding window,
query_pre_attn_scalar and the soft-caps are the published values of the
checkpoint named over the preset.  Only the layer count is cut, and HOW DEEP A
PRESET RUNS IS DECIDED BY THE REPEATING UNIT, not by taste:

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

THE SLIDING WINDOW IS THE SHIPPED ONE, SO IT BITES ONLY PAST ITS LENGTH.  The
band ``torch.tril(..., diagonal=-window)`` selects nothing until seq exceeds the
window: 3-1b's 512 from ``--seq 1024`` on, 3-4b's 1024 from 2048, Gemma 2's
4096 only at 8192.  Below that the local layers compute what the global ones
do; the alternation is still built, it just masks no element.

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
  * vocab 256000 (262144 / 262208 in Gemma 3) with a tied lm_head -- the
    largest vocab in the suite, so the vocab projection shares the embedding.

WHY 2B AND 7B ARE BOTH HERE.  They are not the same block at two widths:

  * 2B is MQA (kv_heads 1) and heads * head_dim == hidden (8 * 256 == 2048).
  * 7B is MHA (kv_heads 16) and heads * head_dim != hidden (16 * 256 == 4096
    against 3072 hidden), so q_proj is not square and o_proj shrinks 4096 ->
    3072.

Dropping either one drops a claim.  ``_assert_character`` cashes every claim
above against the built model, so a preset cannot pass while proving nothing.
Input is batch 1, one prefill of ``--seq`` tokens (default 128), in bfloat16 by
default (``--dtype float32`` for the old run).  ``--part layer`` runs the
preset's first decoder layer alone (tests/one_layer.py) on ``--seq S`` or on one
``--decode S`` step after a KV cache, with the masks the model builds for that
layer's type -- the sliding band included, which the explicit causal mask of the
other parts leaves out.

    source /workspace/torchsim-env.sh
    python tests/models/Gemma/test_gemma.py --preset 7b
    python tests/models/Gemma/test_gemma.py --preset 7b --part layer --decode 1024
"""

import argparse
import copy
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from result_check import check_result, f32, f32_reference
import one_layer


# Each preset is the published config of the checkpoint named over it; only `layers` is cut.
_PRESETS = {
    # google/gemma-2b: MQA (n_rep = 8), head_dim 256, GeGLU 16384.
    "2b": dict(family="gemma", layers=1, hidden=2048, heads=8, kv_heads=1,
               head_dim=256, intermediate=16384, vocab=256000, max_pos=8192,
               rope_theta=10000.0, tie=True),

    # google/gemma-7b: MHA, head_dim 256 decoupled from 3072 hidden, GeGLU 24576.
    "7b": dict(family="gemma", layers=1, hidden=3072, heads=16, kv_heads=16,
               head_dim=256, intermediate=24576, vocab=256000, max_pos=8192,
               rope_theta=10000.0, tie=True),

    # google/gemma-2-2b, TWO layers -- one local, one global, Gemma 2's whole repeating unit.
    "2-2b": dict(family="gemma2", layers=2, hidden=2304, heads=8, kv_heads=4,
                 head_dim=256, intermediate=9216, vocab=256000, max_pos=8192,
                 rope_theta=10000.0, tie=True, window=4096, attn_softcap=50.0,
                 final_softcap=30.0, qpas=256),

    # google/gemma-2-27b: the one Gemma whose query_pre_attn_scalar (144) is NOT head_dim
    # (128), so the pre-attention scale is not head_dim**-0.5.  One layer: a claim about a scalar.
    "2-27b": dict(family="gemma2", layers=1, hidden=4608, heads=32, kv_heads=16,
                  head_dim=128, intermediate=36864, vocab=256000, max_pos=8192,
                  rope_theta=10000.0, tie=True, window=4096, attn_softcap=50.0,
                  final_softcap=30.0, qpas=144),

    # google/gemma-3-1b-pt, SIX layers -- one full 5:1 period, so five local layers and the
    # single global one all run.
    "3-1b": dict(family="gemma3", layers=6, hidden=1152, heads=4, kv_heads=1,
                 head_dim=256, intermediate=6912, vocab=262144, max_pos=32768, tie=True,
                 window=512, rope_local=10000.0, rope_global=1000000.0, rope_scaling=None,
                 pattern=6, qpas=256),

    # google/gemma-3-4b-pt (text_config), one layer.  A width claim only: at pattern 6 layer 0
    # is local, so this preset says nothing about the period -- 3-1b covers that.
    "3-4b": dict(family="gemma3", layers=1, hidden=2560, heads=8, kv_heads=4,
                 head_dim=256, intermediate=10240, vocab=262208, max_pos=131072, tie=True,
                 window=1024, rope_local=10000.0, rope_global=1000000.0,
                 rope_scaling=dict(rope_type="linear", factor=8.0), pattern=6, qpas=256),
}

# (rtol, atol) per dtype; the bf16 atol is 3x the worst bf16 eager-vs-Inductor CPU diff at S = 128.
_TOL = {"bfloat16": (5e-2, 0.4), "float32": (1e-2, 1e-2)}

_FAMILIES = {
    "gemma": ("transformers.models.gemma.configuration_gemma", "GemmaConfig",
              "transformers.models.gemma.modeling_gemma", "GemmaModel", "GemmaForCausalLM"),
    "gemma2": ("transformers.models.gemma2.configuration_gemma2", "Gemma2Config",
               "transformers.models.gemma2.modeling_gemma2", "Gemma2Model", "Gemma2ForCausalLM"),
    "gemma3": ("transformers.models.gemma3.configuration_gemma3", "Gemma3TextConfig",
               "transformers.models.gemma3.modeling_gemma3", "Gemma3TextModel", "Gemma3ForCausalLM"),
}


def _descend(model, attr, depth=4):
    """Walk `.model` until a module has `attr`.

    `lm_head` sits on the ForCausalLM and `layers` on the Model inside it, so
    naming the attribute wanted and walking to it serves both, with --part lm
    and --part body alike.
    """
    seen = model
    for _ in range(depth):
        if hasattr(seen, attr):
            return seen
        nxt = getattr(seen, "model", None)
        if nxt is None or nxt is seen:
            break
        seen = nxt
    return None


def _lm_head(model):
    holder = _descend(model, "lm_head")
    return getattr(holder, "lm_head", None) if holder is not None else None


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


def _build_config(preset):
    """The preset's published config, with `layers` as its only cut.

    Gemma 1's MLP reads hidden_act; 2 and 3 read hidden_activation, so both are set.  Eager
    attention keeps the graph free of the sdpa dispatch.  Gemma 2/3 default
    cache_implementation to "hybrid", which only matters to generate(), so it is cleared.
    """
    p = _PRESETS[preset]
    cfg_mod, cfg_name, _, _, _ = _FAMILIES[p["family"]]
    Config = _import(cfg_mod, cfg_name)
    family = p["family"]

    kwargs = dict(
        vocab_size=p["vocab"],
        hidden_size=p["hidden"],
        num_attention_heads=p["heads"],
        num_key_value_heads=p["kv_heads"],
        head_dim=p["head_dim"],
        intermediate_size=p["intermediate"],
        num_hidden_layers=p["layers"],
        max_position_embeddings=p["max_pos"],
        rms_norm_eps=1e-6,
        hidden_act="gelu_pytorch_tanh",
        hidden_activation="gelu_pytorch_tanh",
        attention_bias=False,
        tie_word_embeddings=p["tie"],
        use_cache=False,
        attn_implementation="eager",
    )

    if family in ("gemma", "gemma2"):
        kwargs["rope_theta"] = p["rope_theta"]
    if family in ("gemma2", "gemma3"):
        kwargs["sliding_window"] = p["window"]
        kwargs["query_pre_attn_scalar"] = p["qpas"]
        kwargs["cache_implementation"] = None
    if family == "gemma2":
        kwargs["attn_logit_softcapping"] = p["attn_softcap"]
        kwargs["final_logit_softcapping"] = p["final_softcap"]
    if family == "gemma3":
        kwargs["rope_theta"] = p["rope_global"]
        kwargs["rope_local_base_freq"] = p["rope_local"]
        kwargs["rope_scaling"] = p["rope_scaling"]
        kwargs["sliding_window_pattern"] = p["pattern"]
        kwargs["attn_logit_softcapping"] = None
        kwargs["final_logit_softcapping"] = None

    return Config(**kwargs)


def _text_body(model):
    """The stack of decoder layers, under whichever wrappers this model has."""
    body = _descend(model, "layers")
    assert body is not None, f"no layer stack under {type(model).__name__}"
    return body


def _layer0(model):
    return _text_body(model).layers[0]


def _assert_character(preset, cfg, model):
    """Fail if the model did not actually get the feature the preset claims.

    A preset is a claim about shape and about which Gemma the block is; this is
    where the claim is cashed.  Every check guards against a run that passes
    without exercising anything the suite does not already cover.

    The (1 + w) offset is checked on the norm module rather than on the config, because the
    config cannot express it: a zero weight that still passes the activation through is only
    visible in the module's own arithmetic.
    """
    p = _PRESETS[preset]
    family = p["family"]
    body = _text_body(model)
    layer = _layer0(model)
    attn = layer.self_attn
    n_rep = cfg.num_attention_heads // cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads

    norm = layer.input_layernorm
    norm_name = type(norm).__name__
    assert norm_name.startswith("Gemma") and norm_name.endswith("RMSNorm"), \
        f"expected a Gemma*RMSNorm, got {norm_name}; the (1 + w) offset is the point of this test"
    probe = torch.ones(1, 1, cfg.hidden_size)
    with torch.no_grad():
        scaled = norm(probe)
    assert torch.allclose(scaled, probe, atol=1e-5), \
        (f"{norm_name} with a zero weight must be the identity on a unit vector; "
         "a Llama-style x*w would return zeros here")

    assert attn.q_proj.bias is None, "Gemma has attention_bias=False; a bias here makes this a Qwen2 block"
    mlp = layer.mlp
    assert type(mlp).__name__.endswith(("MLP", "Mlp"))
    for name in ("gate_proj", "up_proj", "down_proj"):
        assert hasattr(mlp, name), f"GeGLU needs {name}; this MLP is not gated"
    act = type(mlp.act_fn).__name__.lower()
    assert "gelu" in act or "tanh" in act, \
        f"the gated MLP must use a gelu, got {type(mlp.act_fn).__name__}; silu here means this is a SwiGLU block"

    if preset != "2-27b":
        assert head_dim == 256, f"this preset runs head_dim 256, got {head_dim}"
        assert head_dim == 2 * 128, "the suite's previous maximum was 128; this preset exists to double it"

    if family == "gemma":
        assert getattr(cfg, "sliding_window", None) is None, \
            "Gemma 1 has no sliding window; a window here means the wrong config class was built"
        assert getattr(cfg, "attn_logit_softcapping", None) is None, \
            "Gemma 1 does not soft-cap; capping here means this is a Gemma 2 block"

    if family in ("gemma2", "gemma3"):
        assert cfg.sliding_window == p["window"], \
            f"sliding_window must be the shipped {p['window']}, got {cfg.sliding_window}"
        sliding = [_layer_is_sliding(body, i, cfg)
                   for i in range(cfg.num_hidden_layers)]
        assert len(sliding) == cfg.num_hidden_layers
        assert cfg.query_pre_attn_scalar == p["qpas"], \
            f"query_pre_attn_scalar must be {p['qpas']}, got {cfg.query_pre_attn_scalar}"

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

    if family == "gemma3":
        assert cfg.attn_logit_softcapping is None and cfg.final_logit_softcapping is None, \
            "Gemma 3 dropped soft-capping; a cap here means the preset is really a Gemma 2"
        bases = _rope_bases(cfg, body)
        assert bases == (p["rope_local"], p["rope_global"]), \
            f"Gemma 3 must carry local {p['rope_local']} and global {p['rope_global']}; got {bases}"
        if cfg.num_hidden_layers >= p["pattern"]:
            assert sliding.count(False) == cfg.num_hidden_layers // p["pattern"], \
                f"one layer in {p['pattern']} must be global; got {sliding}"
            assert sliding[p["pattern"] - 1] is False, \
                f"the global layer sits last in the period; got {sliding}"
        else:
            assert all(sliding), \
                ("a preset shorter than the period runs local layers only; "
                 "if a global one appeared here the period changed")

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

    if preset == "2-27b":
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
    preset="7b",
    part="lm",
    batch=1,
    seq_len=128,
    dtype="bfloat16",
    compile_model=True,
    rtol=None,
    atol=None,
    decode=False,
):
    """The preset's `part` on one prefill of `seq_len` tokens, or with `decode` (--part layer) one step at
    position seq_len - 1 after a KV cache; CPU against `device`, at `dtype`'s `_TOL` unless given.
    lm and body take test_llama3x.py's explicit 4D causal mask, layer the masks the model builds."""
    p = _PRESETS[preset]
    _, _, mod_mod, body_name, lm_name = _FAMILIES[p["family"]]
    cls = _import(mod_mod, body_name if part == "body" else lm_name)

    torch_dtype = _dtype_from_str(dtype)
    tol = _TOL.get(dtype, _TOL["float32"])
    rtol, atol = tol[0] if rtol is None else rtol, tol[1] if atol is None else atol
    cfg = _build_config(preset)
    if seq_len > cfg.max_position_embeddings:
        raise ValueError(f"--seq {seq_len} is past {preset}'s {cfg.max_position_embeddings} positions")

    torch.manual_seed(0)
    model_cpu = cls(cfg).to(dtype=torch_dtype).eval()

    family = p["family"]
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    n_rep = cfg.num_attention_heads // cfg.num_key_value_heads
    print(f"preset={preset} family={family} part={part} layers={cfg.num_hidden_layers} "
          f"hidden={cfg.hidden_size} heads={cfg.num_attention_heads}/{cfg.num_key_value_heads} "
          f"head_dim={head_dim} n_rep={n_rep} interm={cfg.intermediate_size} "
          f"vocab={cfg.vocab_size} seq={seq_len} dtype={dtype}")
    act = getattr(cfg, "hidden_activation", None) or getattr(cfg, "hidden_act", None)
    print(f"  gemma: act={act} embed_scale=sqrt({cfg.hidden_size})={cfg.hidden_size ** 0.5:.4f} "
          f"tie={cfg.tie_word_embeddings}")
    if family in ("gemma2", "gemma3"):
        _body = _text_body(model_cpu)
        sliding = ["local" if _layer_is_sliding(_body, i, cfg) else "GLOBAL"
                   for i in range(cfg.num_hidden_layers)]
        print(f"  window={cfg.sliding_window} (seq={seq_len}) qpas={cfg.query_pre_attn_scalar} "
              f"layers={'/'.join(sliding)}")
        if family == "gemma2":
            print(f"  softcap: attn={cfg.attn_logit_softcapping} final={cfg.final_logit_softcapping}")
        else:
            _b = _rope_bases(cfg, _body)
            print(f"  rope: local={_b[0]} global={_b[1]} "
                  f"pattern={getattr(cfg, 'sliding_window_pattern', None)}")
    print("model params:", sum(x.numel() for x in model_cpu.parameters()))

    _assert_character(preset, cfg, model_cpu)

    g = torch.Generator().manual_seed(0)
    input_ids = torch.randint(0, cfg.vocab_size, (batch, seq_len),
                              generator=g, dtype=torch.int64)
    if part == "layer":
        body = _text_body(model_cpu)
        cpu_out, npu_out = one_layer.run(body, body.layers[0], cfg, input_ids, device,
                                         decode=decode, compile_model=compile_model)
    else:
        min_dtype = torch.finfo(torch_dtype).min
        causal_mask = torch.full((seq_len, seq_len), fill_value=min_dtype, dtype=torch_dtype)
        if seq_len > 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        attn_mask = causal_mask[None, None, :, :].expand(batch, 1, -1, -1)
        cpu_inputs = {"input_ids": input_ids, "attention_mask": attn_mask}

        cpu_out = _out(f32_reference(model_cpu)(**f32(cpu_inputs)))

        model_npu = copy.deepcopy(model_cpu).to(device).eval()
        if compile_model:
            model_npu = torch.compile(model_npu, dynamic=False)
        npu_out = _out(model_npu(**{k: v.to(device) for k, v in cpu_inputs.items()}))

    check_result(f"Gemma {part} ({preset})", npu_out, cpu_out, rtol=rtol, atol=atol)
    print("Max diff > ", torch.max(torch.abs(npu_out.cpu() - cpu_out)))
    print("Gemma Simulation Done")


# One preset per generation, because a generation is what this file adds and a gate that
# skipped one would leave it claimed-but-unrun: 7b is Gemma 1, 2-2b the Gemma 2 local/global
# pair, 3-1b one full Gemma 3 period.
_GATE = ("7b", "2-2b", "3-1b")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemma 1/2/3 end to end on the Triton route")
    parser.add_argument("--preset", type=str, default=None, choices=sorted(_PRESETS),
                        help=f"one preset; the default runs the gate ({', '.join(_GATE)})")
    parser.add_argument("--part", type=str, default="lm", choices=["lm", "body", "layer"],
                        help="lm = ...ForCausalLM (adds the vocab projection); body = ...Model; "
                             "layer = the first decoder layer alone")
    parser.add_argument("--batch", type=int, default=1)
    one_layer.add_grid_args(parser)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--no-compile", dest="compile", action="store_false", default=True)
    parser.add_argument("--rtol", type=float, default=None, help="default: _TOL of --dtype")
    parser.add_argument("--atol", type=float, default=None, help="default: _TOL of --dtype")
    args = parser.parse_args()
    if args.decode is not None and args.part != "layer":
        parser.error("--decode runs with --part layer")

    sys.path.append(os.environ.get("PYTORCHSIM_ROOT_PATH", "/workspace/PyTorchSim"))
    torch.compiler.is_compiling = lambda: True  # FIXME. Same as test_llama.py.

    for preset in ([args.preset] if args.preset else _GATE):
        run_gemma(
            torch.device("npu:0"),
            preset=preset,
            part=args.part,
            batch=args.batch,
            seq_len=args.decode or args.seq,
            decode=args.decode is not None,
            dtype=args.dtype,
            compile_model=args.compile,
            rtol=args.rtol,
            atol=args.atol,
        )
