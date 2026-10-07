"""The Qwen families transformers 5.x added, on the Triton codegen route.

`test_qwen.py` covers the four families that existed when it was written --
qwen2, qwen2_moe, qwen3, qwen3_moe -- plus the two VL towers in
`test_qwen_vl.py`.  transformers 5.15.0 ships nine more Qwen model types, and
this file is the seven of them that build from a config alone and run.  Same
shape as the others: random weights, no network, no checkpoint, no remote code,
one layer per family.

WHAT EACH ONE ADDS, and why it is not covered by the four already here:

  * qwen3_vl / qwen3_vl_moe   Qwen3's block under a vision tower.  The LANGUAGE
                              half is what this file runs; the vision tower
                              needs pixel inputs and is `test_qwen_vl.py`'s
                              shape of problem, not this one's.
  * qwen3_5 / qwen3_5_moe     A LINEAR-ATTENTION line.  The recurrence unrolls
                              per position, so seq 16 lowers to 121 and 134
                              kernels against the 19 a Qwen3 layer takes --
                              that ratio is the thing this preset pins, and it
                              is why the seq axis matters here more than width.
                              It also puts a Conv1d in the block, which needs
                              inductor_templates' conv1d-to-conv2d registration
                              or it falls to an extern kernel this device has
                              no implementation for.
  * qwen3_next                Hybrid attention -- linear layers interleaved with
                              full ones -- so its cache is not an Attention
                              cache and `use_cache=False` is required to build
                              one at all.  132 kernels.
  * qwen2_audio / qwen3_asr   Audio models.  `--audio` feeds a mel spectrogram
                              so the ENCODER runs -- conv stack, windowed
                              attention, projector -- and the result is spliced
                              into the text sequence.  Without it these run on
                              text alone (17 kernels), which says nothing about
                              the encoder.  See `run_audio` for the four config
                              knobs that are not free to shrink.

WHAT IS NOT COVERED, and it is a large half:

  * THE OMNI PAIR.  qwen2_5_omni and qwen3_omni_moe are a thinker plus a talker
    under one config, and the thinker cannot be constructed on its own: it reads
    `vision_start_token_id`, which no config in the pair carries until the parent
    builds it.  Building the parent means building the talker's vocoder too.
  * DEPTH.  One layer, as everywhere in this suite.
  * WIDTH.  These are shape claims at a small width, not the real ones.

VERIFIED (2026-08-14, transformers 5.15.0, one layer, seq 16):

    qwen3_vl       5.96e-07     qwen3_5       2.38e-07   121 kernels
    qwen3_vl_moe   7.15e-07     qwen3_5_moe   1.49e-07   134 kernels
    qwen2_audio    7.15e-07     qwen3_next    2.38e-07   132 kernels
    qwen3_asr      7.15e-07

    --audio        qwen2_audio  1.31e-06   168s   32 mel x 32 frames,   8 tokens
                   qwen3_asr    1.19e-06   216s   32 mel x 100 frames, 13 tokens

    The two audio figures are max abs diff read off the same shapes at the same
    seed before this ran as a test; `test_result` itself reports only the
    verdict, and what it reported was Passed at rtol 1e-2 / atol 1e-4.

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
from _pytorchsim_utils import test_result

FAMILIES = ("qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe", "qwen3_next",
            "qwen2_audio", "qwen3_asr")

#: The two that have an audio tower, and so accept `--audio`.
AUDIO = ("qwen2_audio", "qwen3_asr")

#: The gate runs one of each KIND rather than all seven: a linear-attention
#: family, its MoE counterpart, and a plain Qwen3 block under another tower.
_GATE = ("qwen3_5", "qwen3_vl_moe")

#: Shrinks cost, not character.  Width carries head_dim and the SwiGLU ratio;
#: depth, vocab and expert count are the cheap axes.
SMALL = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=1,
             num_attention_heads=4, num_key_value_heads=2, vocab_size=256,
             max_position_embeddings=128, head_dim=16,
             num_experts=4, num_experts_per_tok=2, moe_intermediate_size=64,
             shared_expert_intermediate_size=64, decoder_sparse_step=1,
             num_local_experts=4, tie_word_embeddings=True,
             encoder_layers=1, d_model=64, num_mel_bins=32,
             max_source_positions=16, n_window_infer=100, output_dim=64)


def _config_for(family):
    """A small config of this family's own class, with nothing guessed.

    Every knob in SMALL that the class does not take is DROPPED rather than
    forced: these families disagree about which of them exist, and setting one
    that a class ignores would look like a claim about a shape it does not have.
    """
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    cls = CONFIG_MAPPING[family]
    names = set()
    for k in (cls,) + tuple(cls.__mro__):
        try:
            names |= set(inspect.signature(k.__init__).parameters)
        except (TypeError, ValueError):
            pass
    cfg = cls(**{k: v for k, v in SMALL.items() if k in names})
    for sub in ("text_config", "vision_config", "audio_config"):
        s = getattr(cfg, sub, None)
        if s is None:
            continue
        for k, v in SMALL.items():
            if hasattr(s, k):
                setattr(s, k, v)
    _fit_special_tokens(cfg)
    return cfg


def _fit_special_tokens(cfg, depth=0):
    """Bring every special token id inside the shrunken vocabulary.

    A token id is ABSOLUTE and the vocabulary here is not: qwen2_audio's audio
    placeholder is 151646 against 256, so it can never appear in `input_ids` and
    the model then reports zero audio tokens for a batch that has audio.  The
    vocabulary a given id is measured against may live on a SUB config, which is
    why this looks there before deciding an id is in range.
    """
    if cfg is None or depth > 3:
        return
    vocab = getattr(cfg, "vocab_size", None)
    if vocab is None:
        for sub in ("text_config", "thinker_config"):
            t = getattr(cfg, sub, None)
            if t is not None and getattr(t, "vocab_size", None):
                vocab = t.vocab_size
                break
    for name in dir(cfg):
        if not name.endswith(("_token_id", "_token_index")):
            continue
        v = getattr(cfg, name, None)
        if isinstance(v, int) and vocab and v >= vocab:
            setattr(cfg, name, hash(name) % max(vocab - 8, 1) + 4)
    for sub in ("text_config", "vision_config", "audio_config"):
        _fit_special_tokens(getattr(cfg, sub, None), depth + 1)


def build(family):
    """The model, and which Auto class accepted it."""
    from transformers import AutoModel, AutoModelForCausalLM
    cfg = _config_for(family)
    for maker in (AutoModelForCausalLM, AutoModel):
        try:
            return cfg, maker.from_config(cfg), maker.__name__
        except Exception:
            continue
    raise RuntimeError(f"no Auto class accepted {family}")


def run_family(family, seq_len=16, rtol=1e-2, atol=1e-4, compile_it=True):
    torch.manual_seed(0)
    cfg, model, how = build(family)
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"family={family} {type(model).__name__} via {how} "
          f"params={n/1e6:.2f}M seq={seq_len}")

    ids = torch.randint(0, 64, (1, seq_len))
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
        cpu_out = first(model(**kw)).clone()

    model_npu = model.to("npu")
    kw_npu = {k: (v.to("npu") if torch.is_tensor(v) else v) for k, v in kw.items()}
    runner = torch.compile(model_npu, dynamic=False) if compile_it else model_npu
    with torch.no_grad():
        npu_out = first(runner(**kw_npu)).cpu()

    test_result(f"Qwen {family}", npu_out, cpu_out, rtol=rtol, atol=atol)


def run_audio(family, rtol=1e-2, atol=1e-4, compile_it=True):
    """The AUDIO tower: a mel spectrogram in, spliced into the text sequence.

    FOUR CONFIG KNOBS ARE NOT FREE HERE, and each one cost a run:

      * THE FEATURE LENGTH is validated, not accepted.  qwen2_audio wants
        exactly `max_source_positions * conv1.stride * conv2.stride` frames;
        qwen3_asr wants a multiple of `n_window * 2`.  Read it off the config
        rather than picking one.
      * `n_window` IS NOT SHRINKABLE, and it looks like it is.  qwen3_asr's
        post-CNN length is `... + (frames // (n_window * 2)) * 13` with the 13
        WRITTEN IN -- it is the conv stack's output for a 100-frame chunk and
        nothing recomputes it.  Set the window to 8 and the encoder chunks its
        attention by 13 while the CNN handed it 2, so `torch.split` refuses:
        "split_sizes to sum exactly to 2 ... but got split_sizes=[13]".
      * THE PLACEHOLDER COUNT must equal the number of audio embeddings, and
        `frames // 2` is not that number.  qwen2_audio carries the arithmetic as
        a method, qwen3_asr as a module-level function the model does not
        expose, so look in both and REFUSE to guess -- a guess put 50 tokens
        against 13 features.
      * `output_dim` IS THE PROJECTOR'S FAR END and is not derived from the text
        config.  Left at its stock 3584 against a 64-wide language model the
        counts agree and the check still fails, because it compares `numel`:
        "tokens: 13, features: 13".
    """
    import inspect as _inspect
    torch.manual_seed(0)
    cfg, model, _ = build(family)
    model.eval()
    tower = model.audio_tower
    ac = cfg.audio_config
    frames = (ac.max_source_positions * 2 if hasattr(ac, "max_source_positions")
              else ac.n_window * 2)
    feats = torch.randn(1, ac.num_mel_bins, frames)

    lengths = (getattr(model, "_get_feat_extract_output_lengths", None)
               or getattr(tower, "_get_feat_extract_output_lengths", None)
               or getattr(sys.modules[type(model).__module__],
                          "_get_feat_extract_output_lengths", None))
    if lengths is None:
        raise RuntimeError(f"{family}: no output-length arithmetic to ask")
    kw_len = ({"n_window": ac.n_window}
              if "n_window" in _inspect.signature(lengths).parameters else {})
    got = lengths(torch.tensor([frames]), **kw_len)
    n_audio = int((got[-1] if isinstance(got, tuple) else got).flatten()[0])

    tok = getattr(cfg, "audio_token_id", None) or cfg.audio_token_index
    ids = torch.cat([torch.full((1, n_audio), tok, dtype=torch.long),
                     torch.randint(0, 16, (1, 8))], dim=1)
    sig = set(inspect.signature(model.forward).parameters)
    kw = {"input_ids": ids, "input_features": feats}
    if "feature_attention_mask" in sig:
        kw["feature_attention_mask"] = torch.ones(1, frames, dtype=torch.long)
    if "input_features_mask" in sig:
        kw["input_features_mask"] = torch.ones(1, frames, dtype=torch.bool)
    if "attention_mask" in sig:
        kw["attention_mask"] = torch.ones_like(ids)
    if "use_cache" in sig:
        kw["use_cache"] = False

    print(f"family={family} {type(model).__name__} audio tower "
          f"{type(tower).__name__} mel={ac.num_mel_bins}x{frames} "
          f"audio_tokens={n_audio} at id {tok}")

    def first(out):
        return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

    with torch.no_grad():
        cpu_out = first(model(**kw)).clone()

    model_npu = model.to("npu")
    kw_npu = {k: (v.to("npu") if torch.is_tensor(v) else v) for k, v in kw.items()}
    runner = torch.compile(model_npu, dynamic=False) if compile_it else model_npu
    with torch.no_grad():
        npu_out = first(runner(**kw_npu)).cpu()

    test_result(f"Qwen {family} audio", npu_out, cpu_out, rtol=rtol, atol=atol)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="The Qwen families transformers 5.x added, on the Triton route")
    parser.add_argument("--family", type=str, default=None, choices=sorted(FAMILIES),
                        help=f"default: the gate, {' and '.join(_GATE)}")
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--no-compile", dest="compile", action="store_false", default=True)
    parser.add_argument("--audio", action="store_true",
                        help=f"run the audio tower, for {' and '.join(AUDIO)}")
    args = parser.parse_args()

    if args.audio:
        for fam in ([args.family] if args.family else AUDIO):
            if fam not in AUDIO:
                raise SystemExit(f"{fam} has no audio tower")
            run_audio(fam, rtol=args.rtol, atol=args.atol, compile_it=args.compile)
    else:
        for fam in ([args.family] if args.family else _GATE):
            run_family(fam, seq_len=args.seq_len, rtol=args.rtol, atol=args.atol,
                       compile_it=args.compile)
