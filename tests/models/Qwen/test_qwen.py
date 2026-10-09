"""Qwen 1.5 / 2 / 2.5 / 3 and the two MoE lines, on the Triton codegen route.

Built from the ``transformers`` configs with random weights, so it needs no
network and no checkpoint.  One file covers every Qwen version because the
versions are shapes and small deltas on one block, not separate models -- and
a Qwen model is that block repeated, so a preset runs one layer.  What depth
would add is the block-to-block seam, which the Llama and GPT-2 tests already
cross.

WHAT EACH VERSION ADDS, and why it needs its own preset:

  * 1.5      QKV bias.  Every q/k/v projection in the Qwen2 module carries a
             bias term (``bias=True`` is hardcoded in ``Qwen2Attention``), which
             no Llama, GPT-2-style-fused or BERT path in this suite emits: it is
             an addmm epilogue on the three projections that feed attention.
             1.5-1.8B is also MHA (n_rep = 1) and ties lm_head.
  * 2        GQA with n_rep = 6 at 1.5B.
  * 2.5      n_rep = 7 at the real 7B width -- 3584 hidden, head_dim 128,
             SwiGLU 18944.  Seven is the first ODD PRIME repeat in the suite
             (Llama 3.2-3B contributed 3).  2.5 is otherwise config-identical
             to 2: same ``Qwen2ForCausalLM``, so the preset is a width claim.
  * 2-moe    Qwen1.5-MoE-A2.7B's block: routed experts PLUS a shared expert
             whose contribution is scaled by a sigmoid gate, which DeepSeek-V3's
             MoE (shared experts, no gate on them) and Mixtral (no shared
             expert) both miss.  ``norm_topk_prob`` is False here, so the
             routing weights are NOT renormalised after top-k.
  * 3        q_norm / k_norm: an RMSNorm applied per head over head_dim, after
             the projection and before rope.  A reduction of length head_dim
             sitting inside attention, which nothing else in the suite has.
             Bias is gone again (attention_bias=False).
  * 3-0.6b   head_dim DECOUPLED from hidden_size: 16 heads x 128 = 2048 against
             1024 hidden, so q_proj is not square and the o_proj shrinks.  Every
             other attention in this suite has heads * head_dim == hidden.
  * 3-moe    Qwen3's MoE: routed experts only, no shared expert, and
             ``norm_topk_prob`` True.  The counterpart to 2-moe -- the pair is
             what makes the shared-expert path a tested difference rather than
             an assumed one.

WHAT IS ABSENT, and what was measured about each rather than assumed:

  * Qwen 1 (``QWenLMHeadModel``, 2023) ships only as remote code on the Hub.
    Remote code IS supported here -- the DeepSeek-V3 test drives that path,
    import shim and all -- so the cost is the network fetch in CI plus einops
    and transformers_stream_generator, which nothing else needs.  It is also
    NOT free of new lowering, which an earlier version of this note claimed:
    against the Qwen2 block it brings aten::_safe_softmax, split, unbind and
    tril.  It does not reach the backend at all today.  Its rotary cache is
    built inside the remote modeling file and is not a registered buffer, so
    ``.to(device)`` leaves it on the host and dynamo stops at "failed to run
    FX node with fake tensors: mul(npu:0 (1,8,8,32), cpu (1,8,1,32))".
  * Qwen2-VL and Qwen2.5-VL build and run on CPU from config alone (no
    checkpoint), and compile 16 kernels here before stopping in triton_shared:
    the ``masked_scatter`` that splices image embeddings into the text
    sequence lowers to a ``tts.get_structured_state`` whose operand type that
    pass rejects.  Getting that far needed ``triton_helpers.select_one``
    vendored into kernel_spec (M-RoPE emits it); that is in this branch.
  * Qwen2-Audio stops in tnpu's p13_bank_vectorize, on the check that counts
    audio tokens against audio features: one operand is ONE_LANE and the other
    is banked across 128 lanes.
  * Qwen2.5-Coder / Math, QwQ and Qwen3-Embedding are the Qwen2 and Qwen3
    classes at other shapes, so a preset here already describes them.

``_assert_character`` cashes each claim above against the built model.  Without
it a preset can pass while proving nothing -- "qwen3 runs" is worth nothing if
q_norm silently did not exist.

    source /workspace/torchsim-env.sh
    python tests/models/Qwen/test_qwen.py --preset 3-8b

ALL EIGHT PRESETS PASS, measured 2026-08-13 against the same model on CPU, one
cleared cache each.  triton-npu pinned to develop 3434608, the pin the Llama 3.x
table was measured against.

    preset      family      hidden  heads/kv  head_dim  interm  params       kern  max diff    time
    tiny        qwen2          256    8/2        32      1024     1,082,496   18   5.9605e-07   86s
    1.5-1.8b    qwen2         2048   16/16      128      5504    58,994,688   15   4.3064e-06  138s
    2-1.5b      qwen2         1536   12/2       128      8960    53,090,816   19   3.6359e-06  186s
    2.5-7b      qwen2         3584   28/4       128     18944   291,781,632   19   7.6294e-06  377s
    2-moe       qwen2_moe     1024   16/16       64      2816    34,356,224   36   2.4214e-06  228s
    3-0.6b      qwen3         1024   16/8       128      3072    19,926,272   20   2.9802e-06  156s
    3-8b        qwen3         4096   32/8       128     12288   260,059,392   19   6.6757e-06  402s
    3-moe       qwen3_moe     2048   32/4       128       768*   65,034,496   37   3.5167e-06  227s

    * moe_intermediate_size; both MoE presets run 8 experts, top-4.

THE WALL CLOCKS ARE UPPER BOUNDS, not a measurement of this backend: four other
model runs (Gemma, Kimi, DeepSeek) shared the machine throughout, so these are
not comparable with the Llama 3.x table's.  The kernel counts and the diffs are.

Two things the table says that the presets alone do not:

  * MOE COSTS KERNELS, NOT WIDTH.  36 and 37 kernels against 19 for a dense
    block of the same shape, because eight experts lower to eight separate
    MLPs -- the routing picks between them, it does not batch them.  That is
    the shape of the cost to expect when the real 60 and 128 expert counts are
    tried, and it is why the expert count is the axis that got scaled down.
  * WIDTH DOES NOT ADD KERNELS.  19 at 1536 hidden and 19 at 4096, the same
    observation test_llama3x.py made across a 4x width.

WHAT THIS FOUND: one defect, in this repo rather than in Qwen.  Qwen3's q_norm
fuses with rope into a PERSISTENT reduction over head_dim, and 27 live
scratchpad tiles put it 1625120 bytes/lane over a 131072 budget.  The scratchpad
retry in `PyTorchSimCodegen/codecache.py` answered that by halving R0_BLOCK, which
a persistent reduction writes into its own body and never reads from the launch
-- so 128, 8 and 1 all measured the identical overflow and the kernel was
refused.  `_shrink_tile` now moves only a block the kernel actually takes as an
argument, and falls back to XBLOCK when a reduction takes none; 3-moe compiles
on the first retry.  Nothing about Qwen was wrong.
"""

import argparse
import copy
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from _pytorchsim_utils import test_result


# Presets shrink cost, not character.  Width carries head_dim, the GQA repeat
# and the SwiGLU ratio, so width stays honest wherever it is affordable; depth
# and vocab are the cheap axes and are scaled down everywhere.  Expert COUNT is
# a cost axis too -- what distinguishes the two MoE lines is the shared expert
# and the renormalisation, not how many experts there are -- so both MoE presets
# run 8 experts instead of 60 (1.5-MoE) and 128 (3-30B-A3B).
_PRESETS = {
    # Smoke.  Small enough to tell a broken test file from a broken backend.
    "tiny": dict(family="qwen2", layers=1, hidden=256, heads=8, kv_heads=2,
                 intermediate=1024, vocab=256, seq=16),

    # Qwen1.5-1.8B: MHA (n_rep = 1), tied lm_head, SwiGLU 5504.
    "1.5-1.8b": dict(family="qwen2", layers=1, hidden=2048, heads=16, kv_heads=16,
                     intermediate=5504, vocab=4096, seq=32, tie=True),

    # Qwen2-1.5B: n_rep = 6, head_dim 128, tied lm_head.
    "2-1.5b": dict(family="qwen2", layers=1, hidden=1536, heads=12, kv_heads=2,
                   intermediate=8960, vocab=4096, seq=32, tie=True),

    # Qwen2.5-7B at its real width: n_rep = 7, head_dim 128, SwiGLU 18944.
    "2.5-7b": dict(family="qwen2", layers=1, hidden=3584, heads=28, kv_heads=4,
                   intermediate=18944, vocab=8192, seq=32),

    # Qwen1.5-MoE-A2.7B: shared expert + sigmoid gate, norm_topk_prob False.
    "2-moe": dict(family="qwen2_moe", layers=1, hidden=1024, heads=16, kv_heads=16,
                  intermediate=2816, vocab=2048, seq=32,
                  moe_intermediate=704, shared_expert_intermediate=2816,
                  experts=8, top_k=4, norm_topk=False),

    # Qwen3-0.6B: head_dim 128 against 1024 hidden -- decoupled -- and tied.
    "3-0.6b": dict(family="qwen3", layers=1, hidden=1024, heads=16, kv_heads=8,
                   head_dim=128, intermediate=3072, vocab=4096, seq=32, tie=True),

    # Qwen3-8B at its real width: 4096 hidden, head_dim 128, SwiGLU 12288.
    "3-8b": dict(family="qwen3", layers=1, hidden=4096, heads=32, kv_heads=8,
                 head_dim=128, intermediate=12288, vocab=8192, seq=32),

    # Qwen3-30B-A3B: routed experts only, norm_topk_prob True, head_dim 128
    # decoupled from 2048 hidden (32 x 128 = 4096).
    "3-moe": dict(family="qwen3_moe", layers=1, hidden=2048, heads=32, kv_heads=4,
                  head_dim=128, intermediate=6144, vocab=2048, seq=32,
                  moe_intermediate=768, experts=8, top_k=4, norm_topk=True),
}

_FAMILIES = {
    "qwen2": ("transformers.models.qwen2.configuration_qwen2", "Qwen2Config",
              "transformers.models.qwen2.modeling_qwen2", "Qwen2Model", "Qwen2ForCausalLM"),
    "qwen2_moe": ("transformers.models.qwen2_moe.configuration_qwen2_moe", "Qwen2MoeConfig",
                  "transformers.models.qwen2_moe.modeling_qwen2_moe", "Qwen2MoeModel", "Qwen2MoeForCausalLM"),
    "qwen3": ("transformers.models.qwen3.configuration_qwen3", "Qwen3Config",
              "transformers.models.qwen3.modeling_qwen3", "Qwen3Model", "Qwen3ForCausalLM"),
    "qwen3_moe": ("transformers.models.qwen3_moe.configuration_qwen3_moe", "Qwen3MoeConfig",
                  "transformers.models.qwen3_moe.modeling_qwen3_moe", "Qwen3MoeModel", "Qwen3MoeForCausalLM"),
}


def _dtype_from_str(name):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(name, torch.float32)


def _import(module, name):
    import importlib

    return getattr(importlib.import_module(module), name)


def _build_config(preset, seq_len):
    p = _PRESETS[preset]
    cfg_mod, cfg_name, _, _, _ = _FAMILIES[p["family"]]
    Config = _import(cfg_mod, cfg_name)
    seq_len = seq_len if seq_len is not None else p["seq"]

    kwargs = dict(
        vocab_size=p["vocab"],
        hidden_size=p["hidden"],
        num_attention_heads=p["heads"],
        num_key_value_heads=p["kv_heads"],
        intermediate_size=p["intermediate"],
        num_hidden_layers=p["layers"],
        max_position_embeddings=32768,
        rope_theta=1000000.0,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        tie_word_embeddings=p.get("tie", False),
        use_cache=False,
        # Sliding window is off on every shipped Qwen 2.x/3 config below the
        # long-context variants, and at seq=32 it would be inert anyway -- an
        # untested flag is better than a flag that silently changes nothing.
        # sliding_window is cleared as well as the switch: transformers warns
        # about an eager attention with a window set even when the switch is
        # off, and a warning that does not describe the run is noise.
        use_sliding_window=False,
        sliding_window=None,
        # Eager keeps the graph free of the sdpa dispatch, so a failure is about
        # the model rather than about which attention kernel transformers picked.
        attn_implementation="eager",
    )
    if "head_dim" in p:
        kwargs["head_dim"] = p["head_dim"]
    if p["family"] in ("qwen2_moe", "qwen3_moe"):
        kwargs.update(
            moe_intermediate_size=p["moe_intermediate"],
            num_experts=p["experts"],
            num_experts_per_tok=p["top_k"],
            norm_topk_prob=p["norm_topk"],
            decoder_sparse_step=1,
        )
        # 3-moe has no shared expert at all; 2-moe's is the point of the preset.
        if p["family"] == "qwen2_moe":
            kwargs["shared_expert_intermediate_size"] = p["shared_expert_intermediate"]
            kwargs["mlp_only_layers"] = []
        else:
            kwargs["mlp_only_layers"] = []

    return Config(**kwargs), seq_len


def _layer0(model):
    body = model.model if hasattr(model, "model") else model
    return body.layers[0]



def _routing_of(block):
    """(top_k, norm_topk_prob), wherever this transformers keeps them.

    Up to 4.x the sparse block holds both itself. From 5.x the routing is a
    module of its own -- `Qwen2MoeTopKRouter`, `Qwen3MoeTopKRouter` -- and the
    block delegates: `block.gate.top_k`. The QUESTION is the same either way
    and it is the one the preset makes a claim about, so it is asked in one
    place rather than reached for through whichever attribute this version
    happens to have.
    """
    for owner in (block, getattr(block, "gate", None)):
        if owner is not None and hasattr(owner, "top_k"):
            return owner.top_k, getattr(owner, "norm_topk_prob", None)
    return None, None


def _expert_count(block):
    """How many experts this MoE block has, in either spelling.

    A ModuleList of per-expert MLPs up to transformers 4.x; ONE fused module
    from 5.x, holding `gate_up_proj` and `down_proj` as [experts, ...] tensors
    with no `__len__`. The count is the same claim either way, and it is the
    claim the preset makes -- but the SHAPE OF THE GRAPH IS NOT: eight experts
    lower to eight MLPs in the first spelling and to one batched matmul in the
    second, which is why the kernel counts in the docstring are stated against
    the version they were measured on.
    """
    experts = block.experts
    if hasattr(experts, "__len__"):
        return len(experts)
    return getattr(experts, "num_experts", None)


def _assert_character(preset, cfg, model):
    """Fail if the model did not actually get the feature the preset claims.

    A preset is a claim about shape and about which Qwen the block is; this is
    where the claim is cashed.  Every check guards against a run that passes
    without exercising anything the other presets do not already cover.
    """
    p = _PRESETS[preset]
    layer = _layer0(model)
    attn = layer.self_attn
    n_rep = cfg.num_attention_heads // cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads

    if p["family"] in ("qwen2", "qwen2_moe"):
        # The QKV bias is what makes a Qwen2-family block not a Llama block.
        for name in ("q_proj", "k_proj", "v_proj"):
            assert getattr(attn, name).bias is not None, \
                f"Qwen2-family {name} must carry a bias; this preset is a Llama block otherwise"
        assert attn.o_proj.bias is None, "Qwen2's o_proj is unbiased"
        assert not hasattr(attn, "q_norm"), "q_norm belongs to Qwen3, not to a Qwen2-family block"
    else:
        assert hasattr(attn, "q_norm") and hasattr(attn, "k_norm"), \
            "Qwen3 must apply q_norm/k_norm; without them this is a Qwen2 block with no bias"
        assert attn.q_norm.weight.numel() == head_dim, \
            f"q_norm normalises over head_dim ({head_dim}), got {attn.q_norm.weight.numel()}"
        assert attn.q_proj.bias is None, "Qwen3 drops the QKV bias (attention_bias=False)"

    if preset == "1.5-1.8b":
        assert n_rep == 1, f"1.5-1.8B is MHA; got n_rep={n_rep}"
    if preset == "2-1.5b":
        assert n_rep == 6, f"2-1.5B must exercise n_rep=6, got {n_rep}"
    if preset == "2.5-7b":
        assert cfg.hidden_size == 3584 and cfg.intermediate_size == 18944, \
            "2.5-7b preset must run at the real 7B width"
        assert n_rep == 7, f"2.5-7b must exercise the odd-prime GQA repeat, got n_rep={n_rep}"
        assert head_dim == 128, f"2.5-7b must run at the real head_dim, got {head_dim}"
    if preset == "3-8b":
        assert cfg.hidden_size == 4096 and head_dim == 128 and cfg.intermediate_size == 12288, \
            "3-8b preset must run at the real Qwen3-8B width"

    if preset in ("3-0.6b", "3-moe"):
        assert cfg.num_attention_heads * head_dim != cfg.hidden_size, \
            ("this preset exists to decouple head_dim from hidden_size; "
             f"heads*head_dim == hidden ({cfg.hidden_size}) means it proves nothing new")

    if p["family"] == "qwen2_moe":
        block = layer.mlp
        assert hasattr(block, "shared_expert") and hasattr(block, "shared_expert_gate"), \
            "the 2-moe preset exists for the gated shared expert; this block has none"
        top_k, norm = _routing_of(block)
        assert _expert_count(block) == p["experts"] and top_k == p["top_k"]
        assert norm is False, "Qwen1.5-MoE does not renormalise the top-k weights"
    if p["family"] == "qwen3_moe":
        block = layer.mlp
        assert not hasattr(block, "shared_expert"), \
            "Qwen3-MoE has no shared expert; if this block has one it is a Qwen2-MoE block"
        top_k, norm = _routing_of(block)
        assert _expert_count(block) == p["experts"] and top_k == p["top_k"]
        assert norm is True, "Qwen3-MoE renormalises the top-k weights"

    if p.get("tie", False):
        body = model.model if hasattr(model, "model") else model
        head = getattr(model, "lm_head", None)
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
def run_qwen(
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

    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    n_rep = cfg.num_attention_heads // cfg.num_key_value_heads
    print(f"preset={preset} family={p['family']} part={part} layers={cfg.num_hidden_layers} "
          f"hidden={cfg.hidden_size} heads={cfg.num_attention_heads}/{cfg.num_key_value_heads} "
          f"head_dim={head_dim} n_rep={n_rep} interm={cfg.intermediate_size} "
          f"vocab={cfg.vocab_size} seq={seq_len} dtype={dtype}")
    if p["family"].endswith("moe"):
        print(f"  moe: experts={cfg.num_experts} top_k={cfg.num_experts_per_tok} "
              f"moe_interm={cfg.moe_intermediate_size} norm_topk={cfg.norm_topk_prob} "
              f"shared={getattr(cfg, 'shared_expert_intermediate_size', None)}")
    print("model params:", sum(x.numel() for x in model_cpu.parameters()))

    _assert_character(preset, cfg, model_cpu)

    g = torch.Generator().manual_seed(0)
    input_ids = torch.randint(0, cfg.vocab_size, (batch, seq_len), generator=g, dtype=torch.int64)

    # The explicit 4D causal mask is test_llama3x.py's, kept so a failure here is
    # comparable to that file's rather than to whatever _update_causal_mask
    # decided to build.
    min_dtype = torch.finfo(torch_dtype).min
    causal_mask = torch.full((seq_len, seq_len), fill_value=min_dtype, dtype=torch_dtype)
    if seq_len > 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)
    attn_mask = causal_mask[None, None, :, :].expand(batch, 1, -1, -1)

    cpu_out = _out(model_cpu(input_ids=input_ids, attention_mask=attn_mask))

    model_npu = copy.deepcopy(model_cpu).to(device).eval()
    if compile_model:
        model_npu = torch.compile(model_npu, dynamic=False)
    npu_out = _out(model_npu(input_ids=input_ids.to(device),
                             attention_mask=attn_mask.to(device)))

    test_result(f"Qwen {part} ({preset})", npu_out, cpu_out, rtol=rtol, atol=atol)
    print("Max diff > ", torch.max(torch.abs(npu_out.cpu() - cpu_out)))
    print("Qwen Simulation Done")


# THE GATE: what the sweep runs when it runs this file with no arguments.
#
# TWO PRESETS, NOT ONE, because one Qwen preset cannot carry the file's claim.
# Every other model here has a single architecture and gates a width; this file
# gates a family whose members differ from each other, so the gate has to
# straddle the difference:
#
#   3-8b    Qwen3 at its real 8B width -- q_norm/k_norm, head_dim 128, no bias.
#   2-moe   the Qwen2-family QKV BIAS (Qwen2MoeAttention hardcodes it, exactly
#           as Qwen2Attention does) AND the routed + shared-expert MoE block.
#
# Between them: both attention variants, both bias conventions, the real Qwen3
# width, and the MoE routing. 519s measured for this no-argument run, against a
# 1800s sweep timeout and an allowlist that already carries an 11-minute entry.
#
# What the gate therefore does NOT hold, and a command holds instead:
#   --preset 2.5-7b   n_rep=7 at the real 7B width
#   --preset 3-moe    Qwen3-MoE's no-shared-expert block
#   --preset 3-0.6b   head_dim decoupled from hidden_size
# Adding all five would put the entry past 20 minutes for coverage that moves
# with the same block the gate already runs.
_GATE = ("3-8b", "2-moe")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen 1.5/2/2.5/3 end to end on the Triton route")
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

    sys.path.append(os.environ.get("PYTORCHSIM_ROOT_PATH", "/workspace/PyTorchSim"))
    torch.compiler.is_compiling = lambda: True  # FIXME. Same as test_llama.py.

    for preset in ([args.preset] if args.preset else _GATE):
        run_qwen(
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
