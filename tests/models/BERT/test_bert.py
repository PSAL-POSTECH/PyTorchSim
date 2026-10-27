"""BERT end to end on the Triton codegen route.

Built from ``BertConfig`` with random weights, so it needs no network and no
checkpoint. What an encoder-only transformer brings that the suite's kernels do
not: THREE embedding gathers summed together (word + position + token_type),
the additive extended attention mask that is broadcast from (B, S) to
(B, 1, 1, S) and added to the scores, softmax over the last axis, exact GELU
(erf, not the tanh approximation GPT-2 uses), and the pooler's tanh on a single
sliced row.

The encoder is BERT base at its published size (google-bert/bert-base-uncased),
cut to 2 of its 12 layers: 768 wide, 12 heads, FFN 3072 and the full 30522-row
vocab, in bfloat16 by default (``--dtype float32`` for the old run). The input is
one pass of ``--seq`` tokens, batch 1, at most 512 positions. ``--part layer``
runs the first encoder layer alone (tests/one_layer.py); an encoder has no decode.

Judgement is spike's, against the same model's float32 copy on the CPU. Run it with the Triton
route on and timing off (rule 13a):

    source .envrc
    python tests/models/BERT/test_bert.py --seq 128
    python tests/models/BERT/test_bert.py --part layer --seq 512
"""

import argparse
import copy
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from result_check import check_result, f32_reference
import one_layer


# google-bert/bert-base-uncased, published config; only num_hidden_layers is cut (12 -> 2).
_BERT = dict(vocab_size=30522, hidden_size=768, num_hidden_layers=2, num_attention_heads=12,
             intermediate_size=3072, hidden_act="gelu", max_position_embeddings=512,
             type_vocab_size=2, layer_norm_eps=1e-12)

# (rtol, atol) per dtype; the bf16 atol is 3x the worst bf16 eager-vs-Inductor CPU diff at S = 128.
_TOL = {"bfloat16": (5e-2, 0.2), "float32": (1e-2, 1e-2)}


def _dtype_from_str(name):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(name, torch.float32)


def _build_config(seq_len, attn_impl):
    """BERT base with dropout at 0, so the graph is free of RNG ops; dropout is identity under eval().

    eager, not sdpa, by default: sdpa would hand the whole attention to one fused kernel and hide the
    seams this test exists to reach.
    """
    from transformers.models.bert.configuration_bert import BertConfig

    if seq_len > _BERT["max_position_embeddings"]:
        raise ValueError(f"seq {seq_len} exceeds BERT's max_position_embeddings "
                         f"{_BERT['max_position_embeddings']}")
    return BertConfig(
        **_BERT,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        use_cache=False,
        attn_implementation=attn_impl,
    )


def _tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    for name in ("last_hidden_state", "logits"):
        if hasattr(output, name) and getattr(output, name) is not None:
            return getattr(output, name)
    if isinstance(output, (list, tuple)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Unsupported output type for comparison: {type(output)}")


@torch.no_grad()
def run_bert(
    device,
    part="body",
    batch=1,
    seq_len=128,
    dtype="bfloat16",
    attn_impl="eager",
    compile_model=True,
    rtol=None,
    atol=None,
):
    """BERT `part` on one pass of `seq_len` tokens, CPU against `device`, at `dtype`'s `_TOL` unless given."""
    from transformers.models.bert.modeling_bert import BertModel, BertForMaskedLM

    torch_dtype = _dtype_from_str(dtype)
    tol = _TOL.get(dtype, _TOL["float32"])
    rtol, atol = tol[0] if rtol is None else rtol, tol[1] if atol is None else atol
    config = _build_config(seq_len, attn_impl)

    # Seed before construction: config-random weights otherwise differ per run,
    # so the worst element wanders across the threshold and the test is flaky.
    torch.manual_seed(0)
    if part == "mlm":
        model_cpu = BertForMaskedLM(config)
    elif part == "layer":
        model_cpu = BertModel(config, add_pooling_layer=False)
    else:
        # add_pooling_layer=False for "body": the pooler reads row 0 only, so it
        # adds a slice-and-tanh kernel without adding encoder coverage. "pooled"
        # is the variant that asks for it.
        model_cpu = BertModel(config, add_pooling_layer=(part == "pooled"))
    model_cpu = model_cpu.to(dtype=torch_dtype).eval()

    print(f"part={part} n_layer={config.num_hidden_layers} "
          f"hidden={config.hidden_size} n_head={config.num_attention_heads} "
          f"vocab={config.vocab_size} seq={seq_len} dtype={dtype} attn={attn_impl}")
    print("model params:", sum(p.numel() for p in model_cpu.parameters()))

    g = torch.Generator().manual_seed(0)
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len), generator=g, dtype=torch.int64)
    # All-ones mask still exercises the extended-mask broadcast and the add;
    # it just does not mask anything out, so CPU and NPU compare on every row.
    attention_mask = torch.ones((batch, seq_len), dtype=torch.int64)
    token_type_ids = torch.zeros((batch, seq_len), dtype=torch.int64)

    if part == "layer":
        cpu_out, npu_out = one_layer.run(model_cpu, model_cpu.encoder.layer[0], config, input_ids, device,
                                         compile_model=compile_model, attention_mask=attention_mask,
                                         token_type_ids=token_type_ids)
    else:
        cpu_out = _tensor(f32_reference(model_cpu)(input_ids=input_ids,
                                                   attention_mask=attention_mask,
                                                   token_type_ids=token_type_ids))

        model_npu = copy.deepcopy(model_cpu).to(device).eval()
        if compile_model:
            model_npu = torch.compile(model_npu, dynamic=False)
        npu_out = _tensor(model_npu(input_ids=input_ids.to(device),
                                    attention_mask=attention_mask.to(device),
                                    token_type_ids=token_type_ids.to(device)))

    check_result(f"BERT {part}", npu_out, cpu_out, rtol=rtol, atol=atol)
    print("Max diff > ", torch.max(torch.abs(npu_out.cpu() - cpu_out)))
    print("BERT Simulation Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERT end to end on the Triton route")
    parser.add_argument("--part", type=str, default="body", choices=["body", "pooled", "mlm", "layer"],
                        help="body = BertModel encoder only; pooled = + the pooler; "
                             "mlm = BertForMaskedLM (adds the vocab projection); "
                             "layer = the first encoder layer alone")
    parser.add_argument("--batch", type=int, default=1)
    one_layer.add_grid_args(parser, decode=False)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--attn-impl", type=str, default="eager", choices=["eager", "sdpa"])
    parser.add_argument("--no-compile", dest="compile", action="store_false", default=True)
    parser.add_argument("--rtol", type=float, default=None, help="default: _TOL of --dtype")
    parser.add_argument("--atol", type=float, default=None, help="default: _TOL of --dtype")
    args = parser.parse_args()

    run_bert(
        torch.device("npu:0"),
        part=args.part,
        batch=args.batch,
        seq_len=args.seq,
        dtype=args.dtype,
        attn_impl=args.attn_impl,
        compile_model=args.compile,
        rtol=args.rtol,
        atol=args.atol,
    )
