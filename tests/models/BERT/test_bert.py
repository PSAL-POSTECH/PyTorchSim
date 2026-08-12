"""BERT end to end on the Triton codegen route.

Built from ``BertConfig`` with random weights, so it needs no network and no
checkpoint. What an encoder-only transformer brings that the suite's kernels do
not: THREE embedding gathers summed together (word + position + token_type),
the additive extended attention mask that is broadcast from (B, S) to
(B, 1, 1, S) and added to the scores, softmax over the last axis, exact GELU
(erf, not the tanh approximation GPT-2 uses), and the pooler's tanh on a single
sliced row.

Judgement is spike's, against the same model on CPU. Run it with the Triton
route on and timing off (rule 13a):

    source .envrc
    python tests/models/BERT/test_bert.py --preset small
"""

import argparse
import copy
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from _pytorchsim_utils import test_result


# Presets shrink the model, not the shapes' character: the head dim, the 4x
# intermediate ratio and the vocab gather all survive. "small" is the one to
# reach for -- 2 layers of the real 768-wide block is every kernel BERT has.
_PRESETS = {
    #          n_layer  hidden  n_head  intermediate  vocab  seq
    "tiny":   (1,       128,    2,      512,          256,   16),
    "small":  (2,       768,    12,     3072,         1024,  32),
    "medium": (4,       768,    12,     3072,         4096,  32),
    "full":   (12,      768,    12,     3072,         30522, 128),
}


def _dtype_from_str(name):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(name, torch.float32)


def _build_config(preset, seq_len, attn_impl):
    from transformers.models.bert.configuration_bert import BertConfig

    n_layer, hidden, n_head, intermediate, vocab, preset_seq = _PRESETS[preset]
    seq_len = seq_len if seq_len is not None else preset_seq

    return BertConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        num_hidden_layers=n_layer,
        num_attention_heads=n_head,
        intermediate_size=intermediate,
        max_position_embeddings=max(seq_len, 64),
        # Dropout is identity under eval(), but leaving it at 0 keeps the graph
        # free of the RNG ops so a failure is about the model, not about seeds.
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        use_cache=False,
        # eager, not sdpa: sdpa would hand the whole attention to one fused
        # kernel and hide the seams this test exists to reach.
        attn_implementation=attn_impl,
    ), seq_len


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
    preset="small",
    part="body",
    batch=1,
    seq_len=None,
    dtype="float32",
    attn_impl="eager",
    compile_model=True,
    rtol=1e-2,
    atol=1e-2,
):
    from transformers.models.bert.modeling_bert import BertModel, BertForMaskedLM

    torch_dtype = _dtype_from_str(dtype)
    config, seq_len = _build_config(preset, seq_len, attn_impl)

    # Seed before construction: config-random weights otherwise differ per run,
    # so the worst element wanders across the threshold and the test is flaky.
    torch.manual_seed(0)
    if part == "mlm":
        model_cpu = BertForMaskedLM(config)
    else:
        # add_pooling_layer=False for "body": the pooler reads row 0 only, so it
        # adds a slice-and-tanh kernel without adding encoder coverage. "pooled"
        # is the variant that asks for it.
        model_cpu = BertModel(config, add_pooling_layer=(part == "pooled"))
    model_cpu = model_cpu.to(dtype=torch_dtype).eval()

    print(f"preset={preset} part={part} n_layer={config.num_hidden_layers} "
          f"hidden={config.hidden_size} n_head={config.num_attention_heads} "
          f"vocab={config.vocab_size} seq={seq_len} dtype={dtype} attn={attn_impl}")
    print("model params:", sum(p.numel() for p in model_cpu.parameters()))

    g = torch.Generator().manual_seed(0)
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len), generator=g, dtype=torch.int64)
    # All-ones mask still exercises the extended-mask broadcast and the add;
    # it just does not mask anything out, so CPU and NPU compare on every row.
    attention_mask = torch.ones((batch, seq_len), dtype=torch.int64)
    token_type_ids = torch.zeros((batch, seq_len), dtype=torch.int64)

    cpu_out = _tensor(model_cpu(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids))

    model_npu = copy.deepcopy(model_cpu).to(device).eval()
    if compile_model:
        model_npu = torch.compile(model_npu, dynamic=False)
    npu_out = _tensor(model_npu(input_ids=input_ids.to(device),
                                attention_mask=attention_mask.to(device),
                                token_type_ids=token_type_ids.to(device)))

    test_result(f"BERT {part} ({preset})", npu_out, cpu_out, rtol=rtol, atol=atol)
    print("Max diff > ", torch.max(torch.abs(npu_out.cpu() - cpu_out)))
    print("BERT Simulation Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERT end to end on the Triton route")
    parser.add_argument("--preset", type=str, default="small", choices=sorted(_PRESETS))
    parser.add_argument("--part", type=str, default="body", choices=["body", "pooled", "mlm"],
                        help="body = BertModel encoder only; pooled = + the pooler; "
                             "mlm = BertForMaskedLM (adds the vocab projection)")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--attn-impl", type=str, default="eager", choices=["eager", "sdpa"])
    parser.add_argument("--no-compile", dest="compile", action="store_false", default=True)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    args = parser.parse_args()

    run_bert(
        torch.device("npu:0"),
        preset=args.preset,
        part=args.part,
        batch=args.batch,
        seq_len=args.seq_len,
        dtype=args.dtype,
        attn_impl=args.attn_impl,
        compile_model=args.compile,
        rtol=args.rtol,
        atol=args.atol,
    )
