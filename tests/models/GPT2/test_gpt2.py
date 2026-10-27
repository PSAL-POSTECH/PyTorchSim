"""GPT-2 end to end on the Triton codegen route.

Built from ``GPT2Config`` with random weights, so it needs no network and no
checkpoint. The point is the seams a decoder-only transformer hits that the
suite's kernels do not: two embedding gathers (``wte``/``wpe``), Conv1D's
``addmm`` with the fused QKV projection, the causal ``where`` mask, softmax,
and ``gelu_new``'s tanh.

The block is GPT-2 base at its published size (openai-community/gpt2), cut to
2 of its 12 layers: 768 wide, 12 heads, the 4x MLP and the full 50257-row vocab,
in bfloat16 by default (``--dtype float32`` for the old run). The input is one
prefill of ``--seq`` tokens, batch 1, at most n_positions 1024. ``--part layer``
runs the first block alone (tests/one_layer.py), on ``--seq S`` or on one
``--decode S`` step after a KV cache.

Judgement is spike's, against the same model on CPU. Run it with the Triton
route on and timing off (rule 13a):

    source .envrc
    python tests/models/GPT2/test_gpt2.py --seq 128
    python tests/models/GPT2/test_gpt2.py --part layer --decode 1024
"""

import argparse
import copy
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from result_check import check_result
import one_layer


# openai-community/gpt2, published config; only n_layer is cut (12 -> 2).
_GPT2 = dict(vocab_size=50257, n_positions=1024, n_embd=768, n_layer=2, n_head=12, n_inner=None,
             activation_function="gelu_new", layer_norm_epsilon=1e-5)

# (rtol, atol) per dtype; the bf16 atol is 3x the worst bf16 eager-vs-Inductor CPU diff at S = 128.
_TOL = {"bfloat16": (5e-2, 0.05), "float32": (1e-2, 1e-2)}


def _dtype_from_str(name):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(name, torch.float32)


def _build_config(seq_len):
    """GPT-2 base with dropout at 0, so the graph is free of RNG ops; dropout is identity under eval()."""
    from transformers.models.gpt2.configuration_gpt2 import GPT2Config

    if seq_len > _GPT2["n_positions"]:
        raise ValueError(f"seq {seq_len} exceeds GPT-2's n_positions {_GPT2['n_positions']}")
    return GPT2Config(
        **_GPT2,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=False,
        attn_implementation="eager",
    )


def _logits(output):
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, (list, tuple)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Unsupported output type for comparison: {type(output)}")


@torch.no_grad()
def run_gpt2(
    device,
    part="lm",
    batch=1,
    seq_len=128,
    dtype="bfloat16",
    compile_model=True,
    rtol=None,
    atol=None,
    decode=False,
):
    """GPT-2 `part` on a prefill of `seq_len` tokens, or with `decode` (--part layer) on one step at
    position seq_len - 1 after a KV cache of the tokens before it; CPU against `device`, at `_TOL`."""
    from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel, GPT2Model

    torch_dtype = _dtype_from_str(dtype)
    tol = _TOL.get(dtype, _TOL["float32"])
    rtol, atol = tol[0] if rtol is None else rtol, tol[1] if atol is None else atol
    config = _build_config(seq_len)

    # Seed before construction: config-random weights otherwise differ per run,
    # so the worst element wanders across the threshold and the test is flaky.
    torch.manual_seed(0)
    cls = GPT2LMHeadModel if part == "lm" else GPT2Model
    model_cpu = cls(config).to(dtype=torch_dtype).eval()

    print(f"part={part} n_layer={config.n_layer} n_embd={config.n_embd} "
          f"n_head={config.n_head} vocab={config.vocab_size} seq={seq_len} dtype={dtype}")
    print("model params:", sum(p.numel() for p in model_cpu.parameters()))

    g = torch.Generator().manual_seed(0)
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len), generator=g, dtype=torch.int64)

    if part == "layer":
        cpu_out, npu_out = one_layer.run(model_cpu, model_cpu.h[0], config, input_ids, device,
                                         decode=decode, compile_model=compile_model)
    else:
        cpu_out = _logits(model_cpu(input_ids))

        model_npu = copy.deepcopy(model_cpu).to(device).eval()
        if compile_model:
            model_npu = torch.compile(model_npu, dynamic=False)
        npu_out = _logits(model_npu(input_ids.to(device)))

    check_result(f"GPT-2 {part}", npu_out, cpu_out, rtol=rtol, atol=atol)
    print("Max diff > ", torch.max(torch.abs(npu_out.cpu() - cpu_out)))
    print("GPT-2 Simulation Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-2 end to end on the Triton route")
    parser.add_argument("--part", type=str, default="lm", choices=["lm", "body", "layer"],
                        help="lm = GPT2LMHeadModel (adds the vocab projection); body = GPT2Model; "
                             "layer = the first block alone")
    parser.add_argument("--batch", type=int, default=1)
    one_layer.add_grid_args(parser)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--no-compile", dest="compile", action="store_false", default=True)
    parser.add_argument("--rtol", type=float, default=None, help="default: _TOL of --dtype")
    parser.add_argument("--atol", type=float, default=None, help="default: _TOL of --dtype")
    args = parser.parse_args()
    if args.decode is not None and args.part != "layer":
        parser.error("--decode runs with --part layer")

    run_gpt2(
        torch.device("npu:0"),
        part=args.part,
        batch=args.batch,
        seq_len=args.decode or args.seq,
        decode=args.decode is not None,
        dtype=args.dtype,
        compile_model=args.compile,
        rtol=args.rtol,
        atol=args.atol,
    )
