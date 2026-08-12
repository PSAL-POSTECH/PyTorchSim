"""GPT-2 end to end on the Triton codegen route.

Built from ``GPT2Config`` with random weights, so it needs no network and no
checkpoint. The point is the seams a decoder-only transformer hits that the
suite's kernels do not: two embedding gathers (``wte``/``wpe``), Conv1D's
``addmm`` with the fused QKV projection, the causal ``where`` mask, softmax,
and ``gelu_new``'s tanh.

Judgement is spike's, against the same model on CPU. Run it with the Triton
route on and timing off (rule 13a):

    source .envrc
    python tests/models/GPT2/test_gpt2.py --preset small
"""

import argparse
import copy
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from _pytorchsim_utils import test_result


# Presets shrink the model, not the shapes' character: the head dim, the 4x MLP
# ratio and the vocab gather all survive. "small" is the one to reach for --
# 2 layers of the real 768-wide block is already every kernel GPT-2 has.
_PRESETS = {
    #          n_layer  n_embd  n_head  vocab  seq
    "tiny":   (1,       128,    2,      256,   16),
    "small":  (2,       768,    12,     1024,  32),
    "medium": (4,       768,    12,     4096,  32),
    "full":   (12,      768,    12,     50257, 32),
}


def _dtype_from_str(name):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(name, torch.float32)


def _build_config(preset, seq_len, dtype):
    from transformers.models.gpt2.configuration_gpt2 import GPT2Config

    n_layer, n_embd, n_head, vocab, preset_seq = _PRESETS[preset]
    seq_len = seq_len if seq_len is not None else preset_seq

    return GPT2Config(
        vocab_size=vocab,
        n_positions=max(seq_len, 64),
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        # Dropout is identity under eval(), but leaving it at 0 keeps the graph
        # free of the RNG ops so a failure is about the model, not about seeds.
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=False,
        attn_implementation="eager",
    ), seq_len


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
    preset="small",
    part="lm",
    batch=1,
    seq_len=None,
    dtype="float32",
    compile_model=True,
    rtol=1e-2,
    atol=1e-2,
):
    from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel, GPT2Model

    torch_dtype = _dtype_from_str(dtype)
    config, seq_len = _build_config(preset, seq_len, torch_dtype)

    # Seed before construction: config-random weights otherwise differ per run,
    # so the worst element wanders across the threshold and the test is flaky.
    torch.manual_seed(0)
    cls = GPT2LMHeadModel if part == "lm" else GPT2Model
    model_cpu = cls(config).to(dtype=torch_dtype).eval()

    print(f"preset={preset} part={part} n_layer={config.n_layer} n_embd={config.n_embd} "
          f"n_head={config.n_head} vocab={config.vocab_size} seq={seq_len} dtype={dtype}")
    print("model params:", sum(p.numel() for p in model_cpu.parameters()))

    g = torch.Generator().manual_seed(0)
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len), generator=g, dtype=torch.int64)

    cpu_out = _logits(model_cpu(input_ids))

    model_npu = copy.deepcopy(model_cpu).to(device).eval()
    if compile_model:
        model_npu = torch.compile(model_npu, dynamic=False)
    npu_out = _logits(model_npu(input_ids.to(device)))

    test_result(f"GPT-2 {part} ({preset})", npu_out, cpu_out, rtol=rtol, atol=atol)
    print("Max diff > ", torch.max(torch.abs(npu_out.cpu() - cpu_out)))
    print("GPT-2 Simulation Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-2 end to end on the Triton route")
    # THE DEFAULT IS THE GATE: scripts/ci/triton_route_sweep.py runs this file
    # with no arguments. It is `small` -- 2 layers of the real 768-wide block,
    # 12 heads -- because that is what passes: 26 kernels, max diff 2.0862e-06,
    # and zero divergent buffers over 219 goldens with per-kernel verify on.
    #
    # It was `tiny` until the wide-tile gather was fixed (triton-npu
    # develop-select-grid 2007619, pinned by
    # kernels/coverage/gather/gather_masked_deep_in_loop.py): a gathered MVIN
    # used to fill only lane-count elements per lane and repeat them, which
    # took out this model's very first kernel at any width above the lanes.
    parser.add_argument("--preset", type=str, default="small", choices=sorted(_PRESETS))
    parser.add_argument("--part", type=str, default="lm", choices=["lm", "body"],
                        help="lm = GPT2LMHeadModel (adds the vocab projection); body = GPT2Model")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--no-compile", dest="compile", action="store_false", default=True)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    args = parser.parse_args()

    run_gpt2(
        torch.device("npu:0"),
        preset=args.preset,
        part=args.part,
        batch=args.batch,
        seq_len=args.seq_len,
        dtype=args.dtype,
        compile_model=args.compile,
        rtol=args.rtol,
        atol=args.atol,
    )
