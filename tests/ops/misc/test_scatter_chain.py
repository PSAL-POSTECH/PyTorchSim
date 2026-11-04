"""A chain of view scatters, the delta rule's `attn[..., i, :i] = ...` loop, is written in place on the npu:
only the first write, fused with attn's producer, covers all of attn; each later one writes its own row."""

import os
import re
import sys

import torch
from torch._inductor.utils import run_and_get_code

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from result_check import check_result

B, L, STEPS = 4, 16, 8


def _delta_rule(x):
    attn = x * 2
    for i in range(1, STEPS):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    return attn


def _full_size_kernels(src):
    """The kernel calls in a wrapper whose last argument, the element count, is all of attn."""
    calls = re.findall(r"pytorchsim_triton_compiler\w+\(([^()]*)\)", src)
    return [c for c in calls if c.split(",")[-1].strip() == str(B * L * L)]


def test_scatter_chain_in_place(device):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(B, L, L, generator=g) * 0.1
    cpu_out = _delta_rule(x)
    torch._dynamo.reset()
    npu_out, code = run_and_get_code(torch.compile(_delta_rule, dynamic=False), x.to(device))
    check_result("delta-rule scatter chain", npu_out, cpu_out)
    full = _full_size_kernels(code[0])
    assert len(full) == 1, f"{len(full)} kernels over all of attn, want 1 (its producer): {full}"


if __name__ == "__main__":
    device = torch.device("npu:0")
    test_scatter_chain_in_place(device)
