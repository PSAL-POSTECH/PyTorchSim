"""The result check PyTorchSim test files share: check_result compares an npu result to its CPU
reference and prints the line the CI sweep reads; check_routed_rows is its form for a MoE layer's output.
A bf16 device run is checked against f32_reference, the same bf16 weights and inputs computed in float32.

Module name is unique (not ``tests._utils``) because ``ultralytics``
ships a top-level ``tests`` package in site-packages that would shadow it.

Import with:

    import os, sys
    sys.path.insert(0, os.path.join(
        os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
    from result_check import check_result
"""

import copy
import sys

import torch


def f32_reference(module):
    """A float32 copy of `module` holding its bf16 (or fp16) weights exactly: a bf16 run's CPU reference, in
    float32 because a bf16 or fp16 matmul runs some 300x slower than float32 on the CPU."""
    return copy.deepcopy(module).float()


def f32(x):
    """`x` with every floating tensor in it, through tuples, lists and dicts, as float32."""
    if torch.is_tensor(x):
        return x.float() if x.is_floating_point() else x
    if isinstance(x, (tuple, list)):
        return type(x)(f32(v) for v in x)
    if isinstance(x, dict):
        return {k: f32(v) for k, v in x.items()}
    return x


def check_result(name, out, expected, rtol=1e-4, atol=1e-4):
    """Compare ``out`` to ``expected``, in float32 when their dtypes differ; exit 1 on mismatch."""
    out_cpu = out.cpu() if hasattr(out, "cpu") else out
    expected_cpu = expected.cpu() if hasattr(expected, "cpu") else expected
    if out_cpu.dtype != expected_cpu.dtype:
        out_cpu, expected_cpu = out_cpu.float(), expected_cpu.float()

    if torch.allclose(out_cpu, expected_cpu, rtol=rtol, atol=atol):
        msg = f"|{name} Test Passed|"
        bar = "-" * len(msg)
        print(bar)
        print(msg)
        print(bar)
        return

    msg = f"|{name} Test Failed|"
    bar = "-" * len(msg)
    print(bar)
    print(msg)
    print(bar)
    print("custom out: ", out_cpu)
    print("cpu out:    ", expected_cpu)
    try:
        max_diff = (out_cpu - expected_cpu).abs().max().item()
        print(f"Max abs diff: {max_diff}")
    except Exception:
        pass
    sys.exit(1)


def check_routed_rows(name, out, expected, rtol, atol, row_share):
    """check_result over the token rows (last dim) of `out`, of which up to `row_share` may differ: in bf16 a
    router near-tie flips for a few tokens, which then take other experts. More than that fails them all."""
    out_rows = f32(out.cpu() if hasattr(out, "cpu") else out).reshape(-1, out.shape[-1])
    exp_rows = f32(expected.cpu() if hasattr(expected, "cpu") else expected).reshape(-1, expected.shape[-1])
    close = torch.isclose(out_rows, exp_rows, rtol=rtol, atol=atol).all(dim=-1)
    apart, allowed = int((~close).sum()), int(row_share * close.numel())
    print(f"{name}: {apart} of {close.numel()} token rows routed apart, {allowed} allowed")
    if apart > allowed:
        check_result(name, out_rows, exp_rows, rtol=rtol, atol=atol)
    check_result(name, out_rows[close], exp_rows[close], rtol=rtol, atol=atol)
