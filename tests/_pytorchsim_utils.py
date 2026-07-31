"""Shared helpers for PyTorchSim test files.

Module name is unique (not ``tests._utils``) because ``ultralytics``
ships a top-level ``tests`` package in site-packages that would shadow it.

Import with:

    import os, sys
    sys.path.insert(0, os.path.join(
        os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
    from _pytorchsim_utils import test_result
"""

import sys

import torch


def test_result(name, out, expected, rtol=1e-4, atol=1e-4, equal_nan=False):
    """Compare ``out`` to ``expected``; exit 1 on mismatch."""
    out_cpu = out.cpu() if hasattr(out, "cpu") else out
    expected_cpu = expected.cpu() if hasattr(expected, "cpu") else expected

    if torch.allclose(out_cpu, expected_cpu, rtol=rtol, atol=atol, equal_nan=equal_nan):
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
