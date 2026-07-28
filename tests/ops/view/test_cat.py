import argparse
from pathlib import Path

import torch


def _test_result(name, out, cpu_out, rtol=1e-4, atol=1e-4):
    if torch.allclose(out.cpu(), cpu_out, rtol=rtol, atol=atol):
        message = f"|{name} Test Passed|"
        print("-" * len(message))
        print(message)
        print("-" * len(message))
        return

    message = f"|{name} Test Failed|"
    print("-" * len(message))
    print(message)
    print("-" * len(message))
    print("custom out: ", out.cpu())
    print("cpu out: ", cpu_out)
    raise RuntimeError(f"{name} mismatch")

def test_cat_default(device):
    def cat_default_fn(a, b):
        return torch.cat([a, b], dim=0)

    x = torch.randn(8, 16, device=device)
    y = torch.randn(6, 16, device=device)
    opt_fn = torch.compile(dynamic=False)(cat_default_fn)

    out = opt_fn(x, y)

    cpu_out = torch.cat([x.cpu(), y.cpu()], dim=0)
    _test_result("cat.default", out, cpu_out, rtol=1e-4, atol=1e-4)


def test_cat_out(device):
    def cat_out_fn(a, b, out):
        return torch.ops.aten.cat.out([a, b], 0, out=out)

    x = torch.randn(8, 16, device=device)
    y = torch.randn(6, 16, device=device)
    out_buf = torch.empty(14, 16, device=device)
    opt_fn = torch.compile(dynamic=False)(cat_out_fn)

    out = opt_fn(x, y, out_buf)

    cpu_out = torch.cat([x.cpu(), y.cpu()], dim=0)
    _test_result("cat.out", out, cpu_out, rtol=1e-4, atol=1e-4)


def test_cat_4d_dim0(device):
    def cat_4d_dim0_fn(a, b):
        return torch.cat([a, b], dim=0)

    x = torch.randn(2, 3, 4, 5, device=device)
    y = torch.randn(3, 3, 4, 5, device=device)
    opt_fn = torch.compile(dynamic=False)(cat_4d_dim0_fn)

    out = opt_fn(x, y)

    cpu_out = torch.cat([x.cpu(), y.cpu()], dim=0)
    _test_result("cat.4d.dim0", out, cpu_out, rtol=1e-4, atol=1e-4)


def test_cat_4d_dim1(device):
    def cat_4d_dim1_fn(a, b):
        return torch.cat([a, b], dim=1)

    x = torch.randn(2, 3, 4, 5, device=device)
    y = torch.randn(2, 5, 4, 5, device=device)
    opt_fn = torch.compile(dynamic=False)(cat_4d_dim1_fn)

    out = opt_fn(x, y)

    cpu_out = torch.cat([x.cpu(), y.cpu()], dim=1)
    _test_result("cat.4d.dim1", out, cpu_out, rtol=1e-4, atol=1e-4)


def test_cat_4d_dim2(device):
    def cat_4d_dim2_fn(a, b):
        return torch.cat([a, b], dim=2)

    x = torch.randn(2, 3, 4, 5, device=device)
    y = torch.randn(2, 3, 6, 5, device=device)
    opt_fn = torch.compile(dynamic=False)(cat_4d_dim2_fn)

    out = opt_fn(x, y)

    cpu_out = torch.cat([x.cpu(), y.cpu()], dim=2)
    _test_result("cat.4d.dim2", out, cpu_out, rtol=1e-4, atol=1e-4)


def test_cat_4d_dim3(device):
    def cat_4d_dim3_fn(a, b):
        return torch.cat([a, b], dim=3)

    x = torch.randn(2, 3, 4, 5, device=device)
    y = torch.randn(2, 3, 4, 7, device=device)
    opt_fn = torch.compile(dynamic=False)(cat_4d_dim3_fn)

    out = opt_fn(x, y)

    cpu_out = torch.cat([x.cpu(), y.cpu()], dim=3)
    _test_result("cat.4d.dim3", out, cpu_out, rtol=1e-4, atol=1e-4)


def test_cat_three_inputs(device):
    def cat_three_inputs_fn(a, b, c):
        return torch.cat([a, b, c], dim=0)

    x = torch.randn(4, 16, device=device)
    y = torch.randn(5, 16, device=device)
    z = torch.randn(3, 16, device=device)
    opt_fn = torch.compile(dynamic=False)(cat_three_inputs_fn)

    out = opt_fn(x, y, z)

    cpu_out = torch.cat([x.cpu(), y.cpu(), z.cpu()], dim=0)
    _test_result("cat.three_inputs", out, cpu_out, rtol=1e-4, atol=1e-4)


def test_cat_four_inputs(device):
    def cat_four_inputs_fn(a, b, c, d):
        return torch.cat([a, b, c, d], dim=0)

    x = torch.randn(3, 16, device=device)
    y = torch.randn(4, 16, device=device)
    z = torch.randn(5, 16, device=device)
    w = torch.randn(2, 16, device=device)
    opt_fn = torch.compile(dynamic=False)(cat_four_inputs_fn)

    out = opt_fn(x, y, z, w)

    cpu_out = torch.cat([x.cpu(), y.cpu(), z.cpu(), w.cpu()], dim=0)
    _test_result("cat.four_inputs", out, cpu_out, rtol=1e-4, atol=1e-4)


def test_cat_4d_three_inputs(device):
    def cat_4d_three_inputs_fn(a, b, c):
        return torch.cat([a, b, c], dim=1)

    x = torch.randn(2, 3, 4, 5, device=device)
    y = torch.randn(2, 4, 4, 5, device=device)
    z = torch.randn(2, 5, 4, 5, device=device)
    opt_fn = torch.compile(dynamic=False)(cat_4d_three_inputs_fn)

    out = opt_fn(x, y, z)

    cpu_out = torch.cat([x.cpu(), y.cpu(), z.cpu()], dim=1)
    _test_result("cat.4d.three_inputs", out, cpu_out, rtol=1e-4, atol=1e-4)

def test_cat_5d(device, dim=0):
    def cat_5d_fn(a, b):
        return torch.cat([a, b], dim=dim)

    x = torch.randn(2, 3, 4, 5, 6, device=device)
    y = torch.randn(3, 3, 4, 5, 6, device=device)
    opt_fn = torch.compile(dynamic=False)(cat_5d_fn)

    out = opt_fn(x, y)

    cpu_out = torch.cat([x.cpu(), y.cpu()], dim=dim)
    _test_result("cat.5d.dim0", out, cpu_out, rtol=1e-4, atol=1e-4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run cat simulation tests")
    parser.add_argument(
        "--case",
        choices=[
            "default", "out", "4d_dim0", "4d_dim1", "4d_dim2", "4d_dim3", "5d"
            "three_inputs", "four_inputs", "4d_three_inputs", "all"
        ],
        default="all",
        help="Which cat case to run",
    )
    args = parser.parse_args()

    device = torch.device("npu:0")

    if args.case in ("default", "all"):
        test_cat_default(device)
    if args.case in ("out", "all"):
        test_cat_out(device)
    if args.case in ("4d_dim0", "all"):
        test_cat_4d_dim0(device)
    if args.case in ("4d_dim1", "all"):
        test_cat_4d_dim1(device)
    if args.case in ("4d_dim2", "all"):
        test_cat_4d_dim2(device)
    if args.case in ("4d_dim3", "all"):
        test_cat_4d_dim3(device)
    if args.case in ("three_inputs", "all"):
        test_cat_three_inputs(device)
    if args.case in ("four_inputs", "all"):
        test_cat_four_inputs(device)
    if args.case in ("4d_three_inputs", "all"):
        test_cat_4d_three_inputs(device)
    if args.case in ("5d", "all"):
        test_cat_5d(device)
