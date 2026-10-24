import os
import sys
import torch
sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from result_check import check_result


def test_row_matmul(device, hidden_size=256, output_size=384):
    """A 1 x K row times a K x N matrix, which the frontend runs as vector work."""
    def row_matmul(a, b):
        return torch.matmul(a, b)
    torch.manual_seed(0)
    row = torch.randn(1, hidden_size)
    weight = torch.randn(hidden_size, output_size)
    opt_fn = torch.compile(dynamic=False)(row_matmul)
    res = opt_fn(row.to(device=device), weight.to(device=device))
    check_result("Row Matmul Forward", res, row_matmul(row, weight))


def test_row_addmm(device, hidden_size=256, output_size=384):
    """The same with a bias added, as a Linear layer on one row gives."""
    def row_addmm(bias, a, b):
        return torch.addmm(bias, a, b)
    torch.manual_seed(0)
    row = torch.randn(1, hidden_size)
    weight = torch.randn(hidden_size, output_size)
    bias = torch.randn(output_size)
    opt_fn = torch.compile(dynamic=False)(row_addmm)
    res = opt_fn(bias.to(device=device), row.to(device=device), weight.to(device=device))
    check_result("Row Addmm Forward", res, row_addmm(bias, row, weight))


def test_row_matmul_bf16(device, hidden_size=256, output_size=384):
    """The row matmul in bf16: its kernel widens with vfwcvtbf16 (zvfbfmin), which traps as an
    illegal instruction under a Spike ISA other than the one the kernel was compiled for."""
    def row_matmul(a, b):
        return torch.matmul(a, b)
    torch.manual_seed(0)
    row = torch.randn(1, hidden_size, dtype=torch.bfloat16)
    weight = torch.randn(hidden_size, output_size, dtype=torch.bfloat16)
    opt_fn = torch.compile(dynamic=False)(row_matmul)
    res = opt_fn(row.to(device=device), weight.to(device=device))
    want = row_matmul(row.float(), weight.float()).to(torch.bfloat16)
    check_result("Row Matmul bf16 Forward", res.float(), want.float(), rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    device = torch.device("npu:0")
    test_row_matmul(device, 256, 384)
    test_row_matmul(device, 61, 56)
    test_row_addmm(device, 256, 384)
    test_row_matmul_bf16(device, 256, 384)
