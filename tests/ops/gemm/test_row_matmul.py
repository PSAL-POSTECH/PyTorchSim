import os
import sys
import torch
sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from _pytorchsim_utils import test_result


def test_row_matmul(device, hidden_size=256, output_size=384):
    """A 1 x K row times a K x N matrix, which the frontend runs as vector work."""
    def row_matmul(a, b):
        return torch.matmul(a, b)
    torch.manual_seed(0)
    row = torch.randn(1, hidden_size)
    weight = torch.randn(hidden_size, output_size)
    opt_fn = torch.compile(dynamic=False)(row_matmul)
    res = opt_fn(row.to(device=device), weight.to(device=device))
    test_result("Row Matmul Forward", res, row_matmul(row, weight))


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
    test_result("Row Addmm Forward", res, row_addmm(bias, row, weight))


if __name__ == "__main__":
    device = torch.device("npu:0")
    test_row_matmul(device, 256, 384)
    test_row_matmul(device, 61, 56)
    test_row_addmm(device, 256, 384)
