import os
import sys
import torch
import torch._dynamo
import torch.utils.cpp_extension
sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from _pytorchsim_utils import test_result

def test_matmul_scalar(device):
    def matmul_fused(a, b, c):
        return torch.matmul(a, b) * c
    torch.manual_seed(0)
    input = torch.randn(128, 128)
    weight = torch.randn(128, 128)
    bias = torch.randn(128)
    x1 = input.to(device=device)
    w1 = weight.to(device=device)
    x2 = input.to("cpu")
    w2 = weight.to("cpu")
    c = 7
    opt_fn = torch.compile(dynamic=False)(matmul_fused)
    res = opt_fn(x1, w1, c)
    y = matmul_fused(x2, w2, c)
    test_result("Matmul Scalar Fusion Forward", res, y)

if __name__ == "__main__":
    device = torch.device("npu:0")
    test_matmul_scalar(device)
