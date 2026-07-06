import os
import sys
import torch
sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from _pytorchsim_utils import test_result

def test_Transpose3D_1(device, size=(4, 16, 32)):
    def transpose(a, b):
        return a.transpose(1, 2) + b
    torch.manual_seed(0)
    # x = torch.randn(16, 32).to(device=device)
    x = torch.randn(size[0], size[2], size[1]).float().to(device=device)
    y = torch.randn(size[0], size[1], size[2]).float().to(device=device)

    opt_fn = torch.compile(dynamic=False)(transpose)
    res = opt_fn(x, y)
    out = transpose(x.cpu(), y.cpu())
    test_result("Transpose 3D Forward", res, out)

def test_Transpose3D_2(device, size=(4, 16, 32)):
    def transpose(a, b):
        return a.transpose(0, 2) + b
    torch.manual_seed(0)
    # x = torch.randn(16, 32).to(device=device)
    x = torch.randn(size[2], size[1], size[0]).float().to(device=device)
    y = torch.randn(size[0], size[1], size[2]).float().to(device=device)

    opt_fn = torch.compile(dynamic=False)(transpose)
    res = opt_fn(x, y)
    out = transpose(x.cpu(), y.cpu())
    test_result("Transpose 3D Forward", res, out)

def test_Transpose3D_3(device, size=(4, 16, 32)):
    def transpose(a, b):
        return a.transpose(0, 1) + b
    torch.manual_seed(0)
    # x = torch.randn(16, 32).to(device=device)
    x = torch.randn(size[1], size[0], size[2]).float().to(device=device)
    y = torch.randn(size[0], size[1], size[2]).float().to(device=device)

    opt_fn = torch.compile(dynamic=False)(transpose)
    res = opt_fn(x, y)
    out = transpose(x.cpu(), y.cpu())
    test_result("Transpose 3D Forward", res, out)

if __name__ == "__main__":
    device = torch.device("npu:0")
    test_Transpose3D_1(device, [62, 34, 44])
    test_Transpose3D_1(device, [62, 134, 144])
    test_Transpose3D_2(device, [62, 34, 44])
    test_Transpose3D_2(device, [62, 134, 144])
    test_Transpose3D_3(device, [62, 34, 44])
    test_Transpose3D_3(device, [62, 134, 144])

