import os
import sys
import torch
sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from _pytorchsim_utils import test_result

def test_addmm_residual(device, input_size=128, hidden_size=128, output_size=128):
    def addmm_residual(a, b, c, d):
        return torch.addmm(a, b, c) + d
    torch.manual_seed(0)
    input = torch.randn(input_size, hidden_size)
    weight = torch.randn(hidden_size, output_size)
    bias = torch.randn(input_size, output_size)
    residual = torch.randn(input_size, output_size)
    x1 = input.to(device=device)
    w1 = weight.to(device=device)
    b1 = bias.to(device=device)
    r1 = residual.to(device=device)
    x2 = input.to("cpu")
    w2 = weight.to("cpu")
    b2 = bias.to("cpu")
    r2 = residual.to("cpu")
    opt_fn = torch.compile(dynamic=False)(addmm_residual)
    res = opt_fn(b1, x1, w1, r1)
    y = addmm_residual(b2, x2, w2, r2)
    test_result("Addmm + Residual Fusion Forward", res, y)

if __name__ == "__main__":
    device = torch.device("npu:0")
    test_addmm_residual(device, 32, 32, 32)
    test_addmm_residual(device, 128, 128, 128)
    test_addmm_residual(device, 512, 512, 512)
    test_addmm_residual(device, 129, 61, 56)
