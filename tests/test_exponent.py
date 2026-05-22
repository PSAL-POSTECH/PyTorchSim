import os
import sys
import torch
sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from _pytorchsim_utils import test_result

def test_exponent(device, size=(128, 128)):
    def exponent(a):
        return a.exp()
    x = torch.randn(size).to(device=device)
    opt_fn = torch.compile(dynamic=False)(exponent)
    res = opt_fn(x)
    out = exponent(x.cpu())
    test_result("exponent", res, out)

if __name__ == "__main__":
    device = torch.device("npu:0")
    test_exponent(device, size=(32, 32))
