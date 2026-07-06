import os
import sys
import torch
sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from _pytorchsim_utils import test_result

def test_maxpool(device, b=1, c=64, h=112, w=112):
    torch.manual_seed(0)
    model = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1).eval()
    model.to(device=device)
    input = torch.randn(b, c, h, w).to(device=device)
    x1 = input.to(device=device)
    x2 = input.to("cpu")
    opt_fn = torch.compile(dynamic=False)(model)
    res = opt_fn(x1)
    model.to("cpu")
    out = model(x2)
    test_result("Maxpool Forward", res, out) # TODO: MaxPool Functionality is not working

def test_avgpool(device, b=1, c=64, h=112, w=112):
    def avgpool(a):
        return torch.nn.AdaptiveAvgPool2d((1, 1))(a)
    torch.manual_seed(0)
    input = torch.randn(b, c, h, w).to(device=device) #FIXME: channel 8 does not work (range padding issue)
    x1 = input.to(device=device)
    x2 = input.to("cpu")
    opt_fn = torch.compile(dynamic=False)(avgpool)
    res = opt_fn(x1)
    out = avgpool(x2)
    test_result("Avgpool Forward", res, out)

if __name__ == "__main__":
    device = torch.device("npu:0")
    #test_maxpool(device, b=1, c=8, h=16, w=16)
    #test_maxpool(device, b=1, c=8, h=112, w=112)
    test_avgpool(device, b=1, c=512, h=7, w=7)
