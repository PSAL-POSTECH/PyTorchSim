import os
import sys
import torch
import torch._dynamo
import torch.utils.cpp_extension
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from _pytorchsim_utils import test_result

def test_ReLU(device, size=(128, 128)):
    torch.manual_seed(0)
    input = torch.randn(size)
    x1 = input.to(device=device)
    x2 = input.to("cpu")
    ReLU = torch.nn.ReLU()
    opt_fn = torch.compile(dynamic=False)(ReLU)
    y = opt_fn(x1)
    cpu_y = ReLU(x2)
    test_result("ReLU", y, cpu_y)

def test_GeLU(device, size=(128, 128), approximate='none'):
    torch.manual_seed(0)
    input = torch.randn(size)
    x1 = input.to(device=device)
    x2 = input.to("cpu")
    GeLU = torch.nn.GELU(approximate=approximate)
    opt_fn = torch.compile(dynamic=False)(GeLU)
    y = opt_fn(x1)
    cpu_y = GeLU(x2)
    test_result("GeLU", y, cpu_y)

def test_sigmoid(device, size=(128, 128)):
    torch.manual_seed(0)
    input = torch.randn(size)
    x1 = input.to(device=device)
    x2 = input.to("cpu")
    Sigmoid = torch.nn.Sigmoid()
    opt_fn = torch.compile(dynamic=False)(Sigmoid)
    y = opt_fn(x1)
    cpu_y = Sigmoid(x2)
    test_result("Sigmoid", y, cpu_y)

def test_SiLU(device, size=(128, 128)):
    torch.manual_seed(0)
    input = torch.randn(size)
    x1 = input.to(device=device)
    x2 = input.to("cpu")
    SiLU = torch.nn.SiLU()
    opt_fn = torch.compile(dynamic=False)(SiLU)
    y = opt_fn(x1)
    cpu_y = SiLU(x2)
    test_result("SiLU", y, cpu_y)

class SwiGLU(torch.nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

def test_SwiGLU(device, size=(128, 128)):
    torch.manual_seed(0)
    input = torch.randn(size)
    x1 = input.to(device=device)
    x2 = input.to("cpu")
    SwiGLU_fn = SwiGLU()
    opt_fn = torch.compile(dynamic=False)(SwiGLU_fn)
    y = opt_fn(x1)
    cpu_y = SwiGLU_fn(x2)
    test_result("SwiGLU", y, cpu_y)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run LayerNorm test with dynamic shape")
    parser.add_argument('--shape', type=str, default="(512,768)")
    args = parser.parse_args()
    shape = tuple(map(int, args.shape.strip('()').split(',')))

    device = torch.device("npu:0")
    test_ReLU(device, (47, 10))
    test_ReLU(device, (128, 128))
    test_ReLU(device, (4071, 429))
    test_sigmoid(device, (128, 128))
    test_SiLU(device, (128, 128))
    test_SwiGLU(device, (128, 128))
    test_GeLU(device, (128, 128))
    test_GeLU(device, (128, 128), approximate='tanh')
