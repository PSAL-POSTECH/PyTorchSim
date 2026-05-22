import os
import sys
import torch
import torch._dynamo
import torch.utils.cpp_extension
sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from _pytorchsim_utils import test_result

def test_vectoradd(device, size=(128, 128)):
    def vectoradd(a, b):
        return a + b
    x = torch.randn(size).to(device=device)
    y = torch.randn(size).to(device=device)
    opt_fn = torch.compile(dynamic=False)(vectoradd)
    res = opt_fn(x, y)
    out = vectoradd(x.cpu(), y.cpu())
    test_result("VectorAdd", res, out)

def test_vector_scalar_add(device, size=(128, 128)):
    def vectoradd(a, b):
        return a + b
    x = torch.randn(size).to(device=device)
    y = torch.randn([1]).to(device=device)
    opt_fn = torch.compile(dynamic=False)(vectoradd)
    res = opt_fn(x, y)
    out = vectoradd(x.cpu(), y.cpu())
    test_result("VectorScalarAdd", res, out)

def test_vector_tensor_add(device, size=(128, 128)):
    def vectoradd(a, b):
        return a + b
    x = torch.randn(size).to(device=device)
    y = torch.randn(size[-1]).to(device=device)
    opt_fn = torch.compile(dynamic=False)(vectoradd)
    res = opt_fn(x, y)
    out = vectoradd(x.cpu(), y.cpu())
    test_result("VectorTensorAdd", res, out)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run LayerNorm test with dynamic shape")
    parser.add_argument('--shape', type=str, default="(512,768)")
    args = parser.parse_args()
    shape = tuple(map(int, args.shape.strip('()').split(',')))

    device = torch.device("npu:0")
    test_vectoradd(device, (1, 1))
    test_vectoradd(device, (47, 10))
    test_vectoradd(device, (128, 128))
    test_vectoradd(device, (4071, 429))
    test_vector_tensor_add(device, (128, 128))
