import os
import sys
import torch
import torch._dynamo
import torch.utils.cpp_extension
sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from _pytorchsim_utils import test_result

def test_reduce_sum(device, size, dim, keepdim=False):
    def reduce_sum(a, b, dim, keepdim):
        return torch.sum(a + b, axis=dim, keepdim=keepdim)
    x = torch.randn(size).to(device=device)
    y = torch.randn(size).to(device=device)
    opt_fn = torch.compile(dynamic=False)(reduce_sum)
    res = opt_fn(x, y, dim, keepdim)
    out = reduce_sum(x.cpu(), y.cpu(), dim, keepdim)
    test_result("ReduceSum", res, out)

def test_reduce_sum2(device, size, dim=-1, keepdim=False):
    def reduce_sum(a, dim, keepdim):
        return torch.sum(a, axis=dim, keepdim=keepdim)
    x = torch.randn(size).to(device=device)
    opt_fn = torch.compile(dynamic=False)(reduce_sum)
    res = opt_fn(x, dim, keepdim)
    out = reduce_sum(x.cpu(), dim, keepdim)
    test_result("ReduceMax", res, out)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run LayerNorm test with dynamic shape")
    parser.add_argument('--shape', type=str, default="(128,768)")
    args = parser.parse_args()
    shape = tuple(map(int, args.shape.strip('()').split(',')))

    device = torch.device("npu:0")
    test_reduce_sum(device, (29, 47), 1, keepdim=True)
    test_reduce_sum(device, (17, 68), 0, keepdim=True)
    test_reduce_sum(device, (327, 447), 1, keepdim=True)
    test_reduce_sum(device, (327, 447), 0, keepdim=True)
    test_reduce_sum2(device, shape)