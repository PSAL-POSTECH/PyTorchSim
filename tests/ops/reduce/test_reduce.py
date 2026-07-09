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

def test_reduce_gather_bias(device, NW=4, H=3, Q=32, K=32, T=64):
    """A reduction fused with an INDIRECT gather bias, as in SwinV2 cosine-window
    attention: score[w,h,q,k] + table[idx[q,k], h] -> amax over the key axis. The gather
    blocks the head*query dim-merge, so the reduction tile stays 4-D. The reduction axis
    must be hoisted to the outermost in-lane position; the 4-D reduction tile path used to
    skip that reorder and reduce a batch axis (head) instead of the key axis, so head 0's
    max picked up head 1's values (needs H>=2 and NW>=2 to expose the head bleed)."""
    def fn(score, idx, table):
        bias = table[idx.reshape(-1)].reshape(Q, K, H).permute(2, 0, 1).unsqueeze(0)
        return (score + bias).amax(dim=-1)
    torch.manual_seed(0)
    score = torch.randn(NW, H, Q, K).to(device=device)
    idx = torch.randint(0, T, (Q, K)).to(device=device)
    table = torch.randn(T, H).to(device=device)
    res = torch.compile(dynamic=False)(fn)(score, idx, table)
    out = fn(score.cpu(), idx.cpu(), table.cpu())
    test_result("ReduceGatherBias", res, out)

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
    test_reduce_gather_bias(device)