import torch
import torch._dynamo
import torch.utils.cpp_extension
import random
import numpy as np
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.environ.get('TORCHSIM_DIR', default='/root/workspace/PyTorchSim'), 'tests'))
from _pytorchsim_utils import test_result

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def apply_pruning(tensor, sparsity):
    mask = torch.rand_like(tensor) >= sparsity
    tensor *= mask

def sparse_matmul(a, b):
    return torch.sparse.mm(a, b)

def test_sparse_mm(device, input_size=128, hidden_size=128, output_size=128, sparsity=0.0):
    input = torch.randn(input_size, hidden_size)
    weight = torch.randn(hidden_size, output_size)
    apply_pruning(input, sparsity)
    apply_pruning(weight, sparsity)
    x1 = input.to(device=device)
    w1 = weight.to(device=device)
    opt_fn = torch.compile(dynamic=False)(sparse_matmul)
    res = opt_fn(x1, w1)
    cpu_res = sparse_matmul(input.cpu(), weight.cpu())
    #test_result("spmm", res, cpu_res)
 
 
if __name__ == "__main__":
    import os
    import sys
    parser = argparse.ArgumentParser(description="stonne test")
    parser.add_argument("sz", nargs="?", type=int, help="size", default=64)
    parser.add_argument("sparsity", nargs="?", type=float, help="%% of zero", default=0.0)

    args = parser.parse_args()
 
    device = torch.device("npu:0")
    test_sparse_mm(device, args.sz, args.sz, args.sz, args.sparsity)