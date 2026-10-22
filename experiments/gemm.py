import os
import sys
import argparse

base_path = os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim')
sys.path.insert(0, base_path)

import torch

os.environ.setdefault("TORCHSIM_TARGET", "tpuv6e")
os.environ.setdefault("TORCHSIM_TIMING_MODE", "1")

def matmul_fn(a, b):
    return torch.matmul(a, b)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--size', nargs='+', type=int, default=[128, 128, 128], help='M K N')
    args = args.parse_args()
    M, K, N = args.size[0], args.size[1], args.size[2]

    device = torch.device("npu:0")
    torch.manual_seed(0)
    input_a = torch.randn(M, K).to(device=device)
    input_b = torch.randn(K, N).to(device=device)
    opt_fn = torch.compile(dynamic=False)(matmul_fn)

    with torch.npu.TOGSimulator(), torch.no_grad():
        torch.npu.launch_model(opt_fn, input_a, input_b, stream_index=0, timestamp=0)
        torch.npu.synchronize()
    print(f"GEMM {M}x{K}x{N} (MxKxN) Simulation Done")
