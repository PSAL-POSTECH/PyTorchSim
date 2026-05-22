import os
import sys
import torch
import argparse

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))

from Simulator.simulator import TOGSimulator
from system.test_stonne import sparse_matmul


def custom_matmul(a, b):
    return torch.matmul(a, b)


torch.manual_seed(0)
CONFIG_TORCHSIM_DIR = os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--M", type=int, default=128, help="Batch size")
    parser.add_argument("--N", type=int, default=128, help="Input layer size")
    parser.add_argument("--K", type=int, default=128, help="Hidden layer size")
    parser.add_argument("--sparsity", type=float, default=0.9, help="Sparsity")
    parser.add_argument("--config", type=str, default="stonne_big_c1_simple_noc.yml", help="TOGSim config file name under configs/")
    parser.add_argument("--mode", type=int, default=0, help="0=spmm only, 1=dense matmul only, 2=both partitions")
    args = parser.parse_args()

    M, N, K = args.M, args.N, args.K
    sparsity = args.sparsity
    mode = args.mode
    config_path = f"{CONFIG_TORCHSIM_DIR}/configs/{args.config}"

    print("M: ", M)
    print("N: ", N)
    print("K: ", K)
    print("sparsity: ", sparsity)

    device = torch.device("npu:0")

    opt_model1 = torch.compile(custom_matmul)
    opt_model2 = torch.compile(sparse_matmul)

    dense_input1 = torch.randn(M, K, device=device)
    dense_input2 = torch.randn(K, N, device=device)

    sparse_input1 = torch.randn(128, 128, device=device)
    sparse_input2 = torch.randn(128, 128, device=device)
    mask1 = torch.rand(sparse_input1.shape, device=device) > sparsity
    mask2 = torch.rand(sparse_input2.shape, device=device) > sparsity
    sparse_input1 = sparse_input1 * mask1
    sparse_input2 = sparse_input2 * mask2

    with torch.no_grad():
        with TOGSimulator(config_path=config_path):
            if mode == 0:
                torch.npu.launch_model(opt_model2, sparse_input1, sparse_input2, stream_index=0, timestamp=0)
            elif mode == 1:
                torch.npu.launch_model(opt_model1, dense_input1, dense_input2, stream_index=0, timestamp=0)
            elif mode == 2:
                torch.npu.launch_model(opt_model2, sparse_input1, sparse_input2, stream_index=0, timestamp=0)
                torch.npu.launch_model(opt_model1, dense_input1, dense_input2, stream_index=1, timestamp=0)
            else:
                raise ValueError(f"unknown mode {mode}")
            torch.npu.synchronize()
