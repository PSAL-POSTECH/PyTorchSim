import os
import sys
import math
import argparse

base_path = os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim')
sys.path.insert(0, base_path)

import torch
from Simulator.simulator import TOGSimulator

config = os.environ.get('TOGSIM_CONFIG', f'{base_path}/configs/systolic_ws_128x128_c1_simple_noc_tpuv3.yml')
os.environ['TOGSIM_CONFIG'] = config

def attention(query, key, value):
    d_k = query.size(-1)
    scores = torch.matmul(key, query.transpose(-2, -1)) / math.sqrt(d_k)
    p_attn = scores.softmax(dim=-2)
    return torch.matmul(value.transpose(-1, -2), p_attn)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--size', nargs='+', type=int, default=[12, 512, 64], help='Tensor Shape')
    args = args.parse_args()
    size = tuple(args.size)

    device = torch.device("npu:0")
    query = torch.randn(*size).to(device=device)
    key = torch.randn(*size).to(device=device)
    value = torch.randn(*size).to(device=device)
    opt_fn = torch.compile(dynamic=False)(attention)

    with TOGSimulator(config_path=config), torch.no_grad():
        torch.npu.launch_model(opt_fn, query, key, value, stream_index=0, timestamp=0)
        torch.npu.synchronize()
    print(f"Attention {size} Simulation Done")
