import os
import sys
import argparse

base_path = os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim')
sys.path.insert(0, base_path)

import torch
from Simulator.simulator import TOGSimulator

config = os.environ.get('TOGSIM_CONFIG', f'{base_path}/configs/systolic_ws_128x128_c2_simple_noc_tpuv4.yml')
os.environ['TOGSIM_CONFIG'] = config

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--size', nargs='+', type=int, default=[512, 512], help='Tensor Shape')
    args = args.parse_args()
    size = tuple(args.size)
    dim = 1

    device = torch.device("npu:0")
    model = torch.nn.Softmax(dim=dim).to(device=device)
    opt_fn = torch.compile(dynamic=False)(model)
    model_input = torch.randn(*size).to(device=device)

    with TOGSimulator(config_path=config), torch.no_grad():
        torch.npu.launch_model(opt_fn, model_input, stream_index=0, timestamp=0)
        torch.npu.synchronize()
    print(f"Softmax {size} Simulation Done")
