import os
import sys
import argparse

base_path = os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim')
sys.path.insert(0, base_path)

import torch
from torchvision.models import resnet18
from Simulator.simulator import TOGSimulator

config = os.environ.get('TOGSIM_CONFIG', f'{base_path}/configs/systolic_ws_128x128_c1_simple_noc_tpuv3.yml')
os.environ['TOGSIM_CONFIG'] = config

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--batch', type=int, default=1)
    args = args.parse_args()

    device = torch.device("npu:0")
    model = resnet18().eval().to(device=device, memory_format=torch.channels_last)
    opt_fn = torch.compile(dynamic=False)(model)
    model_input = torch.randn(args.batch, 3, 224, 224).to(device=device)

    with TOGSimulator(config_path=config), torch.no_grad():
        torch.npu.launch_model(opt_fn, model_input, stream_index=0, timestamp=0)
        torch.npu.synchronize()
    print("ResNet18 Simulation Done")
