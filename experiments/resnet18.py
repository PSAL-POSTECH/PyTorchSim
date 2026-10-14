import os
import sys
import argparse

base_path = os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim')
sys.path.insert(0, base_path)

import torch
from torchvision.models import resnet18

os.environ.setdefault("TORCHSIM_TARGET", "tpuv6e")
os.environ.setdefault("TORCHSIM_TIMING_MODE", "1")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--batch', type=int, default=1)
    args = args.parse_args()

    device = torch.device("npu:0")
    model = resnet18().eval().to(device=device, memory_format=torch.channels_last)
    opt_fn = torch.compile(dynamic=False)(model)
    model_input = torch.randn(args.batch, 3, 224, 224).to(device=device)

    with torch.npu.simulator(), torch.no_grad():
        torch.npu.launch_model(opt_fn, model_input, stream_index=0, timestamp=0)
        torch.npu.synchronize()
    print("ResNet18 Simulation Done")
