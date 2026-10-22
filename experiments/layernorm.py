import os
import sys
import argparse

base_path = os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim')
sys.path.insert(0, base_path)

import torch

os.environ.setdefault("TORCHSIM_TARGET", "tpuv6e")
os.environ.setdefault("TORCHSIM_TIMING_MODE", "1")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--size', nargs='+', type=int, default=[512, 768], help='Tensor Shape')
    args = args.parse_args()
    size = tuple(args.size)
    normalized_shape = size[-1]

    device = torch.device("npu:0")
    model = torch.nn.LayerNorm(normalized_shape).to(device=device)
    opt_fn = torch.compile(dynamic=False)(model)
    model_input = torch.randn(*size).to(device=device)

    with torch.npu.TOGSimulator(), torch.no_grad():
        torch.npu.launch_model(opt_fn, model_input, stream_index=0, timestamp=0)
        torch.npu.synchronize()
    print(f"LayerNorm {size} Simulation Done")
