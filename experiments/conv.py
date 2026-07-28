import os
import sys
import argparse

base_path = os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim')
sys.path.insert(0, base_path)

import torch
from Simulator.simulator import TOGSimulator

config = os.environ.get('TOGSIM_CONFIG', f'{base_path}/configs/systolic_ws_128x128_c2_simple_noc_tpuv4.yml')
os.environ['TOGSIM_CONFIG'] = config

def conv2d_fn(batch_size, i_h, i_w, i_c, o_c, kernel_size, stride, padding):
    def _conv(a, b, bias):
        conv2d = torch.nn.Conv2d(i_c, o_c, kernel_size, stride=stride, padding=padding, dilation=1, bias=False)
        conv2d.weight = torch.nn.Parameter(b)
        return conv2d(a)
    return _conv

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--size', nargs='+', type=int, default=[8, 28, 28, 128, 128, 3, 1, 1],
                      help='B H W I_C O_C K S P')
    args = args.parse_args()
    batch_size, i_h, i_w, i_c, o_c, kernel_size, stride, padding = args.size

    device = torch.device("npu:0")
    conv_input = torch.randn(batch_size, i_c, i_h, i_w).to(memory_format=torch.channels_last, device=device)
    conv_kernel = torch.randn(o_c, i_c, kernel_size, kernel_size).to(memory_format=torch.channels_last, device=device)
    conv_bias = torch.randn(o_c).to(device=device)

    custom_conv = conv2d_fn(batch_size, i_h, i_w, i_c, o_c, kernel_size, stride, padding)
    opt_fn = torch.compile(dynamic=False)(custom_conv)

    with TOGSimulator(config_path=config), torch.no_grad():
        torch.npu.launch_model(opt_fn, conv_input, conv_kernel, conv_bias, stream_index=0, timestamp=0)
        torch.npu.synchronize()
    print(f"CONV {batch_size}_{i_h}_{i_w}_{i_c}_{o_c}_{kernel_size}_{stride}_{padding} Simulation Done")
