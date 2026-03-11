import os
import sys
import torch
from torchvision.models import resnet18 as model1
from test_transformer import EncoderBlock as model2
from Simulator.simulator import TOGSimulator

base_path = os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim')
config = f'{base_path}/configs/systolic_ws_128x128_c2_simple_noc_tpuv3_partition.yml'
os.environ['TOGSIM_CONFIG'] = config

def test_mxfp4_dequantized_gemm(device):
    torch.manual_seed(0)
    input_size = 128
    hidden_size = 128
    output_size = 128

    input = torch.randn(input_size, hidden_size).to(device=device)
    fp16_weight = torch.randn(hidden_size, output_size).to(device=device).to(dtype=torch.float16)
    mx_fp4_weight = torch.quantize_per_tensor(fp16_weight, scale=1.0, zero_point=0, dtype=torch.quint4x2)
    bias = torch.randn(output_size).to(device=device)
    opt_fn = torch.compile(dynamic=False)(custom_matmul)
    out = opt_fn(input, mx_fp4_weight, bias)
    print("MXFP4 Dequantized GEMM Simulation Done")