import os
import sys
import argparse

base_path = os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim')
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, 'tests'))

import torch
from Simulator.simulator import TOGSimulator

config = os.environ.get('TOGSIM_CONFIG', f'{base_path}/configs/systolic_ws_128x128_c1_simple_noc_tpuv3_timing_only.yml')
os.environ['TOGSIM_CONFIG'] = config

# Try Fusion EncoderBlock first, fall back to standard test_transformer
try:
    from ops.fusion.test_transformer_fusion import EncoderBlock
except ImportError:
    from models.test_transformer import EncoderBlock

HIDDEN_DIM = {'base': 768, 'large': 1024, 'xlarge': 2048}
EMBEDDING_SIZE = {'base': 768, 'large': 1024, 'xlarge': 2048}
HEADS = {'base': 12, 'large': 16, 'xlarge': 32}

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--size', type=str, default='base', choices=['base', 'large', 'xlarge'])
    args.add_argument('--input_size', type=int, default=512)
    args = args.parse_args()

    hidden_dim = HIDDEN_DIM[args.size]
    embedding_size = EMBEDDING_SIZE[args.size]
    heads = HEADS[args.size]

    device = torch.device("npu:0")
    model = EncoderBlock(embedding_size, heads).eval().to(device=device)
    model_input = torch.randn(args.input_size, hidden_dim).to(device=device)
    opt_fn = torch.compile(dynamic=False)(model)

    with TOGSimulator(config_path=config), torch.no_grad():
        torch.npu.launch_model(opt_fn, model_input, stream_index=0, timestamp=0)
        torch.npu.synchronize()
    print(f"BERT-{args.size} Simulation Done")
