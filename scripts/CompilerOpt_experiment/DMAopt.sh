#!/bin/bash
export TOGSIM_CONFIG="/root/workspace/PyTorchSim/configs/systolic_ws_128x128_c2_simple_noc_tpuv2.yml"

# None FG DMA
export TORCHSIM_SUBTILE=0
python experiments/gemm.py --size 128 128 128
python experiments/gemm.py --size 256 256 256
python experiments/gemm.py --size 512 512 512
python experiments/gemm.py --size 1024 1024 1024
python experiments/gemm.py --size 2048 2048 2048

# FG DMA
export TORCHSIM_SUBTILE=1
export TORCHSIM_MANUAL_SUBTILE_SIZE=1
python experiments/gemm.py --size 128 128 128
python experiments/gemm.py --size 256 256 256
python experiments/gemm.py --size 512 512 512
python experiments/gemm.py --size 1024 1024 1024
python experiments/gemm.py --size 2048 2048 2048

# SFG DMA
export TORCHSIM_SUBTILE=1
export TORCHSIM_MANUAL_SUBTILE_SIZE=0
python experiments/gemm.py --size 128 128 128
python experiments/gemm.py --size 256 256 256
python experiments/gemm.py --size 512 512 512
python experiments/gemm.py --size 1024 1024 1024
python experiments/gemm.py --size 2048 2048 2048