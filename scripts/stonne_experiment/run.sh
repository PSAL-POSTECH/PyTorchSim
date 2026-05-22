#!/bin/bash
export TORCHSIM_FORCE_TIME_M=1024
export TORCHSIM_FORCE_TIME_K=1024
export TORCHSIM_FORCE_TIME_N=1024
python3 ../../tests/system/test_hetro.py --M 1024 --N 1024 --K 1024 --sparsity 0.9 --config stonne_big_c1_simple_noc.yml --mode 0 > hetero/big_sparse.log
python3 ../../tests/system/test_hetro.py --M 1024 --N 1024 --K 1024 --sparsity 0.9 --config systolic_ws_128x128_c1_simple_noc_tpuv3_half.yml --mode 1 > hetero/big.log
python3 ../../tests/system/test_hetro.py --M 1024 --N 1024 --K 1024 --sparsity 0.9 --config heterogeneous_c2_simple_noc.yml --mode 2 > hetero/hetero.log

echo "All processes completed!"
