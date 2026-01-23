export TORCHSIM_LOG_PATH=$(pwd)/result
export SPIKE_DUMP_SPARSE_TILE=1
export TORCHSIM_FORCE_TIME_K=8
export TORCHSIM_FORCE_TIME_M=8
export TORCHSIM_FORCE_TIME_N=8

OUTPUT_DIR="12GB"
export TORCHSIM_CONFIG="/workspace/PyTorchSim/configs/systolic_ws_8x8_c1_12G_simple_noc.yml"
python3 /workspace/PyTorchSim/tests/test_sparsity.py --sparsity  0.0  > ${OUTPUT_DIR}/0.0
python3 /workspace/PyTorchSim/tests/test_sparsity.py --sparsity  0.2  > ${OUTPUT_DIR}/0.2
python3 /workspace/PyTorchSim/tests/test_sparsity.py --sparsity  0.4  > ${OUTPUT_DIR}/0.4
python3 /workspace/PyTorchSim/tests/test_sparsity.py --sparsity  0.6  > ${OUTPUT_DIR}/0.6
python3 /workspace/PyTorchSim/tests/test_sparsity.py --sparsity  0.8  > ${OUTPUT_DIR}/0.8

OUTPUT_DIR="24GB"
export TORCHSIM_CONFIG="/workspace/PyTorchSim/configs/systolic_ws_8x8_c1_24G_simple_noc.yml"
python3 /workspace/PyTorchSim/tests/test_sparsity.py --sparsity  0.0  > ${OUTPUT_DIR}/0.0
python3 /workspace/PyTorchSim/tests/test_sparsity.py --sparsity  0.2  > ${OUTPUT_DIR}/0.2
python3 /workspace/PyTorchSim/tests/test_sparsity.py --sparsity  0.4  > ${OUTPUT_DIR}/0.4
python3 /workspace/PyTorchSim/tests/test_sparsity.py --sparsity  0.6  > ${OUTPUT_DIR}/0.6
python3 /workspace/PyTorchSim/tests/test_sparsity.py --sparsity  0.8  > ${OUTPUT_DIR}/0.8

OUTPUT_DIR="48GB"
export TORCHSIM_CONFIG="/workspace/PyTorchSim/configs/systolic_ws_8x8_c1_48G_simple_noc.yml"
python3 /workspace/PyTorchSim/tests/test_sparsity.py --sparsity  0.0  > ${OUTPUT_DIR}/0.0
python3 /workspace/PyTorchSim/tests/test_sparsity.py --sparsity  0.2  > ${OUTPUT_DIR}/0.2
python3 /workspace/PyTorchSim/tests/test_sparsity.py --sparsity  0.4  > ${OUTPUT_DIR}/0.4
python3 /workspace/PyTorchSim/tests/test_sparsity.py --sparsity  0.6  > ${OUTPUT_DIR}/0.6
python3 /workspace/PyTorchSim/tests/test_sparsity.py --sparsity  0.8  > ${OUTPUT_DIR}/0.8

OUTPUT_DIR="12GB_2core"
export TORCHSIM_CONFIG="/workspace/PyTorchSim/configs/systolic_ws_8x8_c2_12G_simple_noc.yml"
python3 /workspace/PyTorchSim/tests/test_sparsity.py --sparsity  0.0  > ${OUTPUT_DIR}/0.0
python3 /workspace/PyTorchSim/tests/test_sparsity.py --sparsity  0.2  > ${OUTPUT_DIR}/0.2
python3 /workspace/PyTorchSim/tests/test_sparsity.py --sparsity  0.4  > ${OUTPUT_DIR}/0.4
python3 /workspace/PyTorchSim/tests/test_sparsity.py --sparsity  0.6  > ${OUTPUT_DIR}/0.6
python3 /workspace/PyTorchSim/tests/test_sparsity.py --sparsity  0.8  > ${OUTPUT_DIR}/0.8

OUTPUT_DIR="24GB_2core"
export TORCHSIM_CONFIG="/workspace/PyTorchSim/configs/systolic_ws_8x8_c2_24G_simple_noc.yml"
python3 /workspace/PyTorchSim/tests/test_sparsity.py --sparsity  0.0  > ${OUTPUT_DIR}/0.0
python3 /workspace/PyTorchSim/tests/test_sparsity.py --sparsity  0.2  > ${OUTPUT_DIR}/0.2
python3 /workspace/PyTorchSim/tests/test_sparsity.py --sparsity  0.4  > ${OUTPUT_DIR}/0.4
python3 /workspace/PyTorchSim/tests/test_sparsity.py --sparsity  0.6  > ${OUTPUT_DIR}/0.6
python3 /workspace/PyTorchSim/tests/test_sparsity.py --sparsity  0.8  > ${OUTPUT_DIR}/0.8

OUTPUT_DIR="48GB_2core"
export TORCHSIM_CONFIG="/workspace/PyTorchSim/configs/systolic_ws_8x8_c2_48G_simple_noc.yml"
python3 /workspace/PyTorchSim/tests/test_sparsity.py --sparsity  0.0  > ${OUTPUT_DIR}/0.0
python3 /workspace/PyTorchSim/tests/test_sparsity.py --sparsity  0.2  > ${OUTPUT_DIR}/0.2
python3 /workspace/PyTorchSim/tests/test_sparsity.py --sparsity  0.4  > ${OUTPUT_DIR}/0.4
python3 /workspace/PyTorchSim/tests/test_sparsity.py --sparsity  0.6  > ${OUTPUT_DIR}/0.6
python3 /workspace/PyTorchSim/tests/test_sparsity.py --sparsity  0.8  > ${OUTPUT_DIR}/0.8
