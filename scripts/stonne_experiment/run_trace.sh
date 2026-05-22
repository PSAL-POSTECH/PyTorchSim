#!/bin/bash

SCRIPT="/workspace/PyTorchSim/tests/system/test_stonne.py"

SIZES=(32 64 128)
SPARSITIES=(0.0 0.2 0.4 0.6 0.8)

for sz in "${SIZES[@]}"; do
    for sparsity in "${SPARSITIES[@]}"; do
        FILE_PATH=$(python "$SCRIPT" "$sz" "$sparsity" | grep -oP '(?<=stored to ")[^"]+')
        TOTAL_CYCLE=$(grep -oP '\[.*?\] \[info\] Stonne Core \[0\] : Total cycle \K\d+' "$FILE_PATH" | tail -n 1)
        echo "Stonne $sz $sparsity $TOTAL_CYCLE"

        FILE_PATH=$(python "$SCRIPT" "$sz" "$sparsity" | grep -oP '(?<=stored to ")[^"]+')
        TOTAL_CYCLE=$(grep -oP '\[.*?\] \[info\] Stonne Core \[0\] : Total cycle \K\d+' "$FILE_PATH" | tail -n 1)
        echo "TOG $sz $sparsity $TOTAL_CYCLE"
    done
done