#!/bin/bash

sizes=(256 512 1024 2048)
for size in "${sizes[@]}"; do
    echo "Processing size: $size"

    # Set environment variables
    export TORCHSIM_TILE_M=$((size / 2))
    export TORCHSIM_TILE_K=$((size / 2))
    export TORCHSIM_TILE_N=$((size / 2))
    export TORCHSIM_LOG_PATH=$(pwd)/chiplet_result/$size
    python3 chiplet_prep.py $size
    #python3 chiplet_run.py $(pwd)/chiplet_result
done