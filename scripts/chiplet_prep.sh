#!/bin/bash

sizes=(256 512 1024 2048)
# 각 size에 대해 처리
for size in "${sizes[@]}"; do
    echo "Processing size: $size"

    # 환경 변수 설정
    export TORCHSIM_FORCE_TIME_M=$((size / 2))
    export TORCHSIM_FORCE_TIME_K=$((size / 2))
    export TORCHSIM_FORCE_TIME_N=$((size / 2))
    export TORCHSIM_DUMP_PATH=$(pwd)/chiplet_result/$size
    python3 chiplet_prep.py $size
    #python3 chiplet_run.py $(pwd)/chiplet_result
done