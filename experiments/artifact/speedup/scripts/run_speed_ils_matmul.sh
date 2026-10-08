#!/bin/bash

base_dir=$TORCHSIM_DIR/experiments/artifact/speedup
config=(
    "systolic_ws_128x128_c2_simple_noc_tpuv3_ils.yml"
)
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
SHAPE_LIST=(
  "512 512 512"
  "1024 1024 1024"
  "2048 2048 2048"
)
output_dir="$base_dir/results"
mkdir -p "$output_dir"

for i in "${config[@]}"; do
  echo "Running with config=$i"
  for shape in "${SHAPE_LIST[@]}"; do
    ops="gemm_${shape// /x}"
    output_file="$output_dir/ils_${ops}_${i}.txt"
    workload="$TORCHSIM_DIR/experiments/gemm.py --size $shape"
    echo "===== config=$i | model=$ops =====" >> "$output_file"
    sum=0.0
    count=0
    config_path="$TORCHSIM_DIR/configs/$i"

    for iter in {1..5}; do
      echo "[Iter $iter] Running simulation for workload=ils_$ops config=$config"
      output=$(bash -c "
        export TOGSIM_CONFIG=$config_path;
        cd $TORCHSIM_DIR && python3 $workload 2>&1
      ")

      sim_time=$(echo "$output" | grep "Wall-clock time for simulation:" | tail -n 1 | sed -E 's/.*Wall-clock time for simulation: ([0-9]+\.[0-9]+) seconds.*/\1/')

      if [[ -n "$sim_time" ]]; then
        echo "Iteration $iter: simulation_time = $sim_time" >> "$output_file"
        sum=$(awk -v a="$sum" -v b="$sim_time" 'BEGIN {printf "%.6f", a + b}')
        count=$((count + 1))
      else
        echo "Iteration $iter: Simulation time not found."
        echo "Iteration $iter: simulation_time = NA" >> "$output_file"
      fi
    done

    if [[ $count -gt 0 ]]; then
      avg=$(awk -v total="$sum" -v n="$count" 'BEGIN {printf "%.6f", total / n}')
      echo "Average simulation time for $ops with config $i: $avg seconds"
      echo "Average simulation time = $avg" >> "$output_file"
    else
      echo "No valid simulation times found for $ops with config $i"
      echo "Average simulation time = NA" >> "$output_file"
    fi
    echo "" >> "$output_file"
  done
done