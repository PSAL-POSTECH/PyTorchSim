#!/bin/bash

# Base directory
BASE_PATH=$1 # Input as the first argument

# Initialize total_sum as string for awk processing
total_sum=0.0

# Find all togsim_result folders
mapfile -t togsim_folders < <(find "$BASE_PATH" -type d -name "togsim_result")

# Iterate over each togsim_result folder
for togsim_folder in "${togsim_folders[@]}"; do
  mapfile -t files < <(find "$togsim_folder" -type f)

  for file in "${files[@]}"; do
    sim_time=$(grep "Wall-clock time for simulation:" "$file" | tail -n 1 | sed -E 's/.*Wall-clock time for simulation: ([0-9]+(\.[0-9]+)?).*/\1/')
    echo "file: $file total_cycle: $sim_time"

    if [[ -n "$sim_time" ]]; then
      total_sum=$(awk -v a="$total_sum" -v b="$sim_time" 'BEGIN {printf "%.6f", a + b}')
    fi
  done
done

# Print the total simulation time
echo "simulation time: $total_sum"
