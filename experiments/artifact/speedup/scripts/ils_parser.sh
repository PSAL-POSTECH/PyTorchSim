#!/bin/bash

ignore_rest=false
gem5_cmd=""
result_path=""
gem5_time=""
togsim_time=""

total_gem5=0
total_togsim=0

while IFS= read -r line; do
  if [[ "$line" == launch* ]]; then
    tile_graph_path=$(echo "$line" | awk '{for (i=1; i<=NF; i++) if ($i ~ /tile_graph\.onnx$/) print $i}')
    if [[ -n "$tile_graph_path" ]]; then
      dir_path=$(dirname "$tile_graph_path")
      sto_log_path="$dir_path/m5out/sto.log"
      echo "sto.log path: $sto_log_path"
      gem5_time=$(grep "Simulation time:" "$sto_log_path" | \
                sed -E 's/^Simulation time: ([0-9.]+) seconds$/\1/')
      echo "GEM5: $gem5_time" 
      total_gem5=$(awk -v a="$total_gem5" -v b="$gem5_time" 'BEGIN {printf "%.6f", a+b}')
    fi
  fi
  if [[ "$line" == *"Simulation time:"* ]]; then
    togsim_time=$(echo "$line" | sed -E 's/.*Simulation time: ([0-9.]+) seconds/\1/')
    echo "TOGSim: $togsim_time"
  fi
done

if [[ -n "$total_gem5" && -n "$total_togsim" ]]; then
  total_time=$(python3 -c "print(round($total_gem5 + $total_togsim, 6))")
  echo "Simulation time: $total_time seconds"
fi