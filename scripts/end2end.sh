#!/bin/bash

# Base directory
BASE_PATH=$1 # Input as the first argument

# Initialize the total cycle sum
total_sum=0
total_core=0
total_vector=0
# Find all togsim_result folders
mapfile -t togsim_folders < <(find "$BASE_PATH" -type d -name "togsim_result")

# Iterate over each togsim_result folder
for togsim_folder in "${togsim_folders[@]}"; do
  # echo "Processing folder: $togsim_folder"

  # Find all files within the togsim_result folder
  mapfile -t files < <(find "$togsim_folder" -type f)

  for file in "${files[@]}"; do
    # echo "Processing $file"

    # Extract the last line containing "Total_cycles"
    total_cycle=$(grep "Total_cycles" "$file" | tail -n 1 | sed -E 's/.*Total_cycles ([0-9]+).*/\1/')
    # echo "total_cycle: $total_cycle"
    active_cycles=($(grep -o 'active_cycles [0-9]*' "$file" | awk '{print $3}'))
    num_cycles=${#active_cycles[@]}
    if [ "$num_cycles" -ge 3 ]; then
        core_cycle=${active_cycles[$((num_cycles-3))]}
    else
        echo "Error: cannot find core active_cycles"
    fi
    if [[ "$num_cycles" -ge 1 ]]; then
        # Extract the last two active_cycless
        vector_core_cycle=${active_cycles[$((num_cycles-1))]}
    else
        echo "Error: cannot find vector core active_cycles"
    fi
    echo "file: $file total_cycle: $total_cycle SA core_cycle: $core_cycle vector_core_cycle: $vector_core_cycle"

    if [[ -n "$total_cycle" ]]; then
      # Add the total cycle to the total sum
      # echo "Adding $total_cycle to total_sum"
      total_sum=$((total_sum + total_cycle))
    fi
    if [[ -n "$core_cycle" ]]; then
      # Add the total cycle to the total sum
      # echo "Adding $total_cycle to total_sum"
      total_core=$((total_core + core_cycle))
    fi
    if [[ -n "$vector_core_cycle" ]]; then
      # Add the total cycle to the total sum
      # echo "Adding $total_cycle to total_sum"
      total_vector=$((total_vector + vector_core_cycle))
    fi
  done
done

# Print the total cycle sum
echo "total end2end cycle: $total_sum"
echo "total core cycle: $total_core"
echo "total vector core cycle: $total_vector"