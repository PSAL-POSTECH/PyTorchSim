#!/bin/bash

# Define the sparsity and block values
sparsity_values=(0.0 0.2 0.4 0.6 0.8 0.9 0.95 0.97 0.99)
block_values=(8 4 1)

# Base output directory
output_dir="results"

# Create the base output directory
mkdir -p "$output_dir"

# Iterate over all combinations of sparsity and block values
for block in "${block_values[@]}"; do
    for sparsity in "${sparsity_values[@]}"; do
        # Construct the folder name and file name
        folder_name="$output_dir/sparsity/${sparsity}/block_${block}"
        output_file="$folder_name/output.txt"
        
        # Create the folder for this combination
        mkdir -p "$folder_name"
        
        # Construct the command
        command="python3 test_sparsity.py --sparsity $sparsity --block $block"
        
        # Print the command for tracking
        echo "Running: $command"
        
        # Execute the command and save output to file
        $command > "$output_file" 2>&1 &
        
        # Check if the command was successful
        if [ $? -ne 0 ]; then
            echo "Error occurred while running: $command"
            echo "Check logs in: $output_file"
        else
            echo "Output saved to: $output_file"
        fi
    done
    wait
done
