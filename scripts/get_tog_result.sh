#!/bin/bash
total_cycles=0

# Read through input stream line by line
while IFS= read -r line; do
    # Check if the line contains both "[TOGSimulator]" and "stored"
    if [[ "$line" == *"[TOGSimulator]"* && "$line" == *"stored"* ]]; then
        # Extract the file path from the line
        file_path=$(echo "$line" | sed -n 's/.*stored to "\(.*\)"$/\1/p')
        
        # If the file exists, grep for "Total cycle" and output the last matching line
        if [[ -f "$file_path" ]]; then
            last_line=$(grep "Total cycle" "$file_path" | tail -n 1)
            echo "$last_line ($file_path)"
            # Accumulate the cycle value
            cycle_value=$(echo "$last_line" | sed -n 's/.*Total cycle \([0-9]\+\)$/\1/p')
            total_cycles=$((total_cycles + cycle_value))
        else
            echo "File not found: $file_path"
        fi
    fi
    # Check if the line ends with "Test passed|"
    if [[ "$line" == *"Test Passed|" ]]; then
        echo "$line"
        echo "Accumulated Total Cycle: $total_cycles"
        total_cycles=0
    fi
    if [[ "$line" == *"Test Failed|" ]]; then
        echo "$line"
        echo "Accumulated Total Cycle: $total_cycles"
        total_cycles=0
    fi
    if [[ "$line" == *"[log]"* ]]; then
        echo "$line"
    fi
done