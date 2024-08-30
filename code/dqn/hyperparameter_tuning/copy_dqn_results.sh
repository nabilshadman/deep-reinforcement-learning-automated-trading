#!/bin/bash

# Create result directories if they don't exist
mkdir -p dqn_tuning_cpu_results
mkdir -p dqn_tuning_gpu_results

# Function to copy SLURM files
copy_slurm_files() {
    local config_type=$1
    local result_dir=$2

    for i in {1..27}; do
        source_dir="dqn_tuning_${config_type}_config${i}"
        if [ -d "$source_dir" ]; then
            echo "Processing $source_dir"
            find "$source_dir" -name "slurm-*.out" -exec cp {} "$result_dir" \;
        else
            echo "Warning: $source_dir does not exist. Skipping."
        fi
    done
}

# Copy CPU config results
copy_slurm_files "cpu" "dqn_tuning_cpu_results"

# Copy GPU config results
copy_slurm_files "gpu" "dqn_tuning_gpu_results"

echo "Script execution completed."