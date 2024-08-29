#!/bin/bash

# Check if required files exist
required_files=("ppo_trader.py" "equities_daily_close_2018_2023.csv" "clean.sh" "ppo_cpu.slurm" "ppo_gpu.slurm")
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "Error: $file not found in the current directory."
        exit 1
    fi
done

# Create 54 folders (27 for CPU and 27 for GPU)
for device in cpu gpu; do
    for i in {1..27}; do
        folder_name="ppo_tuning_${device}_config${i}"
        mkdir -p "$folder_name"

        # Copy common files
        cp ppo_trader.py equities_daily_close_2018_2023.csv clean.sh "$folder_name/"

        # Copy and rename specific config file
        cp "config${i}.yaml" "$folder_name/config.yaml"

        # Copy corresponding slurm file
        cp "ppo_${device}.slurm" "$folder_name/"

        echo "Created and populated $folder_name"
    done
done

echo "All 54 folders have been created and populated."