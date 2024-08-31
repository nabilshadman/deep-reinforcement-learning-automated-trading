#!/bin/bash

# Create folders
for device in cpu gpu; do
    for config in {1..4}; do
        folder_name="ppo_scaling_${device}_config${config}"
        mkdir -p "$folder_name"
        
        # Copy common files
        cp clean.sh ppo_trader.py "$folder_name/"
        
        # Copy device-specific SLURM script
        cp "ppo_${device}.slurm" "$folder_name/"
        
        # Copy and rename config file
        cp "config${config}.yaml" "$folder_name/config.yaml"
        
        # Copy appropriate CSV file
        case $config in
            1) csv_file="equities_daily_close_1_tickers_2018_2023.csv" ;;
            2) csv_file="equities_daily_close_3_tickers_2018_2023.csv" ;;
            3) csv_file="equities_daily_close_5_tickers_2018_2023.csv" ;;
            4) csv_file="equities_daily_close_10_tickers_2018_2023.csv" ;;
        esac
        cp "$csv_file" "$folder_name/"
        
        echo "Created and populated $folder_name"
    done
done

echo "All PPO folders created and files copied successfully!"