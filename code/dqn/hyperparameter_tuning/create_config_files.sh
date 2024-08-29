#!/bin/bash

# Check if config.yaml exists
if [ ! -f "config.yaml" ]; then
    echo "Error: config.yaml not found in the current directory."
    exit 1
fi

# Loop to create 27 copies of config.yaml
for i in {1..27}
do
    cp config.yaml "config${i}.yaml"
    echo "Created config${i}.yaml"
done

echo "All 27 configuration files have been created."