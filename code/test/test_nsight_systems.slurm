#!/bin/bash

#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --gres=gpu:1

# Replace [budget code] below with your project code (e.g. t01)
#SBATCH --account=m22ol-s2134758

# Load the required modules
module load gcc
module load nvidia/nvhpc

# Change to the submission directory
cd $SLURM_SUBMIT_DIR

# Compile the C++ program
g++ -o test_nsight_systems test_nsight_systems.cpp

# Run the C++ program with NSight Systems profiling
srun -n 1 nsys profile --output=nsight1 ./test_nsight_systems
