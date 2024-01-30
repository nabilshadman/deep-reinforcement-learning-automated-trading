#!/bin/bash

# Slurm job options (name, compute nodes, job time)
#SBATCH --job-name=test-pytorch-cpu
#SBATCH --time=0:10:0
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=36

# Replace [budget code] below with your budget code (e.g. dc116-s1234567)
#SBATCH --account=mdisspt-s2134758
# We use the "standard" partition as we are running on CPU nodes
#SBATCH --partition=standard
# We use the "standard" QoS as our runtime is more than 20 mins 
#SBATCH --qos=standard

# Load required modules
module load pytorch

# Load miniconda environment (if needed)
#source /work/m22ol/m22ol/t2134758/miniconda-init.sh
#conda activate pt

# Change to the submission directory
cd $SLURM_SUBMIT_DIR

# Launch the parallel job
python test_dqn.py

