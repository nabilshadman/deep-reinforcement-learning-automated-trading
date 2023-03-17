#!/bin/bash

# Slurm job options (name, compute nodes, job time)
#SBATCH --job-name=dqn-tensorflow
#SBATCH --time=0:10:0
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1

# Replace [budget code] below with your budget code (e.g. dc116-s1234567)
#SBATCH --account=m22ol-s2134758
# We use the "standard" partition as we are running on CPU nodes
#SBATCH --partition=standard
# We use the "standard" QoS as our runtime is more than 20 mins 
#SBATCH --qos=short

module load tensorflow/2.10.0

# Change to the submission directory
cd $SLURM_SUBMIT_DIR

# Launch the parallel job
python rl_trader.py -m train

