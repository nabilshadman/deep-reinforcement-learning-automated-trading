#!/bin/bash
#
#SBATCH --job-name=dqn-tensorflow-gpu
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

# Replace [budget code] below with your project code (e.g. t01)
#SBATCH --account=m22ol-s2134758

# Load the required modules
module load tensorflow/2.9.1-gpu

# Change to the submission directory
cd $SLURM_SUBMIT_DIR

# Launch the parallel job
python rl_trader.py -m train

