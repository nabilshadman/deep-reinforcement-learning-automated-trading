#!/bin/bash
#
#SBATCH --job-name=dqn-pytorch-gpu
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00

# Replace [budget code] below with your project code (e.g. t01)
#SBATCH --account=m22ol-s2134758

# Load the required modules
# module load pytorch

# Load miniconda environment (if needed)
source /work/m22ol/m22ol/t2134758/miniconda-init.sh
conda activate pt

# Change to the submission directory
cd $SLURM_SUBMIT_DIR

# Launch the parallel job
python rl_trader_prof.py -m train

