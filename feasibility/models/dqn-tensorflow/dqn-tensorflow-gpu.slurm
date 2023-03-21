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
# Set the below only when using tensorflow/2.11.0-gpu
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/mnt/lustre/indy2lfs/sw/nvidia/hpcsdk-222/Linux_x86_64/22.2/cuda/
module load tensorflow/2.11.0-gpu

# Change to the submission directory
cd $SLURM_SUBMIT_DIR

# Launch the parallel job
python rl_trader.py -m train

