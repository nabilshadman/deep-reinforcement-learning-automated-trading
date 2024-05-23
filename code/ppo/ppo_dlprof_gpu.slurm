#!/bin/bash
#
#SBATCH --job-name=ppo-dlprof-gpu
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00

# Replace [budget code] below with your project code (e.g. t01)
#SBATCH --account=mdisspt-s2134758

# Load the required modules
module load python/3.10.8-gpu

# Load conda environment
CONDA_ROOT=/work/mdisspt/mdisspt/n2134758/condaenvs
export CONDARC=${CONDA_ROOT}/.condarc
eval "$(conda shell.bash hook)"

conda activate env-pytorch-1.13.1-gpu

# Change to the submission directory
cd $SLURM_SUBMIT_DIR

# Launch the job
# dlprof --mode=pytorch --delay=3 --reports=summary,detail,iteration --output_path=./dlprof_output --force=true python ppo_trader_dlprof.py -m train
# srun -n 1 nsys profile --delay=3 -o prof1 python ppo_trader.py -m train
python ppo_trader_tensorboard.py -m train
