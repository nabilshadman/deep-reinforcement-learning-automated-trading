#!/bin/bash

# Slurm job options (name, compute nodes, job time)
#SBATCH --job-name=dqn-cpu
#SBATCH --time=0:20:0
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
# dlprof --mode=pytorch --delay=3 --reports=summary,detail,iteration --output_path=./dlprof_output --force=true python dqn_trader_dlprof.py -m train
# srun -n 1 nsys profile --delay=3 -o prof1 python dqn_trader.py -m train
python dqn_trader.py -m train
python dqn_trader.py -m test

