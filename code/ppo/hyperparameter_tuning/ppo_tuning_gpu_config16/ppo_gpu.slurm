#!/bin/bash
#
#SBATCH --job-name=ppo-gpu
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

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

# Run the job 3 times
for run in {1..3}
do
    echo "Starting Run $run"
    echo

    python ppo_trader.py -m train -c config.yaml
    python ppo_trader.py -m test -c config.yaml
    
    echo
    echo "Finished Run $run"
    echo
    
    # Clean folders
    source clean.sh
    
    echo "Cleaned folders after Run $run"
    echo
    echo "------------------------"
    echo
done
