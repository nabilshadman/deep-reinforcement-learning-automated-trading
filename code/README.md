# Code Directory

This directory contains all the source code and scripts for the project, organized into subdirectories based on specific functionality and models.

## Structure

- `data/` – Contains datasets used for training and testing the models, as well as scripts to download financial data from Yahoo Finance API.
- `dqn/` – Holds the Deep Q-Network (DQN) implementation, including the main scripts and SLURM scripts for both CPU and GPU execution.
- `ppo/` – Holds the Proximal Policy Optimisation (PPO) implementation, including the main scripts and SLURM scripts for both CPU and GPU execution.
- `test/` – Contains test scripts for validating the environment, agents, and other components.
- `torchrl_dqn/` – Contains DQN implementation using the TorchRL framework (in progress).
- `torchrl_ppo/` – Contains PPO implementation using the TorchRL framework (in progress).

## Usage

Each model folder (e.g., `dqn/`, `ppo/`) contains its own scripts for running experiments and profiling on HPC systems. Refer to the respective README files or SLURM scripts for details on execution and profiling.
