# TorchRL DQN

This folder contains the implementation of the Deep Q-Network (DQN) agent using the PyTorch Reinforcement Learning (TorchRL) library. The program is currently under development and is intended to leverage the flexibility of TorchRL for building, training, and evaluating the DQN agent in a multi-asset trading environment. The agent implementation has been adapted from the open-source examples of [TorchRL](https://github.com/pytorch/rl).

## Contents

- **`dqn_trader_torchrl.py`**: The main script for training the DQN agent using TorchRL. It integrates the custom multi-stock trading environment (`multistock_env_torchrl.py`) to optimise trading strategies over time.
  
- **`multistock_env_torchrl.py`**: Defines the multi-asset trading environment in TorchRL, which serves as the interface between the agent and the market simulation.

## Usage

To run the DQN agent on Cirrus HPC, use the appropriate SLURM script based on the target resources:

### On CPU:
```bash
sbatch dqn_cpu.slurm
```
### On GPU:
```bash
sbatch dqn_gpu.slurm
```

The training logs will be saved in the `logs/` folder for analysis and debugging. The program is under development, and updates will be made to improve functionality and performance. The `dqn_trader_torchrl.py` script can also be run locally for testing purposes, though GPU acceleration is recommended for larger experiments.
