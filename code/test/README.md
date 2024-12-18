# Correctness Testing

The `test/` folder contains scripts designed to test the functionality and correctness of key components in the Deep Reinforcement Learning (DRL) trading models. These tests ensure that the environment and agents behave as expected before full-scale training and experimentation.

## Contents

- **`test_environment.py`**: Tests the multi-asset trading environment by feeding in dummy actions (buy, sell, hold) to check that the environment processes these correctly and returns expected outputs such as updated states and rewards.

- **`test_dqn.py`**: Tests the DQN agent’s functionality by running it in the trading environment. This script verifies if the agent’s policy network correctly selects actions based on the current state and updates its Q-values after each step.

- **`test_ppo.py`**: Tests the PPO agent’s functionality. The script ensures that the actor-critic architecture is working as expected by evaluating actions, log probabilities, and value estimates during each environment step.

- **`test_multistock_env_torchrl.py`**: Tests the TorchRL-based multi-asset trading environment to ensure compatibility with the TorchRL framework. This script is essential for verifying that the environment works well with new RL libraries and toolkits.

- **`logs/`**: This folder stores output logs generated by the test scripts. It is useful for debugging and reviewing the performance of the environment and agents during the tests.

## Usage

These test scripts are essential for verifying that the trading environment and agents are functioning properly before running full experiments. They can be run as follows:

```bash
python test_environment.py
python test_dqn.py
python test_ppo.py
python test_multistock_env_torchrl.py
