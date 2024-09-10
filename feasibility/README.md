# Feasibility Study

This folder contains the resources and analysis conducted during the initial feasibility study for this project. The study aimed to evaluate the suitability of Deep Reinforcement Learning (DRL) models, particularly the Deep Q-Network (DQN), for automated trading using HPC resources.

## Contents

- **`feasibility_report.pdf`**: A summary report of the feasibility study, covering the technical evaluation of DRL models, their performance on multi-stock trading tasks, and their scalability on HPC platforms like Cirrus.
- **`models/`**: This folder contains the models and scripts tested during the study, focusing on the DQN algorithm.
  - **DQN**: Contains both PyTorch and TensorFlow implementations of the DQN algorithm.
- **Profiling Data**: Logs and profiling outputs for DQN models, gathered from tests run on CPU and GPU nodes. This data is useful for understanding execution times, memory usage, and computational performance at scale.

## Usage

- **Run Models**: SLURM scripts are provided in the `models/` directory for executing the DQN models on Cirrus. These include scripts for both CPU and GPU nodes.
- **Performance Metrics**: The `miscellaneous/` subdirectory contains profiling data, including execution times and resource utilisation on HPC resources. Use this data to assess the efficiency and scalability of the models.

For more details, refer to the `feasibility_report.pdf`, which includes findings and insights from the study.
