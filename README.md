
# Deep Reinforcement Learning for Automated Trading on HPC

This repository contains the code, configurations, and results for the MSc dissertation project: **"Comparison of Deep Reinforcement Learning Models for Automated Trading on Heterogeneous HPC Systems"**. The project implements and evaluates two DRL algorithms: **Deep Q-Network (DQN)** and **Proximal Policy Optimisation (PPO)**, focusing on performance in terms of both **trading metrics** and **computational efficiency** across CPU and GPU nodes on the Cirrus high performance computing (HPC) system.

## Repository Structure
Here is a high-level overview of the repository's structure: 

```
/
├── code/
│   ├── data/
│   ├── dqn/
│   ├── ppo/
│   ├── test/
│   ├── torchrl_dqn/
│   └── torchrl_ppo/
├── feasibility/
├── environment.yml
├── requirements.txt
├── README.md
```

- `code/` – Contains all code-related directories for the project.
  - `data/` – Scripts to download and store datasets used in experiments (e.g., equity prices).
  - `dqn/` – Scripts and implementation for the DQN model.
  - `ppo/` – Scripts and implementation for the PPO model.
  - `test/` – Test scripts to validate environments, agents, and other components.
  - `torchrl_dqn/` – Contains DQN implementation using the TorchRL framework (in progress).
  - `torchrl_ppo/` – Contains PPO implementation using the TorchRL framework (in progress).
- `feasibility/` – Documents related to the feasibility study conducted in the initial stage of the project.
- `environment.yml` – Conda environment configuration file for setting up project dependencies.
- `requirements.txt` – Lists Python packages required to run the project.


## How to Set Up

### 1. Setting Up the Environment
To set up the project environment using Conda:
```bash
conda env create -f environment.yml
conda activate your-env-name
```

Alternatively, install Python packages using `pip`:
```bash
pip install -r requirements.txt
```

### 2. Running the Experiments
Use the provided SLURM scripts in the respective model's directory (i.e. DQN, PPO) to run the programs on Cirrus:

- **For CPU nodes**:
  ```bash
  sbatch dqn_cpu.slurm
  sbatch ppo_cpu.slurm
  ```

- **For GPU nodes**:
  ```bash
  sbatch dqn_gpu.slurm
  sbatch ppo_gpu.slurm
  ```

## Configuration

The models use YAML configuration files for hyperparameters. These can be found in the respective model directories. Make sure to adjust the file if you need to change hyperparameters.

## Outputs and Logs

Logs generated from SLURM jobs are stored in the `logs/` folder for each model.  

The models output several key metrics:
- Portfolio value statistics (median, min, max)
- Execution time
- CPU/GPU memory usage

Profiling and performance metrics can be found within the log files.


## Data

The project uses daily equity or exchange traded fund (ETF) close price data, publicly available from the Yahoo Finance [API](https://pypi.org/project/yfinance/). The data files are located at `code/data/`.

## Profiling

Separate SLURM script for profiling are provided in the respective model's folder. These scripts integrate profiling tools such as `torch.profiler` to gather performance metrics like execution time and memory usage. Use these profiling-specific scripts to enable profiling.

## GPU Monitoring

For GPU runs, there's a commented-out section in the SLURM scripts to collect GPU monitoring data (e.g. power, utilisation) using `nvidia-smi`. Uncomment this section to enable GPU monitoring.

## Open-Source Code and Enhancements

This project builds upon the open-source implementations of the DQN algorithm (developed by [Lazy Programmer](https://github.com/lazyprogrammer/machine_learning_examples)) and the PPO algorithm (developed by [Phil Tabor](https://github.com/philtabor/Youtube-Code-Repository)). Both authors are experienced machine learning practitioners who promote experimentation with their implementations. 

These algorithms were adapted and enhanced for a multi-asset trading environment and integrated with HPC resources. Some of our enhancements include GPU support, environment extensions, YAML-based configuration management, and model architecture improvements for automated trading tasks on HPC systems. 

For more details on these enhancements and their impact, refer to the project report.

## Project Wiki

For detailed notes, meeting summaries, experimental observations, and literature references, please refer to the [Project Wiki](https://git.ecdf.ed.ac.uk/msc-22-23/s2134758/-/wikis/home). The wiki includes key information about the project's progress, experimental results, and resources used throughout the development process.

## Contributors

**Researcher:** Nabil Shadman  
**Advisors:** Dr Joseph Lee, Dr Michael Bareford  

## Contact
For questions or issues, feel free to contact Nabil Shadman at n.shadman@sms.ed.ac.uk.

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.  
