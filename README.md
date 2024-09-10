
# Deep Reinforcement Learning for Automated Trading on HPC

This repository contains the implementation and evaluation of Deep Reinforcement Learning (DRL) models for automated trading on High-Performance Computing (HPC) system. The project focuses on two widely used DRL algorithms: Deep Q-Network (DQN) and Proximal Policy Optimisation (PPO). Our goal is to compare these models in terms of both trading performance and computational efficiency across CPU and GPU nodes on the Cirrus HPC system.

Key Features:
- Implementation of DQN and PPO algorithms for multi-asset trading
- Integration with HPC resources for scalable training and testing
- Performance evaluation on CPU and GPU nodes
- Comprehensive analysis of trading metrics and computational efficiency

This project is part of an MSc dissertation titled "Comparison of Deep Reinforcement Learning Models for Automated Trading on Heterogeneous HPC System".

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

## Hardware Environment
[Cirrus](https://www.epcc.ed.ac.uk/hpc-services/cirrus) is our primary HPC platform for testing our implementations, offering both CPU and GPU nodes to efficiently train and evaluate our DRL models.  

## Software Environment
[PyTorch](https://pytorch.org/) is our primary machine learning framework for implementing DQN and PPO models. In addition to PyTorch, we explored other frameworks or libraries such as TensorFlow and TorchRL during the feasibility and prototyping phases to assess their suitability for the project. The environment is managed through [Conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) to ensure reproducibility across platforms.

## How to Set Up (on Cirrus)

### 1. Set Up the Environment
To set up the project environment using Conda:
```bash
conda env create -f environment.yml
conda activate your-env-name
```

Alternatively, install Python packages using `pip`:
```bash
pip install -r requirements.txt
```

### 2. Run the Experiments
Use the provided SLURM scripts in the respective model's directory (i.e. DQN, PPO) to run the programs on Cirrus:

- **For CPU nodes**:
  ```bash
  # DQN
  sbatch dqn_cpu.slurm

  # PPO
  sbatch ppo_cpu.slurm
  ```

- **For GPU nodes**:
  ```bash
  # DQN
  sbatch dqn_gpu.slurm

  # PPO
  sbatch ppo_gpu.slurm
  ```

## How to Set Up (on local machine)
To run the DQN or PPO programs on a local machine (e.g, Windows, Mac):  

### 1. Install dependencies:

```bash
# Using conda
conda env create -f environment.yml
conda activate your-env-name

# Or using pip
pip install -r requirements.txt
```

### 2. Train the model:
From the respective model's directory:  
```bash
# DQN
python dqn_trader.py -m train -c config.yaml

# PPO
python ppo_trader.py -m train -c config.yaml
```

### 3. Test the model:
From the respective model's directory:
```bash
# DQN
python dqn_trader.py -m test -c config.yaml

# PPO
python ppo_trader.py -m test -c config.yaml
```

Ensure any necessary configuration changes are made in the ```config.yaml``` file before running.

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

The project uses daily equity or exchange traded fund (ETF) close price data, publicly available from the [Yahoo Finance](https://pypi.org/project/yfinance/) API. The data files are located at `code/data/`.

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
