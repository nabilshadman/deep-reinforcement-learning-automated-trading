
# Deep Reinforcement Learning for Automated Trading on HPC

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)
![HPC](https://img.shields.io/badge/HPC-Cirrus-orange)

This repository contains the implementation and evaluation of Deep Reinforcement Learning (DRL) models for automated trading on High-Performance Computing (HPC) system. The project focuses on two widely used DRL algorithms: Deep Q-Network (DQN) and Proximal Policy Optimisation (PPO). Our goal is to compare these models in terms of both trading performance and computational efficiency across CPU and GPU nodes on the Cirrus HPC system.

Key Features:
- Implementation of DQN and PPO algorithms for multi-asset trading
- Integration with HPC resources for scalable training and testing
- Performance evaluation on CPU and GPU nodes
- Comprehensive analysis of trading metrics and computational efficiency

This project is part of an MSc [dissertation](https://github.com/nabilshadman/deep-reinforcement-learning-automated-trading/blob/main/report/report.pdf) titled "Comparison of Deep Reinforcement Learning Models for Automated Trading on Heterogeneous HPC System".

<table width="100%">
<tr>
<td width="50%"><strong>Deep Q-Network (DQN)</strong></td>
<td width="50%"><strong>Proximal Policy Optimisation (PPO)</strong></td>
</tr>
<tr>
<td><img src="https://github.com/user-attachments/assets/9411f821-e2db-4941-a863-2b22c21e3c2f" width="100%" alt="DQN Architecture"></td>
<td><img src="https://github.com/user-attachments/assets/1a3e477b-271f-4b32-83dd-566dc3fb3524" width="100%" alt="PPO Architecture"></td>
</tr>
<tr>
<td><em>DQN architecture [1]</em></td>
<td><em>PPO architecture [2]</em></td>
</tr>
</table>


## Table of Contents
- [Repository Structure](#repository-structure)
- [Hardware Environment](#hardware-environment)
- [Software Environment](#software-environment)
- [Prerequisites](#prerequisites)
- [How to Set Up (on Cirrus)](#how-to-set-up-on-cirrus)
- [How to Set Up (on local machine)](#how-to-set-up-on-local-machine)
- [Configuration](#configuration)
- [Outputs and Logs](#outputs-and-logs)
- [Data](#data)
- [Profiling](#profiling)
- [GPU Monitoring](#gpu-monitoring)
- [Open-Source Code and Enhancements](#open-source-code-and-enhancements)
- [Project Wiki](#project-wiki)
- [Contributors](#contributors)
- [Contact](#contact)
- [License](#license)
- [Citation](#citation)
- [References](#references)

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
├── experiments/
├── feasibility/
├── report/
├── environment.yml
├── requirements.txt
```

- `code/` – Contains all code-related directories for the project.
  - `data/` – Scripts to download and store datasets used in experiments (e.g., equity prices).
  - `dqn/` – Scripts and implementation for the DQN model.
  - `ppo/` – Scripts and implementation for the PPO model.
  - `test/` – Test scripts to validate environments, agents, and other components.
  - `torchrl_dqn/` – Contains DQN implementation using the TorchRL framework (in progress).
  - `torchrl_ppo/` – Contains PPO implementation using the TorchRL framework (in progress).
- `experiments/` – Contains Excel files and charts documenting the results of experiments, including baseline comparisons, hyperparameter tuning, scaling tests, and transferability.
- `feasibility/` – Documents related to the feasibility study conducted in the initial stage of the project.
- `report/` – Includes project reports and presentations (PDF format).
- `environment.yml` – Conda environment configuration file for setting up project dependencies.
- `requirements.txt` – Lists Python packages required to run the project.


## Hardware Environment
[Cirrus](https://www.epcc.ed.ac.uk/hpc-services/cirrus) is our primary HPC platform for testing our implementations, offering both CPU and GPU nodes to efficiently train and evaluate our DRL models.  

<table width="400">
<tr>
<td><strong>Cirrus</strong></td>
</tr>
<tr>
<td><img src="https://github.com/user-attachments/assets/d06c7ebd-fada-4a58-936b-10f73bc92372" width="350" alt="Cirrus Architecture"></td>
</tr>
<tr>
<td><em>Cirrus at EPCC's Advanced Computing Facility [3]</em></td>
</tr>
</table>


## Software Environment
[PyTorch](https://pytorch.org/) is our primary machine learning framework for implementing DQN and PPO models. In addition to PyTorch, we explored other frameworks or libraries such as TensorFlow and TorchRL during the feasibility and prototyping phases to assess their suitability for the project. The environment is managed through [Conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) to ensure reproducibility across platforms.


## Prerequisites

- Python 3.10+
- PyTorch and supporting libraries (see `requirements.txt` or `environment.yml`)
- Conda or pip package manager
- CUDA-capable GPU recommended
- Cirrus HPC access credentials (for HPC usage)


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

For more details on these enhancements and their impact, refer to the project [report](https://github.com/nabilshadman/deep-reinforcement-learning-automated-trading/blob/main/report/report.pdf).


## Project Wiki

For detailed notes, meeting summaries, experimental observations, and literature references, please refer to the [Project Wiki](https://git.ecdf.ed.ac.uk/msc-22-23/s2134758/-/wikis/home). The wiki includes key information about the project's progress, experimental results, and resources used throughout the development process.


## Contributors

**Researcher:** Nabil Shadman  
**Advisors:** Dr Joseph Lee, Dr Michael Bareford  


## Contact
For questions or issues, feel free to contact Nabil Shadman at nabil.shadman@gmail.com.


## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.  


## Citation  

If you use this work in your research, please cite:  

```bibtex  
@misc{drl-automated-trading-hpc,
  author = {Shadman, Nabil},
  title = {Comparison of Deep Reinforcement Learning Models for Automated Trading on Heterogeneous HPC System},
  year = {2024},
  month = {9},
  publisher = {GitHub},
  url = {https://github.com/nabilshadman/deep-reinforcement-learning-automated-trading},
  note = {Master's Dissertation},
  institution = {The University of Edinburgh}
}
```


## References

```reference
[1] A. Nair et al., "Massively parallel methods for deep reinforcement learning," arXiv.org, 
    https://arxiv.org/abs/1507.04296 (accessed August 24, 2024).
```

```reference
[2] N. Firdous, N. Mohi Ud Din, and A. Assad, "An imbalanced classification approach for 
    establishment of cause-effect relationship between Heart-Failure and Pulmonary Embolism 
    using Deep Reinforcement Learning," Engineering Applications of Artificial Intelligence,
    Sept. 2023.
```

```reference
[3] The University of Edinburgh, "High Performance Computing services," 
    https://www.epcc.ed.ac.uk/high-performance-computing-services (accessed September 16, 2024).
```
