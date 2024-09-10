# PPO Folder

This folder contains the implementation of the Proximal Policy Optimisation (PPO) algorithm, supporting both CPU and GPU environments. It includes scripts for training, testing, profiling, and analysing the PPO agent’s performance in an automated trading environment.

## Contents

- **`ppo_trader.py`**: The main script for running the PPO trader, which applies the PPO algorithm to optimise trading strategies based on historical market data.

- **`ppo_trader_dlprof.py`**: A profiling script designed to capture performance metrics using NVIDIA DLProf on GPU nodes, providing insights into CUDA and model performance.

- **`ppo_trader_prof.py`**: Profiling script for performance monitoring (using PyTorch Profiler) on both CPU and GPU nodes to evaluate resource usage during model training.

- **`ppo_trader_tensorboard.py`**: Script for logging and visualising training performance using TensorBoard.

- **`config.yaml`**: Configuration file containing hyperparameters and experimental setups for the PPO trader.

- **`ppo_cpu.slurm`**, **`ppo_gpu.slurm`**: SLURM scripts for running the PPO trader on CPU and GPU nodes on the Cirrus HPC system.

- **`ppo_prof_cpu.slurm`**, **`ppo_prof_gpu.slurm`**: SLURM scripts for profiling the PPO trader on CPU and GPU nodes.

- **`equities_daily_close_10_tickers_2018_2023.csv`**, **`equities_daily_close_2018_2023.csv`**: Datasets containing daily close prices for selected equities from 2018 to 2023 used for training the PPO model.

- **`plot_ppo_trader_rewards.py`**: Script to plot the rewards earned by the PPO trader during training and testing.

- **`scaling_stocks/`**: Contains experiments related to scaling the number of stocks in the PPO trader’s portfolio to test performance with larger datasets.

- **`hyperparameter_tuning/`**: Contains the results and configurations from hyperparameter tuning experiments, aiming to optimise PPO performance.

- **`transferability_analysis/`**: Scripts and results for analysing the transferability of the PPO model across different datasets and asset classes.

- **`logs/`**: Directory containing logs generated during training and testing, useful for debugging and performance analysis.

- **`tensorboard_runs/`**: Directory containing TensorBoard logs for visualising training progress and performance metrics.

- **`clean.sh`**: Script for cleaning temporary files and logs created during experiments.


## Usage (on Cirrus)

To run the PPO trader on CPU node of Cirrus:
```bash
sbatch ppo_cpu.slurm
```

To run the PPO trader on GPU node of Cirrus:
```bash
sbatch ppo_gpu.slurm
```  

To profile the PPO trader:  
```bash
sbatch ppo_prof_cpu.slurm  # For CPU profiling
sbatch ppo_prof_gpu.slurm  # For GPU profiling
```

Make sure to adjust `config.yaml` for specific hyperparameters and experiment setups before running the scripts.  

## Usage (on local machine)

To run the PPO trader on your local machine without SLURM, you can use the following commands. Ensure that your environment is set up correctly and that all dependencies listed in `requirements.txt` or `environment.yml` are installed.

**Training** the PPO Trader:
```bash
python ppo_trader.py -m train -c config.yaml
```

This command will start training the PPO trader on the provided dataset. You can adjust the configurations in the `config.yaml` file before running to fine-tune the hyperparameters.

**Testing** the PPO Trader:  
```bash
python ppo_trader.py -m test -c config.yaml
```
This command will run the PPO trader in test mode, using a trained model to make predictions based on the historical data.

## Data  

The provided CSV files contain daily closing prices of selected equities from 2018 to 2023. These files are used as the input dataset for training and testing the PPO model. Modify the data files or include your own datasets in similar formats if needed.  

## Code Base
The PPO implementation in this project is adapted from the open-source work of [Phil Tabor](https://github.com/philtabor/Youtube-Code-Repository), an experienced Machine Learning practitioner. Tabor encourages experimentation with his implementation. The main enhancement includes integrating the code into a multi-asset trading environment, enabling the model to handle multiple assets simultaneously. Additionally, we’ve added support for profiling capabilities, hyperparameter tuning, and flexible configuration management for more efficient experimentation and scalability. For more details on these enhancements and their impact, refer to the project report.

## Single-File Design
The program follows a single-file design, where the core PPO logic, environment, and agent setup are all contained within `ppo_trader.py`. This design pattern is commonly found in several open-source RL libraries (e.g., [CleanRL](https://github.com/vwxyzjn/cleanrl)) and research codes.

This structure allows for simplicity and easy understanding of the entire workflow in one place, making it easier to modify, debug, and extend the model. However, configuration and data management are externalised to improve flexibility and scalability.
