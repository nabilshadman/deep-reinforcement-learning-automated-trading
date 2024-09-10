# Deep Q-Network (DQN)

This folder contains the implementation of the Deep Q-Network (DQN) algorithm, supporting both CPU and GPU environments. It includes scripts for training, testing, profiling, and analysing the DQN agent's performance in an automated trading environment.

## Contents

- **`dqn_trader.py`**: The main script for running the DQN trader, which utilises the DQN algorithm to make trading decisions based on historical market data.
  
- **`dqn_trader_dlprof.py`**: A profiling script designed to gather performance data using NVIDIA DLProf on GPU nodes. It provides insights into CUDA performance during model training.
  
- **`dqn_trader_prof.py`**: Another profiling script (using PyTorch Profiler) that gathers performance data for CPU and GPU nodes to evaluate resource usage during training and testing.
  
- **`dqn_trader_tensorboard.py`**: Script for tracking training progress and performance metrics using TensorBoard for visualisation.
  
- **`config.yaml`**: Configuration file containing hyperparameters and experiment setups for running the DQN trader.
  
- **`dqn_cpu.slurm`**, **`dqn_gpu.slurm`**: SLURM scripts for running the DQN trader on CPU and GPU nodes on the Cirrus HPC system.
  
- **`dqn_prof_cpu.slurm`**, **`dqn_prof_gpu.slurm`**: SLURM scripts for profiling the DQN trader on CPU and GPU nodes using different profilers.
  
- **`equities_daily_close_10_tickers_2018_2023.csv`**, **`equities_daily_close_2018_2023.csv`**: Datasets containing daily close prices for selected equities from 2018 to 2023 used for training and testing the DQN model.
  
- **`plot_dqn_trader_rewards.py`**: Script for plotting the rewards earned by the DQN trader during training and testing phases.

- **`scaling_stocks/`**: Contains experiments related to scaling the number of equities in the DQN trader's portfolio to test performance with larger datasets.

- **`hyperparameter_tuning/`**: Contains the results and configurations from hyperparameter tuning experiments, optimising DQN performance.

- **`transferability_analysis/`**: Scripts and results for analysing the transferability of the DQN model to different datasets and asset classes.

- **`logs/`**: Directory containing logs generated during model training and testing, useful for debugging and performance analysis.

- **`tensorboard_runs/`**: Directory storing TensorBoard logs for visualising training progress.

- **`clean.sh`**: Script to clean temporary files and logs created during the experiments.

## Usage (on Cirrus)

To run the DQN trader on CPU node of Cirrus:
```bash
sbatch dqn_cpu.slurm
```

To run the DQN trader on GPU node of Cirrus:
```bash
sbatch dqn_gpu.slurm
```  

To profile the DQN trader:  
```bash
sbatch dqn_prof_cpu.slurm  # For CPU profiling
sbatch dqn_prof_gpu.slurm  # For GPU profiling
```

Make sure to adjust `config.yaml` for specific hyperparameters and experiment setups before running the scripts.  

## Usage (on local machine)

To run the DQN trader on your local machine without SLURM, you can use the following commands. Ensure that your environment is set up correctly and that all dependencies listed in `requirements.txt` or `environment.yml` are installed.

**Training** the DQN Trader:
```bash
python dqn_trader.py -m train -c config.yaml
```

This command will start training the DQN trader on the provided dataset. You can adjust the configurations in the `config.yaml` file before running to fine-tune the hyperparameters.

**Testing** the DQN Trader:  
```bash
python dqn_trader.py -m test -c config.yaml
```
This command will run the DQN trader in test mode, using a trained model to make predictions based on the historical data.

## Data  

The provided CSV files contain daily closing prices of selected equities from 2018 to 2023. These files are used as the input dataset for training and testing the DQN model. Modify the data files or include your own datasets in similar formats if needed.  

## Code Base
This DQN implementation is adapted from the open-source work of [Lazy Programmer](https://github.com/lazyprogrammer/machine_learning_examples) (LP), an experienced Machine Learning practitioner. LP encourages experimentation with his implementation. His original DQN code serves as the foundation for our implementation, with additional enhancements made to support GPU training, an improved environment, hyperparameter tuning and profiling capabilities for HPC systems. For more details on these enhancements and their impact, refer to the project report.

## Single-File Design
The program follows a single-file design, where the core DQN logic, environment, and agent setup are all contained within `dqn_trader.py`. This design pattern is commonly found in several open-source RL libraries (e.g., [CleanRL](https://github.com/vwxyzjn/cleanrl)) and research codes.

This structure allows for simplicity and easy understanding of the entire workflow in one place, making it easier to modify, debug, and extend the model. However, configuration and data management are externalised to improve flexibility and scalability.
