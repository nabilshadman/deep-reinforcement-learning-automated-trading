# Deep Reinforcement Learning Trading Algorithm with Proximal Policy Optimisation (PPO) Agent  
**Tech stack:** Python, PyTorch, NumPy, pandas  


## Code  
The includes an implementation of **PPO** model for algorithmic trading.  


## Data  
As supplied, the dataset consists of historical **stock** prices of AAPL (Apple), MSI (Motorola),  
and SBUX (Starbucks).  


## Load relevant module on Cirrus
To load the latest **pytorch** module, execute:  
`module load pytorch`


## How to run the script on frontend node of Cirrus  
To run the script in **training** mode, execute:   
`python ppo_trader.py -m train`  

For **testing** mode, execute:   
`python ppo_trader.py -m test`  


## How to run the script on backend node of Cirrus  
To run on a **CPU** node, execute:  
`sbatch dqn-pytorch-cpu.slurm`

To run on a **GPU** node, execute:  
`sbatch dqn-pytorch-gpu.slurm`


## To plot results  
A script is included to plot portfolio value against number of episodes.  

To plot **training** results:  
`python plot_rl_rewards.py -m train`

To plot **testing** results:  
`python plot_rl_rewards.py -m test`
