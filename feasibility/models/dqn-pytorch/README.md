# Deep Reinforcement Learning Trading Algorithm with Deep Q Learning (DQN) Agent  
**Tech stack:** python, pytorch, numpy, pandas  


## Code 
The code includes an open source [implementation](https://github.com/lazyprogrammer/machine_learning_examples/tree/master/pytorch) of DQN model for algorithmic trading.  

The script has a train/test mode. In both cases, there is a main loop called    
play_one_episode() that is called several times. The loop mainly goes back and  
forth between the agent and the environment. The environment produces states and    
rewards. The agent takes in states and returns actions to perform in the environments.  

During train mode, the agent will store the states, actions, and rewards, and perform  
Q-Learning updates in order to train the Q function approximator.  

The agent uses a [Deep Q Learning](https://arxiv.org/pdf/1312.5602.pdf) (DQN) algorithm.  



## Data  
As supplied, the [dataset](https://git.ecdf.ed.ac.uk/msc-22-23/s2134758/-/blob/main/feasibility/models/dqn-pytorch/aapl_msi_sbux.csv) consists of historical stock prices of AAPL (Apple), MSI (Motorola),  
and SBUX (Starbucks).  


## Load relevant module on Cirrus
To load the latest **pytorch** module, execute:
`module load pytorch`


## How to run the script on frontend node of Cirrus  
To run the script in **training** mode, execute:  
`python rl_trader.py -m train`  

For **testing** mode, execute:  
`python rl_trader.py -m test`  


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
