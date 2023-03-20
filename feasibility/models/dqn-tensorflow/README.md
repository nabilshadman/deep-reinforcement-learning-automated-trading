# Deep Reinforcement Learning Trading Algorithm with Deep Q Learning (DQN) Agent  
Tech stack: python, tensorflow, numpy, pandas  

The script has a train/test mode. In both cases, there is a main loop called    
play_one_episode() that is called several times. The loop mainly goes back and  
forth between the agent and the environment. The environment produces states and    
rewards. The agent takes in states and returns actions to perform in the environments.  

During train mode, the agent will store the states, actions, and rewards, and perform  
Q-Learning updates in order to train the Q function approximator.  

The agent uses a [Deep Q Learning](https://www.geeksforgeeks.org/deep-q-learning/) (DQN) algorithm.  


## Data  
As supplied, the dataset consists of historical stock prices of AAPL (Apple), MSI (Motorola),  
and SBUX (Starbucks).  


## How to run the trading script  
To run the script in training mode, execute:  
python rl_trader.py -m train  

For testing mode, execute:  
python rl_trader.py -m test  


## To plot results  
A script is included to plot portfolio value against number of episodes.  

To plot training results:  
python plot_rl_rewards.py -m train  

To plot testing results:  
python plot_rl_rewards.py -m test    
