"""
PPO Trader Implementation

The PPO agent code is adapted from the original open-source implementation by Phil Tabor 
(https://github.com/philtabor). Phil Tabor's version encourages experimentation 
and provides a strong foundation for PPO-based reinforcement learning programs.

Our implementation extends the original work by integrating it into a multi-asset 
trading environment, enabling the model to handle multiple assets simultaneously. 
Additional enhancements include support for profiling, flexible configuration management, 
and hyperparameter tuning for more efficient experimentation and scalability. For more 
details on the modifications and their impact, please refer to the project report.

Credit: Original PPO agent code by Phil Tabor 
(https://github.com/philtabor/Youtube-Code-Repository)
"""


import random
import yaml
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from datetime import datetime
import time
import itertools
import argparse
import re
import os
import pickle

from sklearn.preprocessing import StandardScaler

# import psutil
# if torch.cuda.is_available():
#   import pynvml


# Set up device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Set random seeds for reproducibility across different runs
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True


# Function to load configuration from a YAML file
def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


# Load asset price data from a CSV file
def get_data(data_file):
  # Returns a T x 3 list of stock prices
  # Each row is a different stock
  df = pd.read_csv(data_file)
  return df.values


# Obtain a scaler object to normalise environment states
def get_scaler(env):
  # Return scikit-learn scaler object to scale the states

  states = []
  for _ in range(env.n_step):
    action = np.random.choice(env.action_space)
    state, reward, done, info = env.step(action)
    states.append(state)
    if done:
      break

  scaler = StandardScaler()
  scaler.fit(states)
  return scaler


# Create a directory if it doesn't exist
def maybe_make_dir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)


# Memory class for storing experiences for PPO training
class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


# Actor Network for PPO (policy network)
class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
            fc1_dims=32, fc2_dims=32, chkpt_dir='ppo_trader_models'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        self.actor = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, n_actions),
                nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        
        return dist

    def save_checkpoint(self):
       torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
       self.load_state_dict(torch.load(self.checkpoint_file))

    # def save_checkpoint(self, path):
    #    torch.save(self.state_dict(), path)

    #def load_checkpoint(self, path):
    #    self.load_state_dict(torch.load(path))


# Critic Network for PPO (value network)
class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=32, fc2_dims=32,
            chkpt_dir='ppo_trader_models'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self.critic = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

    # def save_checkpoint(self, path):
    #     torch.save(self.state_dict(), path)

    # def load_checkpoint(self, path):
    #     self.load_state_dict(torch.load(path))


# Proximal Policy Optimisation (PPO) Agent
class PPOAgent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
            policy_clip=0.2, batch_size=32, n_epochs=4, chkpt_dir='ppo_trader_models'):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(n_actions=n_actions, input_dims=input_dims, 
                                  alpha=alpha, chkpt_dir=chkpt_dir)
        self.critic = CriticNetwork(input_dims=input_dims, alpha=alpha, 
                                    chkpt_dir=chkpt_dir)
        self.memory = PPOMemory(batch_size)
       
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        state = torch.tensor(observation, dtype=torch.float).to(self.actor.device)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, probs, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.actor.device)

            values = torch.tensor(values).to(self.actor.device)
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.actor.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = torch.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = torch.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                #prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()               

    def print_model_summary(self):
       print(self.actor, "\n", "\n", self.critic, "\n")


# Multi-Asset Trading Environment
class MultiStockEnv:
  """
  A multi-asset trading environment.
  State: vector of size 7 (n_stock * 2 + 1) for 3 equities
    - # shares of stock 1 owned
    - # shares of stock 2 owned
    - # shares of stock 3 owned
    - price of stock 1 (using daily close price)
    - price of stock 2
    - price of stock 3
    - cash owned (can be used to purchase more stocks)
  Action: categorical variable with 27 (3^3) possibilities
    - for each stock, you can:
    - 0 = sell
    - 1 = hold
    - 2 = buy
  """
  def __init__(self, data, initial_investment=20000, transaction_cost_rate=0.02):
    # Data
    self.stock_price_history = data
    self.n_step, self.n_stock = self.stock_price_history.shape

    # Instance attributes
    self.initial_investment = initial_investment
    self.transaction_cost_rate = transaction_cost_rate
    self.cur_step = None
    self.stock_owned = None
    self.stock_price = None
    self.cash_in_hand = None

    self.action_space = np.arange(3**self.n_stock)

    # Action permutations
    # Returns a nested list with elements like:
    # [0,0,0]
    # [0,0,1]
    # [0,0,2]
    # [0,1,0]
    # [0,1,1]
    # etc.
    # 0 = sell
    # 1 = hold
    # 2 = buy
    # self.action_list = np.array(list(itertools.product([0, 1, 2], repeat=self.n_stock)))
    self.action_list = list(map(list, itertools.product([0, 1, 2], repeat=self.n_stock)))

    # Calculate size of state
    self.state_dim = self.n_stock * 2 + 1

    self.reset()

  def reset(self):
    self.cur_step = 0
    self.stock_owned = np.zeros(self.n_stock)
    self.stock_price = self.stock_price_history[self.cur_step]
    self.cash_in_hand = self.initial_investment
    return self._get_obs()

  def step(self, action):
    assert action in self.action_space

    # Get current value before performing the action
    prev_val = self._get_val()

    # Update price, i.e. go to the next day
    self.cur_step += 1
    self.stock_price = self.stock_price_history[self.cur_step]

    # Perform the trade
    self._trade(action)

    # Get the new value after taking the action
    cur_val = self._get_val()

    # Reward is the increase in porfolio value
    reward = cur_val - prev_val

    # Done if we have run out of data
    done = self.cur_step == self.n_step - 1

    # Store the current value of the portfolio here
    info = {'cur_val': cur_val}

    # Conform to the Gym API
    return self._get_obs(), reward, done, info

  def _get_obs(self):
    obs = np.empty(self.state_dim)
    obs[:self.n_stock] = self.stock_owned
    obs[self.n_stock:2*self.n_stock] = self.stock_price
    obs[-1] = self.cash_in_hand
    return obs
    
  def _get_val(self):
    return self.stock_owned.dot(self.stock_price) + self.cash_in_hand

  def _trade(self, action):
    # Index the action we want to perform
    # 0 = sell
    # 1 = hold
    # 2 = buy
    # e.g. [2,1,0] means:
    # Buy first stock
    # Hold second stock
    # Sell third stock
    # action_vec = self.action_list[action, :]
    action_vec = self.action_list[action]

    # Determine which stocks to buy or sell
    sell_index = [] # Stores index of stocks we want to sell
    buy_index = [] # Stores index of stocks we want to buy
    for i, a in enumerate(action_vec):
      if a == 0:
        sell_index.append(i)
      elif a == 2:
        buy_index.append(i)

    # Sell any stocks we want to sell, 
    # then buy any stocks we want to buy
    if sell_index:
      # To simplify the problem, when we sell, we will sell ALL shares of that stock
      for i in sell_index:
      #   self.cash_in_hand += self.stock_price[i] * self.stock_owned[i]
      #   self.stock_owned[i] = 0
         total_sell_value = self.stock_price[i] * self.stock_owned[i]
         transaction_costs = total_sell_value * self.transaction_cost_rate
         self.cash_in_hand += (total_sell_value - transaction_costs)
         self.stock_owned[i] = 0
    if buy_index:
      # When buying, we will loop through each stock we want to buy,
      # and buy one share at a time until we run out of cash
      can_buy = True
      while can_buy:
        for i in buy_index:
          if self.cash_in_hand > (self.stock_price[i] 
                                  + self.stock_price[i] 
                                  * self.transaction_cost_rate):
            # self.stock_owned[i] += 1 # buy one share
            # self.cash_in_hand -= self.stock_price[i]
            self.stock_owned[i] += 1
            self.cash_in_hand -= (self.stock_price[i] + self.stock_price[i] * self.transaction_cost_rate)
          else:
            can_buy = False


# Play one episode of the trading environment
def play_one_episode(agent, env, is_train, n_steps, N, learn_iters):
  # After transforming states are already 1xD
  state = env.reset()
  state = scaler.transform([state])
  done = False
  score = 0

  while not done:
    action, prob, val = agent.choose_action(state)
    next_state, reward, done, info = env.step(action)
    next_state = scaler.transform([next_state])
    n_steps += 1
    score += reward
    if is_train == 'train':
      agent.remember(state, action, prob, val, reward, done)
      if n_steps % N == 0:
        agent.learn()
        learn_iters += 1
    state = next_state

  return info['cur_val']


if __name__ == '__main__':

  # Record start time
  start_time = time.time()  

  # Set the seed for reproducibility
  seed = random.randint(0, 100000)
  set_seeds(seed)  # You can choose any integer as the seed

  # # Start pynvml if using cuda
  # if torch.cuda.is_available():
  #   pynvml.nvmlInit()

  # Additional info when using cuda
  # if device.type == 'cuda':
  #     print(torch.cuda.get_device_name(0))
  #     print('Memory Usage:')
  #     print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
  #     print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

  # Adding command-line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('-m', '--mode', type=str, required=True,
                      help='either "train" or "test"')
  parser.add_argument('-c', '--config', type=str, required=True,
                        help='Path to the configuration file (YAML)')
  args = parser.parse_args()

  # Load configuration
  config = load_config(args.config)

  # Configuration for the trading environment and simulation
  data_file = config['data_file']
  models_folder = config['models_folder']
  rewards_folder = config['rewards_folder']
  num_episodes = config['num_episodes']
  initial_investment = config['initial_investment']
  transaction_cost_rate = config['transaction_cost_rate']
  
  # Hyperparameters for the PPO (Proximal Policy Optimisation) agent
  N = config['N'] # Number of steps between each learning update
  gamma = config['gamma'] # Discount factor
  alpha = config['alpha'] # Learning rate
  gae_lambda = config['gae_lambda'] # GAE lambda parameter
  policy_clip = config['policy_clip'] # Clipping parameter
  batch_size = config['batch_size'] # Batch size for training
  n_epochs = config['n_epochs'] # Number of epochs to train on each batch of data

  # Determine the mode string and formatting
  mode_str = "Training Mode" if args.mode == "train" else "Testing Mode"
  # Print with visual separation
  print("\n", "=" * 20, "\n")  # Top separator
  print(f"PPO Trader - {mode_str}")
  print("\n", "=" * 20, "\n")  # Bottom separator

  # Log device info
  print('Using device:', device, "\n")

  maybe_make_dir(models_folder)
  maybe_make_dir(rewards_folder)

  data = get_data(data_file)
  n_timesteps, n_stocks = data.shape

  n_train = n_timesteps // 2
  train_data = data[:n_train]
  test_data = data[n_train:]

  env = MultiStockEnv(train_data, initial_investment, transaction_cost_rate)
  
  action_size = len(env.action_space)
  state_size = env.state_dim
  input_dims = torch.tensor([env.state_dim], dtype=torch.int)
  
  agent = PPOAgent(
     n_actions=action_size, input_dims=input_dims, gamma=gamma,
     alpha=alpha, gae_lambda=gae_lambda, policy_clip=policy_clip,
     batch_size=batch_size, n_epochs=n_epochs, chkpt_dir=models_folder
     )

  # Print model summary
  agent.print_model_summary()
  scaler = get_scaler(env)

  # Report data filename
  print(f"Data file: {data_file}")

  # Report all hyperparameters
  print("\nHyperparameters:")
  print(f"Mode: {args.mode}")
  print(f"Number of episodes: {num_episodes}")
  print(f"Number of stocks: {env.n_stock}")
  print(f"Initial investment ($): {initial_investment}")
  print(f"Transaction cost rate: {transaction_cost_rate}")
  print(f"Steps between learning updates (N) (training): {N}")
  print(f"Discount factor (gamma): {gamma}")
  print(f"Learning rate (alpha) (training): {alpha}")
  print(f"GAE lambda: {gae_lambda}")
  print(f"Policy clip (training): {policy_clip}")
  print(f"Batch size (training): {batch_size}")
  print(f"Number of epochs (training): {n_epochs}")
  print(f"Random seed: {seed}")
  print("\n")

  # Store the final value of the portfolio (end of episode)
  portfolio_value = []

  if args.mode == 'test':
    # Then load the previous scaler
    with open(f'{models_folder}/scaler.pkl', 'rb') as f:
      scaler = pickle.load(f)

    # Remake the env with test data
    env = MultiStockEnv(test_data, initial_investment)

    # Make sure epsilon is not 1!
    # No need to run multiple episodes if epsilon = 0, it's deterministic
    # agent.epsilon = 0.01

    # Load trained weights
    agent.load_models()

  # Set counters for iterations and steps
  learn_iters = 0
  n_steps = 0

  # Play the game num_episodes times
  for e in range(num_episodes):
    t0 = datetime.now()
    val = play_one_episode(agent, env, args.mode, n_steps, N, learn_iters)
    dt = datetime.now() - t0
    print(f"episode: {e + 1}/{num_episodes}, episode end value (USD): {val:.2f}, "
            f"duration (seconds): {dt.total_seconds()}")
    portfolio_value.append(val) # append episode end portfolio value

  # Save the weights when we are done
  if args.mode == 'train':
    # Save the PPO
    agent.save_models()

    # Save the scaler
    with open(f'{models_folder}/scaler.pkl', 'wb') as f:
      pickle.dump(scaler, f)

  # # Measure cpu metrics with psutil
  # process = psutil.Process(os.getpid())
  # memory_info = process.memory_info()
  # memory_usage = memory_info.rss / (1024 ** 2)  # convert to mb
  # num_threads = process.num_threads()

  # # Print cpu metrics
  # print("\npsutil metrics:")
  # print(f"CPU memory usage (MB): {memory_usage:.3f}")
  # print(f"Number of threads: {num_threads}")

  # # Print pynvml metrics (if using cuda) and shutdown pynvml
  # if torch.cuda.is_available():

  #   print("\nPyNVML metrics:")
  #   handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # assuming single gpu
  #   mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle) # memory information
  #   gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu # gpu utilisation
  #   power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) # power usage

  #   print(f"GPU memory total (MB): {mem_info.total / (1024 ** 2):.2f}")
  #   print(f"GPU memory usage (MB): {mem_info.used / (1024 ** 2):.2f}")
  #   print(f"GPU utilisation (%): {gpu_utilization}")
  #   print(f"Power usage (W): {power_usage / 1000:.2f}")

  #   pynvml.nvmlShutdown()  # shutdown pynvml after use

  # Save portfolio value for each episode
  np.save(f'{rewards_folder}/{args.mode}.npy', portfolio_value)

  # Print key statistics of the portfolio values
  print("\nPortfolio key statistics:")
  print(f"Median portfolio value (USD): {np.median(portfolio_value):.2f}")
  print(f"Minimum portfolio value (USD): {np.min(portfolio_value):.2f}")
  print(f"Maximum portfolio value (USD): {np.max(portfolio_value):.2f}")

  # Calculate and print the median portfolio value of the last 30 episodes
  if len(portfolio_value) >= 30:
      last_30_portfolio_values = portfolio_value[-30:]
      median_last_30 = np.median(last_30_portfolio_values)
      print(f"Median portfolio value of last 30 episodes (USD): {median_last_30:.2f}")
  else:
      print("Not enough episodes to calculate median portfolio value of last 30 episodes.")

  # After all processing is done, calculate and print total execution time
  end_time = time.time()
  total_time = end_time - start_time
  print(f"\nTotal execution time (seconds): {total_time:.3f}")
