import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

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
import torch.profiler as profiler


# Set up device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True


# Let's use AAPL (Apple), MSI (Motorola), SBUX (Starbucks)
def get_data(data_file):
  # returns a T x 3 list of stock prices
  # each row is a different stock
  # 0 = AAPL
  # 1 = MSI
  # 2 = SBUX
  df = pd.read_csv(data_file)
  return df.values


### The experience replay memory ###
class ReplayBuffer:
  def __init__(self, obs_dim, act_dim, size, batch_size):
    self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
    self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
    self.acts_buf = np.zeros(size, dtype=np.uint8)
    self.rews_buf = np.zeros(size, dtype=np.float32)
    self.done_buf = np.zeros(size, dtype=np.uint8)
    self.ptr, self.size, self.max_size = 0, 0, size
    self.batch_size = batch_size

  def store(self, obs, act, rew, next_obs, done):
    self.obs1_buf[self.ptr] = obs
    self.obs2_buf[self.ptr] = next_obs
    self.acts_buf[self.ptr] = act
    self.rews_buf[self.ptr] = rew
    self.done_buf[self.ptr] = done
    self.ptr = (self.ptr+1) % self.max_size
    self.size = min(self.size+1, self.max_size)

  def sample_batch(self):
    idxs = np.random.randint(0, self.size, size=self.batch_size)
    return dict(s=self.obs1_buf[idxs],
                s2=self.obs2_buf[idxs],
                a=self.acts_buf[idxs],
                r=self.rews_buf[idxs],
                d=self.done_buf[idxs])


def get_scaler(env):
  # return scikit-learn scaler object to scale the states
  # note: you could also populate the replay buffer here

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


def maybe_make_dir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)


class MLP(nn.Module):
  def __init__(self, n_inputs, n_action, n_hidden_layers=2, hidden_dim=32):
    super(MLP, self).__init__()

    M = n_inputs
    self.layers = []
    for _ in range(n_hidden_layers):
      layer = nn.Linear(M, hidden_dim)
      M = hidden_dim
      self.layers.append(layer)
      self.layers.append(nn.ReLU())

    # final layer
    self.layers.append(nn.Linear(M, n_action))
    self.layers = nn.Sequential(*self.layers)

    # move the model to the device
    self.to(device)

  def forward(self, X):
    return self.layers(X)

  def save_weights(self, path):
    torch.save(self.state_dict(), path)

  def load_weights(self, path):
    self.load_state_dict(torch.load(path))


def predict(model, np_states):
  with torch.no_grad():
    # ensure model is on the correct device
    model.to(device)
    # convert numpy array to torch tensor and move it to the device
    inputs = torch.from_numpy(np_states.astype(np.float32)).to(device) 
    output = model(inputs)
    #print("output:", output)
    # transfer predictions back to CPU for NumPy operations
    return output.cpu().numpy()


def train_one_step(model, criterion, optimizer, inputs, targets):
  # convert to tensors
  inputs = torch.from_numpy(inputs.astype(np.float32)).to(device)
  targets = torch.from_numpy(targets.astype(np.float32)).to(device)

  # zero the parameter gradients
  optimizer.zero_grad()

  # forward pass
  outputs = model(inputs)
  loss = criterion(outputs, targets)
        
  # backward and optimize
  loss.backward()
  optimizer.step()


class MultiStockEnv:
  """
  A 3-stock trading environment.
  State: vector of size 7 (n_stock * 2 + 1)
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
    # data
    self.stock_price_history = data
    self.n_step, self.n_stock = self.stock_price_history.shape

    # instance attributes
    self.initial_investment = initial_investment
    self.transaction_cost_rate = transaction_cost_rate
    self.cur_step = None
    self.stock_owned = None
    self.stock_price = None
    self.cash_in_hand = None

    self.action_space = np.arange(3**self.n_stock)

    # action permutations
    # returns a nested list with elements like:
    # [0,0,0]
    # [0,0,1]
    # [0,0,2]
    # [0,1,0]
    # [0,1,1]
    # etc.
    # 0 = sell
    # 1 = hold
    # 2 = buy
    self.action_list = list(map(list, itertools.product([0, 1, 2], repeat=self.n_stock)))

    # calculate size of state
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

    # get current value before performing the action
    prev_val = self._get_val()

    # update price, i.e. go to the next day
    self.cur_step += 1
    self.stock_price = self.stock_price_history[self.cur_step]

    # perform the trade
    self._trade(action)

    # get the new value after taking the action
    cur_val = self._get_val()

    # reward is the increase in porfolio value
    reward = cur_val - prev_val

    # done if we have run out of data
    done = self.cur_step == self.n_step - 1

    # store the current value of the portfolio here
    info = {'cur_val': cur_val}

    # conform to the Gym API
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
    # index the action we want to perform
    # 0 = sell
    # 1 = hold
    # 2 = buy
    # e.g. [2,1,0] means:
    # buy first stock
    # hold second stock
    # sell third stock
    action_vec = self.action_list[action]

    # determine which stocks to buy or sell
    sell_index = [] # stores index of stocks we want to sell
    buy_index = [] # stores index of stocks we want to buy
    for i, a in enumerate(action_vec):
      if a == 0:
        sell_index.append(i)
      elif a == 2:
        buy_index.append(i)

    # sell any stocks we want to sell
    # then buy any stocks we want to buy
    if sell_index:
      # NOTE: to simplify the problem, when we sell, we will sell ALL shares of that stock
      for i in sell_index:
        # self.cash_in_hand += self.stock_price[i] * self.stock_owned[i]
        # self.stock_owned[i] = 0
        # deduct transaction costs when selling
        total_sell_value = self.stock_price[i] * self.stock_owned[i]
        transaction_costs = total_sell_value * self.transaction_cost_rate
        self.cash_in_hand += (total_sell_value - transaction_costs)
        self.stock_owned[i] = 0
    if buy_index:
      # NOTE: when buying, we will loop through each stock we want to buy,
      #       and buy one share at a time until we run out of cash
      can_buy = True
      while can_buy:
        for i in buy_index:
          if self.cash_in_hand > (self.stock_price[i] 
                                  + self.stock_price[i] 
                                  * self.transaction_cost_rate):
            # self.stock_owned[i] += 1 # buy one share
            # self.cash_in_hand -= self.stock_price[i]
            # deduct transaction costs when buying
            self.stock_owned[i] += 1
            self.cash_in_hand -= (self.stock_price[i] + self.stock_price[i] * self.transaction_cost_rate)
          else:
            can_buy = False


class DQNAgent(object):
  def __init__(self, state_size, action_size, batch_size=32, buffer_size=500, gamma=0.99, 
               epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, alpha=0.0003):
    self.state_size = state_size
    self.action_size = action_size
    self.batch_size = batch_size
    self.memory = ReplayBuffer(obs_dim=state_size, act_dim=action_size, 
                               size=buffer_size, batch_size=batch_size)
    self.gamma = gamma  # discount rate
    self.epsilon = epsilon  # exploration rate
    self.epsilon_min = epsilon_min
    self.epsilon_decay = epsilon_decay
    self.model = MLP(state_size, action_size).to(device) # initialize model and move it to device

    # loss and optimizer
    self.criterion = nn.MSELoss()
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=alpha)

  def update_replay_memory(self, state, action, reward, next_state, done):
    self.memory.store(state, action, reward, next_state, done)

  def act(self, state):
    if np.random.rand() <= self.epsilon:
      return np.random.choice(self.action_size)
    act_values = predict(self.model, state)
    return np.argmax(act_values[0])  # returns action

  def replay(self):
    # first check if replay buffer contains enough data
    if self.memory.size < self.batch_size:
      return

    # sample a batch of data from the replay memory
    minibatch = self.memory.sample_batch()
    states = minibatch['s']
    actions = minibatch['a']
    rewards = minibatch['r']
    next_states = minibatch['s2']
    done = minibatch['d']

    # calculate the target: Q(s',a)
    target = rewards + (1 - done) * self.gamma * np.amax(predict(self.model, next_states), axis=1)

    # With the PyTorch API, it is simplest to have the target be the 
    # same shape as the predictions.
    # However, we only need to update the network for the actions
    # which were actually taken.
    # We can accomplish this by setting the target to be equal to
    # the prediction for all values.
    # Then, only change the targets for the actions taken.
    # Q(s,a)
    target_full = predict(self.model, states)
    target_full[np.arange(self.batch_size), actions] = target

    # run one training step
    train_one_step(self.model, self.criterion, self.optimizer, states, target_full)

    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay

  def load(self, name):
    print('... loading models ...')
    self.model.load_weights(name)

  def save(self, name):
    print('... saving models ...')
    self.model.save_weights(name)

  def print_model_summary(self):
    print(self.model, "\n")


def play_one_episode(agent, env, is_train):
  # note: after transforming states are already 1xD
  state = env.reset()
  state = scaler.transform([state])
  done = False

  while not done:
    action = agent.act(state)
    next_state, reward, done, info = env.step(action)
    next_state = scaler.transform([next_state])
    if is_train == 'train':
      agent.update_replay_memory(state, action, reward, next_state, done)
      agent.replay()
    state = next_state

  return info['cur_val']


if __name__ == '__main__':

  # Record start time
  start_time = time.time()  

  # Set the seed for reproducibility
  seed = random.randint(0, 100000) # 42
  set_seeds(seed)  # You can choose any integer as the seed

  # # start pynvml if using cuda
  # if torch.cuda.is_available():
  #   pynvml.nvmlInit()
  
  # Profiler Setup (Start)
  with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
        profile_memory=True,  
        record_shapes=True,
        with_stack=False
    ) as prof:

    # additional info when using cuda
    # if device.type == 'cuda':
    #     print(torch.cuda.get_device_name(0))
    #     print('Memory Usage:')
    #     print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    #     print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
    
    # Configuration for the trading environment and simulation
    data_file = 'equities_daily_close_2018_2023.csv'
    models_folder = 'dqn_trader_models'
    rewards_folder = 'dqn_trader_rewards'
    num_episodes = 10
    initial_investment = 100_000
    transaction_cost_rate = 0.02

    # Hyperparameters for the DQN (Deep Q-Network) agent
    batch_size = 32
    buffer_size = 500
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    alpha = 0.0003

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True,
                        help='either "train" or "test"')
    args = parser.parse_args()

    # determine the mode string and formatting
    mode_str = "Training Mode" if args.mode == "train" else "Testing Mode"
    # print with visual separation
    print("\n", "=" * 20, "\n")  # top separator
    print(f"DQN Trader - {mode_str}")
    print("\n", "=" * 20, "\n")  # bottom separator

    # log device info
    print('Using device:', device, "\n")

    maybe_make_dir(models_folder)
    maybe_make_dir(rewards_folder)

    data = get_data(data_file)
    n_timesteps, n_stocks = data.shape

    n_train = n_timesteps // 2
    train_data = data[:n_train]
    test_data = data[n_train:]

    env = MultiStockEnv(train_data, initial_investment, transaction_cost_rate)
    state_size = env.state_dim
    action_size = len(env.action_space)
    agent = DQNAgent(state_size=state_size, action_size=action_size, 
                    batch_size=batch_size, buffer_size=buffer_size, 
                    gamma=gamma, epsilon=epsilon, epsilon_min=epsilon_min, 
                    epsilon_decay=epsilon_decay, alpha=alpha)

    # print model summary
    agent.print_model_summary()
    scaler = get_scaler(env)

    # Report all hyperparameters
    print("\nHyperparameters:")
    print(f"Mode: {args.mode}")
    print(f"Number of episodes: {num_episodes}")
    print(f"Number of stocks: {env.n_stock}")
    print(f"Initial investment ($): {initial_investment}")
    print(f"Transaction cost rate: {transaction_cost_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Replay buffer size: {buffer_size}")
    print(f"Discount factor (gamma): {gamma}")
    print(f"Initial epsilon (training): {epsilon}")
    print(f"Minimum epsilon (training): {epsilon_min}")
    print(f"Epsilon decay rate (training): {epsilon_decay}")
    print(f"Learning rate (alpha) (training): {alpha}")
    print(f"Random seed: {seed}")
    print("\n")

    # store the final value of the portfolio (end of episode)
    portfolio_value = []

    if args.mode == 'test':
      # then load the previous scaler
      with open(f'{models_folder}/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

      # remake the env with test data
      env = MultiStockEnv(test_data, initial_investment)

      # make sure epsilon is not 1!
      # no need to run multiple episodes if epsilon = 0, it's deterministic
      agent.epsilon = 0.01

      # load trained weights
      agent.load(f'{models_folder}/dqn.ckpt')

    # play the game num_episodes times
    for e in range(num_episodes):
      t0 = datetime.now()
      val = play_one_episode(agent, env, args.mode)
      dt = datetime.now() - t0
      print(f"episode: {e + 1}/{num_episodes}, episode end value (USD): {val:.2f}, "
            f"duration (seconds): {dt.total_seconds()}")
      portfolio_value.append(val) # append episode end portfolio value

    # save the weights when we are done
    if args.mode == 'train':
      # save the DQN
      agent.save(f'{models_folder}/dqn.ckpt')

      # save the scaler
      with open(f'{models_folder}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

  # Profiler Setup (End)
  # Save the trace, naming it based on the mode
  # trace_filename = f"dqn_trace_{args.mode}.json"  
  # prof.export_chrome_trace(trace_filename)  # Change filename
  
  # Print Table of Profiler Results
  # print("\nDetailed Profiler Table:")
  # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

  # Print Total Metrics
  print("\nPyTorch Profiler Metrics:")
  metrics = {
      "CPU Time Total (seconds)": prof.key_averages().total_average().cpu_time_total / 1e6,  
      "CUDA Time Total (seconds)": prof.key_averages().total_average().cuda_time_total / 1e6,
      "CPU Memory Usage (MB)": prof.key_averages().total_average().cpu_memory_usage / 1024**2,
      "CUDA Memory Usage (MB)": prof.key_averages().total_average().cuda_memory_usage / 1024**2
  }
  for key, value in metrics.items():
      print(f"{key}: {value:.3f}")

  # # measure cpu metrics with psutil
  # process = psutil.Process(os.getpid())
  # memory_info = process.memory_info()
  # memory_usage = memory_info.rss / (1024 ** 2)  # convert to mb
  # num_threads = process.num_threads()

  # # print cpu metrics
  # print("\npsutil metrics:")
  # print(f"CPU memory usage (MB): {memory_usage:.3f}")
  # print(f"Number of threads: {num_threads}")

  # # print pynvml metrics (if using cuda) and shutdown pynvml
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

  # save portfolio value for each episode
  np.save(f'{rewards_folder}/{args.mode}.npy', portfolio_value)

  # Print key statistics of the portfolio values
  print("\nPortfolio key statistics:")
  print(f"Median portfolio value (USD): {np.median(portfolio_value):.2f}")
  print(f"Minimum portfolio value (USD): {np.min(portfolio_value):.2f}")
  print(f"Maximum portfolio value (USD): {np.max(portfolio_value):.2f}")

  # After all processing is done, calculate and print total execution time
  end_time = time.time()
  total_time = end_time - start_time
  print(f"\nTotal execution time (seconds): {total_time:.3f}")
