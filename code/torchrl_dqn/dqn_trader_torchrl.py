import argparse
import os
import pickle
import psutil
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torchrl.data import LazyTensorStorage, DiscreteTensorSpec, ReplayBuffer
from torchrl.modules import EGreedyModule, MLP, QValueActor
from torchrl.objectives import DQNLoss, ValueEstimators
from multistock_env_torchrl import MultiStockEnvTorch
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict import TensorDict


# Load stock data
stock_data = pd.read_csv('equities_close_prices_daily.csv').values


# Configuration
config = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'initial_investment': 20000,
    'transaction_cost_rate': 0.02,
    'num_episodes': 2,
    'batch_size': 32,
    'lr': 1e-3,
    'gamma': 0.99,
}


# Create environment
def make_env(data, initial_investment, transaction_cost_rate):
    env = MultiStockEnvTorch(data, initial_investment, transaction_cost_rate)
    return env


# Model creation function
def make_dqn_model(state_dim, action_dim):
    model = MLP(in_features=state_dim, out_features=action_dim, depth=2, num_cells=128)
    return TensorDictModule(model, in_keys=['state'], out_keys=['action_value'])


# Function to save model and scaler
def save_model(agent, scaler, models_folder):
    torch.save(agent.state_dict(), f'{models_folder}/dqn.ckpt')
    with open(f'{models_folder}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)


# Function to load model and scaler
def load_model(agent, models_folder):
    agent.load_state_dict(torch.load(f'{models_folder}/dqn.ckpt'))
    with open(f'{models_folder}/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return scaler


# Function to play one episode
def play_one_episode(agent, policy, env, replay_buffer, optimizer, loss_module, mode):
    state = env.reset()
    
    state_tensor = state['observation'].unsqueeze(0).to(config['device'])

    done = False
    total_reward = 0
    step = 0

    while not done:
        # action_tensordict = policy(TensorDict({"state": state_tensor}, batch_size=[1]))  # Ensure batch size is [1]
        action_tensordict = policy(TensorDict({"state": state_tensor}))
        action = action_tensordict['action']
        
        # action_tensordict = TensorDict({"action": action}, batch_size=[1])  # Ensure batch size is [1]
        action_tensordict = TensorDict({"action": action})  # Ensure batch size is [1]
        next_state = env.step(action_tensordict)  # Pass the TensorDict to env.step()
        
        next_state_tensor = next_state['observation'].unsqueeze(0).to(config['device'])
        
        reward = next_state["reward"]
        done = next_state["done"]
        
        total_reward += reward.item()

        replay_buffer.extend({
            "state": state_tensor,
            "action": action,
            "reward": reward,
            "next_state": next_state_tensor,
            "done": done,
        })

        if mode == "train":
            batch = replay_buffer.sample(batch_size=config['batch_size'])
            loss = loss_module(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        state_tensor = next_state_tensor
        step += 1
    
    return total_reward


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True, help='either "train" or "test"')
    args = parser.parse_args()

    # Print mode
    mode_str = "Training Mode" if args.mode == "train" else "Testing Mode"
    print("\n", "=" * 20, "\n")  # top separator
    print(f"DQN Trader - {mode_str}")
    print("\n", "=" * 20, "\n")  # bottom separator

    # Setup device
    device = config['device']
    print('Using device:', device, "\n")

    # Create folders
    models_folder = 'dqn_trader_models'
    rewards_folder = 'dqn_trader_rewards'
    os.makedirs(models_folder, exist_ok=True)
    os.makedirs(rewards_folder, exist_ok=True)

    # Split data
    n_timesteps, n_stocks = stock_data.shape
    n_train = n_timesteps // 2
    train_data = stock_data[:n_train]
    test_data = stock_data[n_train:]

    # Initialize environment
    env = make_env(train_data, config['initial_investment'], config['transaction_cost_rate'])
    state_size = env.state_dim
    action_size = env.action_space

    # Define the action spec for DiscreteTensorSpec
    action_spec = DiscreteTensorSpec(n=action_size)

    # Create the agent
    agent_module = make_dqn_model(state_size, action_size).to(device)
    
    # Create the QValueActor
    agent = QValueActor(
        module=agent_module,
        spec=action_spec,
        action_value_key="action_value"
    )

    # Create epsilon-greedy policy
    policy = TensorDictSequential(
        agent,
        EGreedyModule(spec=action_spec, eps_init=1.0, eps_end=0.1, 
                      annealing_num_steps=1000, action_key='action')
    )

    # Load scaler
    scaler = None

    if args.mode == 'test':
        scaler = load_model(agent, models_folder)
        env = make_env(test_data, config['initial_investment'], config['transaction_cost_rate'])
        policy.module[1].eps_init = 0.1

    # Replay buffer
    replay_buffer = ReplayBuffer(storage=LazyTensorStorage(config['batch_size']))

    # Optimizer
    optimizer = optim.Adam(agent.parameters(), lr=config['lr'])

    # Loss
    loss_module = DQNLoss(agent, action_space=action_spec)
    loss_module.make_value_estimator(ValueEstimators.TD0, gamma=config['gamma'])

    # Train/Test loop
    portfolio_value = []

    for e in range(config['num_episodes']):
        t0 = datetime.now()
        val = play_one_episode(agent, policy, env, replay_buffer, optimizer, loss_module, args.mode)
        dt = datetime.now() - t0
        print(f"episode: {e + 1}/{config['num_episodes']}, episode end value: {val:.2f}, duration: {dt}")
        portfolio_value.append(val)

    # Save the model
    if args.mode == 'train':
        save_model(agent, scaler, models_folder)

    # Measure CPU metrics with psutil
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage = memory_info.rss / (1024 ** 2)  # Convert to MB
    num_threads = process.num_threads()

    # Print CPU metrics
    print("\npsutil Metrics:")
    print(f"CPU Memory Usage: {memory_usage:.3f} MB")
    print(f"Number of Threads: {num_threads}")

    # Print PyNVML metrics (if using cuda) and shutdown pynvml
    if torch.cuda.is_available():
        import pynvml
        pynvml.nvmlInit()
        print("\nPyNVML Metrics:")
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming single GPU
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)  # Memory information
        gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu  # GPU utilization
        power_usage = pynvml.nvmlDeviceGetPowerUsage(handle)  # Power usage

        print(f"GPU Memory Total: {mem_info.total / (1024 ** 2):.2f} MB")
        print(f"GPU Memory Usage: {mem_info.used / (1024 ** 2):.2f} MB")
        print(f"GPU Utilization: {gpu_utilization} %")
        print(f"Power Usage: {power_usage / 1000:.2f} W")

        pynvml.nvmlShutdown()  # Shutdown pynvml after use

    # Save portfolio value for each episode
    np.save(f'{rewards_folder}/{args.mode}.npy', portfolio_value)

if __name__ == "__main__":
    main()
