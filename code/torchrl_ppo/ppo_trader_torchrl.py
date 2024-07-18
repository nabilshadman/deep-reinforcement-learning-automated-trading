import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import psutil
import os
import argparse
import pickle
from sklearn.preprocessing import StandardScaler
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value.advantages import GAE
from torchrl.modules import MLP
from tensordict import TensorDict

from multistock_env_torchrl import MultiStockEnvTorch


class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dims, n_actions):
        super(ActorCriticNetwork, self).__init__()
        self.actor = MLP(
            in_features=input_dims,
            depth=2,
            num_cells=256,
            out_features=n_actions,
            activation_class=nn.Tanh,
            activate_last_layer=True
        )
        self.critic = MLP(
            in_features=input_dims,
            depth=2,
            num_cells=256,
            out_features=1,
            activation_class=nn.Tanh,
            activate_last_layer=False
        )
    
    def forward(self, state):
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value


def evaluate(actor_critic, env, device, scaler, num_episodes=3):
    rewards = []
    for _ in range(num_episodes):
        obs = env.reset()
        obs = scaler.transform(obs.unsqueeze(0).cpu().numpy()).squeeze(0)  # Normalize
        obs = torch.tensor(obs, dtype=torch.float32).to(device)
        done = False
        total_reward = 0
        while not done:
            with torch.no_grad():
                action_probs, _ = actor_critic(obs.unsqueeze(0))
                action = torch.argmax(action_probs, dim=-1).cpu().numpy()
            obs, reward, done, _ = env.step(action)
            obs = scaler.transform(obs.unsqueeze(0).cpu().numpy()).squeeze(0)  # Normalize
            obs = torch.tensor(obs, dtype=torch.float32).to(device)
            total_reward += reward
        rewards.append(total_reward)
    return np.mean(rewards)


def maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def play_one_episode(agent, env, replay_buffer, optimizer, loss_module, scaler, mode, config, collector):
    for i, tensordict in enumerate(collector):
        # Normalize observations
        obs = tensordict['observation']
        obs = scaler.transform(obs.cpu().numpy())
        tensordict['observation'] = torch.tensor(obs, dtype=torch.float32).to(config['device'])
        
        # Store experiences in replay buffer
        replay_buffer.add(tensordict)

        if len(replay_buffer) >= config['batch_size']:
            # Sample a batch from replay buffer
            batch = replay_buffer.sample(config['batch_size'])

            # Optimization step
            optimizer.zero_grad()
            loss = loss_module(batch)
            loss.backward()
            optimizer.step()
    
    val = evaluate(agent, env, config['device'], scaler)
    return val


def main():
    # Configuration
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'initial_investment': 20000,
        'transaction_cost_rate': 0.02,
        'num_episodes': 2,
        'batch_size': 32,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_param': 0.2,
        'lr': 1e-3,
        'n_step': 200,
        'models_folder': 'ppo_trader_models',
        'rewards_folder': 'ppo_trader_rewards',
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True,
                        help='either "train" or "test"')
    args = parser.parse_args()

    # Determine the mode string and formatting
    mode_str = "Training Mode" if args.mode == "train" else "Testing Mode"
    print("\n", "=" * 20, "\n")  # Top separator
    print(f"PPO Trader - {mode_str}")
    print("\n", "=" * 20, "\n")  # Bottom separator

    # Log device info
    print('Using device:', config['device'], "\n")

    maybe_make_dir(config['models_folder'])
    maybe_make_dir(config['rewards_folder'])

    # Load data
    data = pd.read_csv('equities_close_prices_daily.csv').values
    n_timesteps, n_stocks = data.shape
    n_train = n_timesteps // 2
    train_data = data[:n_train]
    test_data = data[n_train:]

    env = MultiStockEnvTorch(train_data, config['initial_investment'], config['transaction_cost_rate'])
    test_env = MultiStockEnvTorch(test_data, config['initial_investment'], config['transaction_cost_rate'])

    # Initialize scaler
    scaler = StandardScaler()
    obs = env.reset()['observation']
    scaler.fit(obs.unsqueeze(0).cpu().numpy())

    # Initialize model
    input_dims = env.state_dim
    n_actions = env.action_space
    actor_critic = ActorCriticNetwork(input_dims, n_actions).to(config['device'])
    optimizer = optim.Adam(actor_critic.parameters(), lr=config['lr'])

    if args.mode == 'test':
        with open(f"{config['models_folder']}/scaler.pkl", 'rb') as f:
            scaler = pickle.load(f)
        env = MultiStockEnvTorch(test_data, config['initial_investment'], config['transaction_cost_rate'])
        actor_critic.load_state_dict(torch.load(f"{config['models_folder']}/ppo_actor_critic.pth"))

    # Replay Buffer
    storage = LazyMemmapStorage(config['batch_size'] * config['num_episodes'])
    replay_buffer = TensorDictReplayBuffer(storage=storage, sampler=SamplerWithoutReplacement())

    # GAE and PPO Loss
    gae = GAE(value_network=actor_critic.critic, gamma=config['gamma'], lmbda=config['gae_lambda'])
    ppo_loss = ClipPPOLoss(actor_network=actor_critic.actor, critic_network=actor_critic.critic, advantage=gae, clip_param=config['clip_param'])

    # Data Collector
    collector = SyncDataCollector(
        create_env_fn=lambda: env,
        policy=actor_critic,
        frames_per_batch=config['batch_size'],
        total_frames=config['n_step'],
        device=config['device'],
        storing_device=config['device'],
        max_frames_per_traj=config['n_step'],
    )

    # Train/Test loop
    portfolio_value = []
    for e in range(config['num_episodes']):
        t0 = datetime.now()
        val = play_one_episode(actor_critic, env, replay_buffer, optimizer, ppo_loss, scaler, args.mode, config, collector)
        dt = datetime.now() - t0
        print(f"episode: {e + 1}/{config['num_episodes']}, episode end value: {val:.2f}, duration: {dt}")
        portfolio_value.append(val)

    # Save model
    if args.mode == 'train':
        torch.save(actor_critic.state_dict(), f"{config['models_folder']}/ppo_actor_critic.pth")
        with open(f"{config['models_folder']}/scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)

    # Log performance metrics
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage = memory_info.rss / (1024 ** 2)
    num_threads = process.num_threads()
    
    print("\npsutil Metrics:")
    print(f"CPU Memory Usage: {memory_usage:.3f} MB")
    print(f"Number of Threads: {num_threads}")

    if torch.cuda.is_available():
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        power_usage = pynvml.nvmlDeviceGetPowerUsage(handle)

        print("\nPyNVML Metrics:")
        print(f"GPU Memory Total: {mem_info.total / (1024 ** 2):.2f} MB")
        print(f"GPU Memory Usage: {mem_info.used / (1024 ** 2):.2f} MB")
        print(f"GPU Utilization: {gpu_utilization} %")
        print(f"Power Usage: {power_usage / 1000:.2f} W")

        pynvml.nvmlShutdown()

    np.save(f'{config["rewards_folder"]}/{args.mode}.npy', portfolio_value)


if __name__ == "__main__":
    main()
