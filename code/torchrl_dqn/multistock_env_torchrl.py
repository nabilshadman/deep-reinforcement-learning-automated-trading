import numpy as np
import pandas as pd
import torch
from torchrl.envs import EnvBase
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec, DiscreteTensorSpec, BoundedTensorSpec
from tensordict import TensorDict


# def get_data():
#     # This function loads stock prices from a CSV file and returns it as a NumPy array
#     df = pd.read_csv('equities_close_prices_daily.csv')
#     return df.values


class MultiStockEnvTorch(EnvBase):
    def __init__(self, data, initial_investment=20000, transaction_cost_rate=0.02):
        super().__init__()
        self.stock_price_history = data
        self.n_step, self.n_stock = self.stock_price_history.shape

        self.initial_investment = initial_investment
        self.transaction_cost_rate = transaction_cost_rate
        self.cur_step = 0
        self.stock_owned = np.zeros(self.n_stock)
        self.stock_price = self.stock_price_history[self.cur_step]
        self.cash_in_hand = self.initial_investment

        self.action_space = 3 ** self.n_stock
        self.state_dim = self.n_stock * 2 + 1

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.set_seed(42)

    def _get_obs(self):
        obs = np.empty(self.state_dim)
        obs[:self.n_stock] = self.stock_owned
        obs[self.n_stock:2 * self.n_stock] = self.stock_price
        obs[-1] = self.cash_in_hand
        return torch.tensor(obs, dtype=torch.float32, device=self.device)

    def _reset(self, tensordict=None):
        self.cur_step = 0
        self.stock_owned = np.zeros(self.n_stock)
        self.stock_price = self.stock_price_history[self.cur_step]
        self.cash_in_hand = self.initial_investment
        obs = self._get_obs()
        return TensorDict({'observation': obs, 'done': torch.tensor(False, device=self.device)}, batch_size=[])

    def _step(self, tensordict):
        action = tensordict.get('action')
        prev_val = self._get_val()
        self.cur_step += 1
        self.stock_price = self.stock_price_history[self.cur_step]
        self._trade(action.item())
        cur_val = self._get_val()
        reward = cur_val - prev_val
        done = self.cur_step == self.n_step - 1
        obs = self._get_obs()
        tensordict.update({
            'observation': obs,
            'reward': torch.tensor(reward, dtype=torch.float32, device=self.device),
            'done': torch.tensor(done, device=self.device),
            'cur_val': torch.tensor(cur_val, dtype=torch.float32, device=self.device)
        })
        return tensordict

    def _get_val(self):
        return self.stock_owned.dot(self.stock_price) + self.cash_in_hand

    def _trade(self, action):
        action_vec = [int(x) for x in np.base_repr(action, base=3).zfill(self.n_stock)]
        sell_index = [i for i, a in enumerate(action_vec) if a == 0]
        buy_index = [i for i, a in enumerate(action_vec) if a == 2]
        if sell_index:
            for i in sell_index:
                total_sell_value = self.stock_price[i] * self.stock_owned[i]
                transaction_costs = total_sell_value * self.transaction_cost_rate
                self.cash_in_hand += (total_sell_value - transaction_costs)
                self.stock_owned[i] = 0
        if buy_index:
            can_buy = True
            while can_buy:
                for i in buy_index:
                    if self.cash_in_hand > (self.stock_price[i] + self.stock_price[i] * self.transaction_cost_rate):
                        self.stock_owned[i] += 1
                        self.cash_in_hand -= (self.stock_price[i] + self.stock_price[i] * self.transaction_cost_rate)
                    else:
                        can_buy = False

    def _set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)

    def spec(self):
        observation_spec = CompositeSpec(
            observation=UnboundedContinuousTensorSpec(shape=(self.state_dim,)),
        )
        action_spec = CompositeSpec(
            action=DiscreteTensorSpec(self.action_space)
        )
        reward_spec = CompositeSpec(
            reward=UnboundedContinuousTensorSpec(shape=())
        )
        done_spec = CompositeSpec(
            done=BoundedTensorSpec(0, 1, shape=(), dtype=torch.bool)
        )
        return CompositeSpec(
            observation=observation_spec,
            action=action_spec,
            reward=reward_spec,
            done=done_spec
        )


# if __name__ == '__main__':
#     data = get_data()
#     env = MultiStockEnvTorch(data[:data.shape[0]//2])  # Create the environment
#     env.set_seed(42)  # Set the seed to 42 for reproducibility

#     # Print the device being used
#     print(f"Using device: {env.device}")

#     # Reset and get initial state
#     tensordict = env.reset()
#     print("Initial Observation:", tensordict.get('observation'))

#     # Take a random action
#     # action = torch.randint(0, env.action_space, (1,), device=env.device)
#     action = torch.tensor([26], device=env.device)
#     tensordict = TensorDict({'action': action}, batch_size=[])
#     tensordict = env.step(tensordict)

#     print("Next Observation:", tensordict.get('observation'))
#     print("Reward:", tensordict.get('reward'))
#     print("Done:", tensordict.get('done'))
#     print("Info (Current Value):", tensordict.get('cur_val'))
