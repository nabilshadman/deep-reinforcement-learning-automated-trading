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


import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', type=str, required=True,
                    help='either "train" or "test"')
args = parser.parse_args()

a = np.load(f'rl_trader_rewards/{args.mode}.npy')

print(f"average reward: {a.mean():.2f}, min: {a.min():.2f}, max: {a.max():.2f}")

if args.mode == 'train':
  # show the training progress
  plt.plot(a)
else:
  # test - show a histogram of rewards
  plt.hist(a, bins=20)

plt.title(args.mode)
plt.show()