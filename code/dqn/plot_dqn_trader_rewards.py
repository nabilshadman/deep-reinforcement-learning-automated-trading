"""
DQN Trader Implementation

This code is adapted from the original open-source implementation by Lazy Programmer 
(https://github.com/lazyprogrammer). Lazy Programmer's version provides the foundational 
DQN structure, which encourages experimentation and customisation.

Our implementation extends the original work by adding support for GPU acceleration, 
an enhanced trading environment, flexible configuration management, 
hyperparameter tuning, and profiling capabilities for high-performance computing 
(HPC) systems. For more details on the modifications and their impact, please 
refer to the project report.

Credit: Original DQN code by Lazy Programmer 
(https://github.com/lazyprogrammer/machine_learning_examples)
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