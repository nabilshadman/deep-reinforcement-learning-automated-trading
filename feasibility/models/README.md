# Feasibility Study (Models)

This folder contains the baseline implementations of the Deep Q-Network (DQN) algorithm using both PyTorch and TensorFlow frameworks. These implementations were used during the feasibility study phase of the project to compare the performance and suitability of each framework for the final model development.

The study builds upon the open-source implementations of the DQN algorithm (developed by [Lazy Programmer](https://github.com/lazyprogrammer/machine_learning_examples)). The author is an experienced machine learning practitioner who promotes experimentation with their implementations. 

## Contents

- **`dqn-pytorch/`**: This folder contains the DQN implementation using the PyTorch framework. It includes the core agent code, environment setup, and utility scripts to train and evaluate the DQN model.

- **`dqn-tensorflow/`**: This folder contains the DQN implementation using the TensorFlow framework. It includes the necessary scripts to train and evaluate the DQN model using TensorFlow and supports testing on both CPU and GPU nodes.

## Purpose

The feasibility study aimed to compare the performance of PyTorch and TensorFlow for implementing the DQN algorithm in an automated equity trading environment. By evaluating these two frameworks, we aimed to understand their respective strengths in terms of speed, resource utilisation, and ease of integration with HPC systems like Cirrus.

## Usage

Each folder contains framework-specific code for the DQN algorithm. Use the appropriate folder depending on the framework you want to evaluate.

### Example (PyTorch):
```bash
cd dqn-pytorch
python rl_trader.py -m train
python rl_trader.py -m test
```

### Example (TensorFlow):
```bash
cd dqn-tensorflow
python rl_trader.py -m train
python rl_trader.py -m test
```

Logs and performance metrics will be generated after running the models, providing insight into the comparative performance of each framework during the feasibility study.  
