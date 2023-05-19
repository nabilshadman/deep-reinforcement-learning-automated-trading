# Comparison of Deep Reinforcement Learning Models for Automated Trading on Heterogeneous HPC System  

**Researcher's Name:** Nabil Shadman  
**Supervisor's Name:** Dr Joseph Lee  
**Target HPC System:** [Cirrus](https://www.epcc.ed.ac.uk/hpc-services/cirrus)  

This repository currently contains the proof-of-concept (POC) implementation of a broader study to develop, paralellise, and test DRL models on our target heterogeneous HPC system (i.e., Cirrus).  


## Background
**Deep Reinforcement Learning (DRL)** and **High-Performance Computing (HPC)** technologies are applied to solve problems in areas such as video games, large language models, autonomous driving, and automated trading. DRL models, especially when applied in a data-intensive field such as finance, require a vast amount of computational power.   

It is essential for traders to experiment with different models to discover profitable strategies. We believe utilising the computational power of HPC systems can make the process time efficient. In this study, we discuss the feasibility to build DRL models for automated trading on heterogeneous HPC systems (i.e., with a combination of CPUs and GPUs in our case).  

**Algorithmic trading** as the use of computers to execute and monitor trades based on predefined logic. Reinforcement learning (RL) is a machine learning approach where intelligent agents learn to take actions in an environment to maximise cumulative reward. Neural networks, composed of interconnected nodes, are the basis for learning complex representations of data. Deep reinforcement learning (DRL) is the combination of RL algorithms with neural networks to enable agents to learn and make decisions based on their environment. 

There are different types of DRL algorithms: value-based algorithms (e.g., [DQN](https://arxiv.org/pdf/1312.5602.pdf)), policy-based algorithms (e.g., [PPO](https://arxiv.org/pdf/1707.06347.pdf)), and actor-critic algorithms (e.g., [DDPG](https://arxiv.org/pdf/1509.02971.pdf)).  

In our experiments, we used an open-source implementation of a DQN algorithm for multi-stock trading developed with the Python programming language. There are implementations of the model in [PyTorch](https://github.com/lazyprogrammer/machine_learning_examples/tree/master/pytorch) and in [TensorFlow](https://github.com/lazyprogrammer/machine_learning_examples/tree/master/tf2.0), which are two widely used machine learning (ML) frameworks. We ran the models on the backend nodes of Cirrus, on both CPU and GPU nodes.  


## Proposal
The requirements of project are categorised using the [**MoSCoW**](https://en.wikipedia.org/wiki/MoSCoW_method) method. The "Must Have" requirements include developing two variants of DRL models (value-based DQN and policy-based PPO) using both PyTorch and TensorFlow, testing the models on CPU and GPU nodes to measure performance, and tuning hyperparameters to optimise the models. The "Should Have" requirements involve developing an actor-critic DRL model (e.g., DDPG) using PyTorch and TensorFlow and testing its performance on CPU and GPU nodes. The "Could Have" requirements include building a risk management module to prevent large drawdowns and forward testing the models using live real-time data.  


## Content  
**Feasibility**  
The [feasibility](https://git.ecdf.ed.ac.uk/msc-22-23/s2134758/-/tree/main/feasibility) folder contains the feasibility report, data, code, and results from POC experiments. Further information on the data and code files, and instructions on reproducing the results are included in each particular model's folder.  


**Wiki**  
The [wiki](https://git.ecdf.ed.ac.uk/msc-22-23/s2134758/-/wikis/home) contains the notes from experimental observations, meetings, and literature review.  

