# DAgger

DAgger (Dataset Aggregation) is an imitation learning technique, using expert policy a policy is trained. Both the expert policy and trained policy are Neural Nets, the expert policy has an MSE of [0.001<] radians which was trained on 4 million state-action pairs. The goal is to increase the capabilities of 2 finger gripper controlled by 4 process parameters (Joint angles).
The imitation_learning.py file contains training of the policy (Neural Net) and aggregation of dataset.
The fingers.py file contains necessary utilities for training the policy including the architectures of expert and trained policy.
XML folder consists of model xml file.
