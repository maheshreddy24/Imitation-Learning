DAgger (Dataset Aggregation)

DAgger (Dataset Aggregation) is an advanced imitation learning algorithm that iteratively refines a policy by leveraging an expert policy. This method is particularly useful in cases where direct supervised learning from an expert is insufficient due to compounding errors in long-horizon tasks.

In this project, DAgger is utilized to enhance the performance of a two-finger robotic gripper, which is controlled by four process parameters (joint angles). The expert policy, implemented as a neural network, provides high-accuracy demonstrations, and the trained policy learns from the expert through iterative dataset aggregation.

Project Structure

imitation_learning.py - Implements the DAgger algorithm, including training and dataset aggregation.

fingers.py - Contains utility functions and neural network architectures for the expert and trained policies.

XML/ - Includes the XML model file for simulating the two-finger gripper.

Installation

To set up the project, clone the repository and install the required dependencies:

git clone https://github.com/yourusername/DAgger.git
cd DAgger
pip install -r requirements.txt

Usage

Training the Policy

To train the policy using the DAgger approach, run the imitation_learning.py script:

python imitation_learning.py

This script performs the following steps:

Loads the expert policy and initializes the dataset.

Trains the initial policy using the dataset.

Iteratively refines the policy by incorporating expert-labeled corrections into the dataset.

Updates the trained policy with the aggregated dataset.

Utilities and Architectures

The fingers.py file contains:

Neural Network Architectures: Defines both the expert policy and the trained policy.

Data Preprocessing Functions: Prepares state-action pairs for training.

Evaluation Functions: Measures policy performance against the expert policy.

Model Files

The XML/ folder contains the simulation model for the two-finger gripper, which is required for training and evaluation in a simulated environment.

Expert Policy

The expert policy is a neural network trained to achieve an MSE of less than 0.001 radians. It was trained on a dataset of 4 million state-action pairs, making it highly reliable for providing ground-truth actions.

