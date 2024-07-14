

# DAgger (Dataset Aggregation)

DAgger (Dataset Aggregation) is an imitation learning technique that uses an expert policy to train a new policy. Both the expert policy and the trained policy are implemented using Neural Networks. The expert policy achieves a mean squared error (MSE) of less than 0.001 radians and was trained on 4 million state-action pairs. The primary goal of this project is to enhance the capabilities of a two-finger gripper controlled by four process parameters (joint angles).

## Project Structure

- `imitation_learning.py`: This file contains the training of the policy (Neural Network) and the aggregation of the dataset.
- `fingers.py`: This file includes necessary utilities for training the policy, including the architectures of the expert and trained policies.
- `XML/`: This folder consists of the model XML file.

## Installation

To get started with this project, clone the repository and install the required dependencies.

```bash
git clone https://github.com/yourusername/DAgger.git
cd DAgger
pip install -r requirements.txt
```

## Usage

### Training the Policy

To train the policy using the imitation learning approach, run the `imitation_learning.py` script.

```bash
python imitation_learning.py
```

This script performs the following tasks:
- Loads the expert policy and the initial dataset.
- Trains the neural network policy using the dataset.
- Aggregates the dataset by incorporating the trained policy.

### Utilities and Architectures

The `fingers.py` file includes necessary utilities for training the policy, such as:
- Definition of the neural network architectures for both the expert and trained policies.
- Functions to preprocess data and evaluate policies.

### Model Files

The `XML` folder contains the model XML file required for simulating the two-finger gripper.

## Expert Policy

The expert policy is a neural network trained to achieve a mean squared error (MSE) of less than 0.001 radians. It was trained on 4 million state-action pairs, providing a robust foundation for training the new policy.

## Goal

The primary goal of this project is to improve the capabilities of a two-finger gripper, which is controlled by four process parameters (joint angles). By leveraging imitation learning, we aim to train a policy that can effectively control the gripper with high precision.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.


