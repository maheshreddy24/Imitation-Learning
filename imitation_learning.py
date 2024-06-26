import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from fingers import Actor, RobotEnv



class DAgger:
    def __init__(self, actor, learning_rate=0.001):
        self.model = actor
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        #self.expert_data = expert_data
        self.aggregated_data = []
        self.target_count = 0

    def aggregate_data(self, state, action, expert_action):
        self.aggregated_data.append((state, action, expert_action))

    def train_policy(self):
        inputs = torch.tensor([xyz[0] for xyz in self.aggregated_data], dtype=torch.float32)
        actions = torch.tensor([xyz[1] for xyz in self.aggregated_data], dtype=torch.float32) #this is predicted actions
        outputs = torch.tensor([xyz[2] for xyz in self.aggregated_data], dtype=torch.float32, requires_grad=True) #this is predicted actions
        self.model.train()
        self.optimizer.zero_grad()
        #outputs = self.model(inputs)
        print('ouputs:', outputs)
        loss = self.criterion(outputs, actions)
        print('loss:', {loss.item()})
        loss.backward()
        self.optimizer.step()
        print('INside train_policy')
        return loss.item()

    def predict_action(self, state):
        self.model.eval()
        state_tensor = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            return self.model(state_tensor).numpy()

    def run(self, iterations):
        beta = 1
        beta_decay = 0.95
        for x in range(iterations):
            if(self.target_count <= iterations):
                robot.reset()
                done = False
                print('inside run')
                if(True):
                    #state = current_state()
                    target_state = robot.set_new_state_target(self.target_count)
                    self.target_count += 1

                    expert_action = robot.get_expert_action(target_state)
                    model_action = self.predict_action(target_state)
                    action = beta * expert_action + (1 - beta) * model_action
                    robot.step(action, self.target_count-1)    
                    self.aggregate_data(target_state, action, expert_action)

                    #done = True
                print("iteration count: ", x)
                self.train_policy()
                beta *= beta_decay


if __name__ == "__main__":
    # Initialization of the policy model
    actor = Actor()
    #expert_data = np.load('expert_data.npy', allow_pickle=True)  # Ensure this is the correct path and format
    robot = RobotEnv() #instance of class robotevn
    # Create an instance of the DAgger class
    dagger = DAgger(actor)
    print('into')
    # Run the DAgger algorithm
    dagger.run(10)  # Set the number of iterations as needed
    torch.save(actor.state_dict(), 'actor-model-1.pt')
    print('last')