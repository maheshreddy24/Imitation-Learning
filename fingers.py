import torch
import torch.nn as nn
import numpy as np
import mujoco
from PIL import Image
import os
import time
import pandas as pd



path = 'xml/gripper.xml'

# Initialize MuJoCo components
mj_model = mujoco.MjModel.from_xml_path(path)
data = mujoco.MjData(mj_model)
cam = mujoco.MjvCamera()
opt = mujoco.MjvOption()

mujoco.mj_step(mj_model, data)


#dont open this function just has manually written data
def data_cube():
    # List of y-values
    cube_y = [
        [0.15, 0.17494196281460858],
        [0.15, 0.18126395973783844],
        [0.15, 0.1875859566610683],
        [0.15, 0.19390795358429817],
        [0.15, 0.20022995050752803],
        [0.15, 0.2065519474307579],
        [0.15, 0.21287394435398776],
        [0.15, 0.21919594127721763],
        [0.15, 0.2255179382004475],
        [0.15, 0.23183993512367736]
    ]

    # List of x-values
    cube_x = [
        [0.15, 0.23183993512367736],
        [0.145, 0.23183993512367736],
        [0.14, 0.23183993512367736],
        [0.135, 0.23183993512367736],
        [0.13, 0.23183993512367736],
        [0.125, 0.23183993512367736],
        [0.12, 0.23183993512367736],
        [0.115, 0.23183993512367736],
        [0.11, 0.23183993512367736],
        [0.105, 0.23183993512367736],
        [0.1, 0.23183993512367736],
        [0.095, 0.23183993512367736],
        [0.09, 0.23183993512367736],
        [0.085, 0.23183993512367736],
        [0.08, 0.23183993512367736],
        [0.075, 0.23183993512367736],
        [0.07, 0.23183993512367736],
        [0.065, 0.23183993512367736],
        [0.06, 0.23183993512367736],
        [0.055, 0.23183993512367736]
    ]

    # List of y_neg-values
    cube_y_neg = [
        [0.055, 0.22183993512367736],
        [0.055, 0.21183993512367736],
        [0.055, 0.20183993512367736],
        [0.055, 0.19183993512367736],
        [0.055, 0.18183993512367736],
        [0.055, 0.17183993512367736],
        [0.055, 0.16183993512367736],
        [0.055, 0.15183993512367736],
        [0.055, 0.14183993512367736],
        [0.055, 0.13183993512367736]
    ]

    # Combining all the arrays
    combined_array = cube_y + cube_x + cube_y_neg
    return np.array(combined_array)

def save_image(num = 0):
    renderer = mujoco.Renderer(mj_model)
    renderer.update_scene(data)
    # Render the image
    rendered_image = renderer.render()
    pil_image = Image.fromarray(rendered_image)
    folder_path = "cube-vid"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    filename = f'image_{num}.png'
    image_path = os.path.join(folder_path, filename)
    pil_image.save(image_path)

save_image()

class ExpertModel(nn.Module):
    def __init__(self):
        super(ExpertModel, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 4),
        )

    def forward(self, z):
        logits = self.linear_relu_stack(z)
        return logits

torch.manual_seed(42)
expert_model = ExpertModel()
expert_model.load_state_dict(torch.load('model.pt'))
expert_model.eval()

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 4),
        )

    def forward(self, z):
        logits = self.linear_relu_stack(z)
        return logits

torch.manual_seed(42)
actor_model = Actor()




class RobotEnv:
    def __init__(self): #11
        self.target_pos = data_cube() #this returns cube trajectory

    def reset(self):
        print('Resetting the environment')
        mujoco.mj_resetData(mj_model, data)
        mujoco.mj_step(mj_model, data)
        save_image()
        return self.current_state()

    def set_new_state_target(self, i):
        # Simulate getting a new target position
        res = self.target_pos[i]
        ret = np.array([res[0],res[1]])
        return ret

    def get_expert_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            output = expert_model(state)
        return output.numpy()

    def step(self, action, i):
        # Ensure action is within the control range
        action = np.clip(action, -1, 1)

        # Assign control values to the actuators
        data.ctrl[0] = action[0]
        data.ctrl[1] = action[1]
        data.ctrl[2] = action[2]
        data.ctrl[3] = action[3]
        

        print(action)
        # Step the simulation forward
        for _ in range(10):  # Step multiple times for stability
            mujoco.mj_step(mj_model, data)

        save_image(i+1)
        return

    def current_state(self):
        mujoco.mj_step(mj_model, data)
        return np.array([data.geom_xpos[1][0], data.geom_xpos[1][1]])

    def __call__(self):
        return self.reset(), self.get_expert_action(self.current_state()), self.step(), self.current_state(), self.set_new_state_target()
