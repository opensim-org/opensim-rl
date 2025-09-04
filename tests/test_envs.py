import os
import opensim as osim
import numpy as np
from environments.osimgym import Gait3DEnv

# Use absolute path relative to this file's location
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'Gait3D.osim')


env = Gait3DEnv(model_path)

actions = np.zeros(env.model.get_num_controls())
actions[5] = 0.5

n_steps = 1000

for step in range(n_steps):
    obs, reward, terminated, truncated, info = env.step(actions)
    print(reward)
    print(terminated, truncated)
    print(info)
