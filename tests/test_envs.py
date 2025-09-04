import os
import opensim as osim
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register

register(id="Gait3DEnv",
         entry_point="environments.osimgym:Gait3DEnv",
         )

env = gym.make("Gait3DEnv")
actions = np.random.rand(env.env.env.model.get_num_controls())

env.reset()
n_steps = 1000
for step in range(n_steps):
    obs, reward, terminated, truncated, info = env.step(actions)
    print(reward)
    print(terminated, truncated)
    print(info)
