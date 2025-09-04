import os
import opensim as osim
import numpy as np
import time
from gymnasium.envs.registration import register
import gymnasium as gym

from deprl import env_wrappers
from deprl.dep_controller import DEP


register(id="Gait3DEnv",
         entry_point="environments.osimgym:Gait3DEnv",
         )

env = gym.make("Gait3DEnv")
env = env_wrappers.GymWrapper(env)

import pdb; pdb.set_trace()

dep = DEP()
dep.initialize(env.observation_space, env.action_space)

env.reset()
for i in range(1000):
    action = dep.step(env.muscle_lengths())[0, :]
    next_state, reward, done, _ = env.step(action)
    time.sleep(0.005)