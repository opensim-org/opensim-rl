import os
import opensim as osim
import numpy as np
import time
from gymnasium.envs.registration import register
import gymnasium as gym

from deprl import env_wrappers
from deprl.dep_controller import DEP


register(id="Gait3DEnv",
         entry_point="environments.osimgym:Gait3DEnv"
         )

env = gym.make("Gait3DEnv", visualize=True)
env = env_wrappers.OpenSimWrapper(env)

dep = DEP()
dep.initialize(env.observation_space, env.action_space)

env.reset()
for i in range(1000):
    muscle_lengths = np.concatenate([env.muscle_lengths(), np.array([1.0, 1.0, 1.0])], dtype=np.float32)
    action = dep.step(muscle_lengths)[0, :]
    next_state, reward, done, _ = env.step(action)
    time.sleep(0.005)