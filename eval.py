import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
from gymnasium.wrappers import FlattenObservation
import os

from stable_baselines3 import PPO

models_dir = "models_gait3d/PPO" # location of models
model_path = f"{models_dir}/1000000.zip" # the name of the model to use

register(id="Gait3DEnv",
         entry_point="environments.osimgym:Gait3DEnv",
         )

env = gym.make("Gait3DEnv", visualize=False)
wrapped_env = FlattenObservation(env)

policy = PPO.load(model_path, env=wrapped_env)

terminated = False
obs, info = wrapped_env.reset(seed=0)
for i in range(25000):
    action, _ = policy.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = wrapped_env.step(action)
    print(reward)

