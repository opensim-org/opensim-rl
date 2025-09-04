import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

import os
import torch as th

from stable_baselines3 import PPO

models_dir = "models_gait3d/PPO"
logs_dir = "logs_gait3d"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

register(id="Gait3DEnv",
         entry_point="environments.osimgym:Gait3DEnv",
         )

def make_env(env_id="Gait3DEnv", seed=0, rank=0):

    def _init():
        env = gym.make(env_id)
        env = FlattenObservation(env)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    num_envs = 5
    train_envs = SubprocVecEnv([make_env(seed=42 + i) for i in range(num_envs)])
    policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[128, 64], vf=[64, 64]))


    model = PPO("MlpPolicy", train_envs, verbose=2,
            policy_kwargs=policy_kwargs, gamma=0.998)


    timesteps = 10000
    for i in range(1,250):
        model.learn(total_timesteps=timesteps, reset_num_timesteps=False,
                    tb_log_name="PPO_test", progress_bar=True)
        model.save(f"{models_dir}/{timesteps*i}")