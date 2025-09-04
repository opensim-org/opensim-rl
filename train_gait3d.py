import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

# from wandb.integration.sb3 import WandbCallback


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

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 1000000,
    "env_id": "Gait3DEnv",
}
# run = wandb.init(
#     project="gait3d",
#     config=config,
#     sync_tensorboard=True,
# )
if __name__ == "__main__":
    num_envs = 10
    envs = SubprocVecEnv([make_env(seed=42 + i) for i in range(num_envs)])
    model = PPO(config["policy_type"], envs, verbose=1)
            # tensorboard_log=f"runs/{run.id}")
    model.learn(
        total_timesteps=config["total_timesteps"],
        progress_bar=True,
        # callback=WandbCallback(
        #     model_save_path=f"models/{run.id}",
        #     verbose=2,
        # ),
    )
    model.save(f"{models_dir}/{config['total_timesteps']}")
    # run.finish()