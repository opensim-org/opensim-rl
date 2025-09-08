import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

import wandb
from wandb.integration.sb3 import WandbCallback

import os
import torch as th
import argparse

from stable_baselines3 import PPO


register(id="Gait3D",
         entry_point="environments.osimgym:Gait3D",
         )

def make_env(env_id="Gait3D", seed=0, rank=0):
    def _init():
        env = gym.make(env_id)
        env = FlattenObservation(env)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init

def train(env_id, output_dir, timesteps=1000000, num_envs=10, seed=42):

    # Update directories based on command line arguments
    models_dir = os.path.join(output_dir, 'models', 'PPO')
    logs_dir = os.path.join(output_dir, 'logs', 'PPO')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": timesteps,
        "env_id": env_id,
        "num_envs": num_envs,
        "seed": seed,
    }

    # Initialize WandB if not disabled
    run = wandb.init(
        project=env_id,
        config=config,
        sync_tensorboard=True,
    )
    tensorboard_log = f"runs/{run.id}"

    # Create environments
    envs = SubprocVecEnv([make_env(seed=seed + i) for i in range(num_envs)])

    # Create model
    model = PPO(
        config["policy_type"],
        envs,
        verbose=1,
        tensorboard_log=tensorboard_log
    )

    # Prepare callbacks
    callbacks = []
    wandb_callback = WandbCallback(
        model_save_path=f"models/{run.id}",
        verbose=2,
    )
    callbacks.append(wandb_callback)

    # Train the model
    model.learn(
        total_timesteps=config["total_timesteps"],
        progress_bar=True,
        callback=callbacks,
    )

    # Save the model
    model.save(f"{models_dir}/{config['total_timesteps']}")

    # Finish WandB run
    run.finish()

    print(f"Training completed. Model saved to {models_dir}/{config['total_timesteps']}.zip")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train OpenSim RL simulations.')
    parser.add_argument('--env-id', type=str,
                        help='Gym environment ID')
    parser.add_argument('--output-dir', type=str,
                        help='Directory to save trained models')
    parser.add_argument('--timesteps', type=int, default=100000,
                        help='Total timesteps per training iteration (default: 1000000)')
    parser.add_argument('--iterations', type=int, default=1,
                        help='Number of training iterations (default: 1)')
    parser.add_argument('--num-envs', type=int, default=10,
                        help='Number of parallel environments (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    train(args.env_id, args.output_dir, args.timesteps, args.num_envs, args.seed)