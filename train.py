import os
import argparse
import json

import wandb
from wandb.integration.sb3 import WandbCallback
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO

register(id='Gait3D',
         entry_point='environments.osimgym:Gait3D',
         )

def make_env(env_id, seed=0, rank=0):
    def _init():
        env = gym.make(env_id)
        env = FlattenObservation(env)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init

def train(env_id, model_name, output_dir, timesteps=1000000, num_envs=10, seed=42):

    # Update directories based on command line arguments
    model_dir = os.path.join(output_dir, model_name)
    models_dir = os.path.join(model_dir, 'models', 'PPO')
    logs_dir = os.path.join(model_dir, 'logs', 'PPO')
    wandb_dir = os.path.join(model_dir, 'wandb')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    if not os.path.exists(wandb_dir):
        os.makedirs(wandb_dir)

    config = {
        'policy_type': 'MlpPolicy',
        'total_timesteps': timesteps,
        'env_id': env_id,
        'num_envs': num_envs,
        'seed': seed,
    }

    # Save the config to a JSON file.
    with open(f'{models_dir}/config.json', 'w') as f:
        json.dump(config, f)

    # Initialize WandB if not disabled
    run = wandb.init(
        dir=wandb_dir,
        project=env_id,
        config=config,
        sync_tensorboard=True,
    )
    tensorboard_log = f'runs/{run.id}'

    # Create environments
    envs = SubprocVecEnv(
        [make_env(env_id=env_id, seed=seed + i) for i in range(num_envs)]
    )

    # Create model
    model = PPO(
        config['policy_type'],
        envs,
        verbose=1,
        tensorboard_log=tensorboard_log
    )

    # Prepare callbacks
    callbacks = []
    wandb_callback = WandbCallback(
        model_save_path=os.path.join(wandb_dir, run.id),
        verbose=2,
    )
    callbacks.append(wandb_callback)

    # Train the model
    model.learn(
        total_timesteps=config['total_timesteps'],
        progress_bar=True,
        callback=callbacks,
    )

    # Save the model
    model.save(os.path.join(models_dir, f'{model_name}_{timesteps}steps'))

    # Finish WandB run
    run.finish()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train OpenSim RL simulations.')
    parser.add_argument('--env-id', type=str,
                        help='Gym environment ID')
    parser.add_argument('--model-name', type=str,
                        help='Name of the trained RL model')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Root directory for saving outputs (default: outputs)')
    parser.add_argument('--timesteps', type=int, default=100000,
                        help='Total timesteps per training iteration (default: 100000)')
    parser.add_argument('--iterations', type=int, default=1,
                        help='Number of training iterations (default: 1)')
    parser.add_argument('--num-envs', type=int, default=10,
                        help='Number of parallel environments (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    train(args.env_id, args.model_name, args.output_dir, args.timesteps, args.num_envs,
          args.seed)