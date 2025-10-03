import os
import json
import argparse
import opensim as osim

import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO

register(id="Gait3D",
        entry_point="environments.osimgym:Gait3D",
        )

def viz(model_name, outputs_dir, timesteps):

    eval_dir = os.path.join(outputs_dir, model_name, 'eval')
    model = osim.Model(os.path.join(eval_dir, f'{model_name}.osim'))
    model.initSystem()
    table = osim.TimeSeriesTable(
            os.path.join(eval_dir, f'{model_name}_{timesteps}steps.sto'))
    osim.VisualizerUtilities.showMotion(model, table)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train OpenSim RL simulations.')
    parser.add_argument('--model-name', type=str,
                        help='Directory containing the trained model and config file.')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Root directory for saving outputs (default: outputs)')
    parser.add_argument('--timesteps', type=int,
                        help='Total timesteps used per training iteration.')
    args = parser.parse_args()

    viz(args.model_name, args.output_dir, args.timesteps)