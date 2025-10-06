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

def eval(model_name, outputs_dir):

    model_path = os.path.join(outputs_dir, model_name, 'models', 'PPO', model_name)
    config_path = os.path.join(outputs_dir, model_name, 'models', 'PPO', 'config.json')
    config = json.load(open(config_path, 'r'))

    eval_dir = os.path.join(outputs_dir, model_name, 'eval')
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    env = gym.make(config['env_id'], visualize=False)
    wrapped_env = FlattenObservation(env)

    policy = PPO.load(model_path, env=wrapped_env)

    states_trajectory = osim.StatesTrajectory()
    obs, info = wrapped_env.reset(seed=0)
    for i in range(1000):
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, _, _, info = wrapped_env.step(action)
        state = wrapped_env.unwrapped.get_model().get_state()
        states_trajectory.append(state)

    model = wrapped_env.unwrapped.get_model().get_model()
    states_table = states_trajectory.exportToTable(model)
    states_table.addTableMetaDataString('inDegrees', 'no')

    sto = osim.STOFileAdapter()
    sto.write(states_table,
              os.path.join(eval_dir, f'{model_name}.sto'))

    model.printToXML(os.path.join(eval_dir, f'{model_name}.osim'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate trained RL models.')
    parser.add_argument('--model-name', type=str,
                        help='Directory containing the trained model and config file.')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Root directory for saving outputs (default: outputs)')
    args = parser.parse_args()

    eval(args.model_name, args.output_dir)