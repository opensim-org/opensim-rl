from base_osim import armEnv
import numpy as np
from gymnasium.wrappers import FlattenObservation

model_path = 'C:/Users/Nicos/Documents/OpenSim/4.5-2024-01-10-3b63585/Models/Arm26/arm26_4_PC.osim'

osim_env = armEnv(visualize=False)
print(osim_env.get_observation_space_size())
#osim_env.load_model(model_path=model_path)

osim_env.reset()
print(osim_env.get_state_desc()["joint_pos"])
print(len(osim_env.get_observation()))
#print(osim_env.get_state_desc()["joint_vel"])
#print(osim_env.get_state_desc()["joint_acc"])
#print(osim_env.get_state_desc()["muscles"])

#print(osim_env.get_action_space_size())

#osim_env.osim_model.markerSet.get(2).getName()

action_vec = np.ones(osim_env.get_action_space_size()) * 0.00
action_vec[5] = 0.5

n_steps = 1000

for step in range(n_steps):
    obs, reward = osim_env.step(action_vec, obs_as_dict=False)
    print(reward)


osim_env.osim_model.write_and_viz()