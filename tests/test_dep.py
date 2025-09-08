import os
import opensim as osim
import numpy as np
import time
from gymnasium.envs.registration import register
import gymnasium as gym

from deprl import env_wrappers
from deprl.dep_controller import DEP


register(id="Gait3D",
         entry_point="environments.osimgym:Gait3D"
         )

env = gym.make("Gait3D", visualize=False)
env = env_wrappers.OpenSimWrapper(env)

dep = DEP()
dep.initialize(env.observation_space, env.action_space)

states_traj = osim.StatesTrajectory()
env.reset()
for i in range(1000):
    print('step: ', i)
    muscle_lengths = np.concatenate([env.muscle_lengths(), np.array([1.0, 1.0, 1.0])], dtype=np.float32)
    action = dep.step(muscle_lengths)[0, :]
    obs, reward, done, _ = env.step(action)
    states_traj.append(env.get_state())
    time.sleep(0.005)

table = states_traj.exportToTable(env.get_model())
table.addTableMetaDataString("inDegrees", "no")
osim.VisualizerUtilities.showMotion(env.get_model(), table)
