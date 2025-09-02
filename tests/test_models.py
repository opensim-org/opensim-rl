import os
import opensim as osim
import numpy as np

# Import directly - this works when running from project root
from environments.model import OpenSimModel

# Use absolute path relative to this file's location
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'Gait3D.osim')
model = OpenSimModel(model_path, visualize=False, accuracy=1e-5, step_size=0.01)

# num_controls = model.get_num_controls()
# for i in range(10):
#     actions = np.random.uniform(-1, 1, size=(num_controls,))
#     model.actuate(actions)
#     model.step()

# model.reset()

# num_controls = model.get_num_controls()
# for i in range(10):
#     actions = np.random.uniform(-1, 1, size=(num_controls,))
#     model.actuate(actions)
#     model.step()

# obs = model.compute_observations()
# print(obs)


obs = model.get_observations()