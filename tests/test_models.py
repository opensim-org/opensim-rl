import os
import opensim as osim
import numpy as np

# Import directly - this works when running from project root
from environments.model import OpenSimModel

from gymnasium.utils import passive_env_checker

# Use absolute path relative to this file's location
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'Gait3D.osim')
model = OpenSimModel(model_path, visualize=False, accuracy=1e-5, step_size=0.01)

