import os
import opensim as osim
import numpy as np

# Import directly - this works when running from project root
from environments.model import OpenSimModel

from gymnasium.utils import passive_env_checker

# Use absolute path relative to this file's location
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'Gait3D.osim')

limit_torques = [
            "/forceset/lumbar_coord_0_stop",
            "/forceset/lumbar_coord_1_stop",
            "/forceset/lumbar_coord_2_stop",
            "/forceset/hip_l_coord_0_stop",
            "/forceset/hip_l_coord_1_stop",
            "/forceset/hip_l_coord_2_stop",
            "/forceset/hip_r_coord_0_stop",
            "/forceset/hip_r_coord_1_stop",
            "/forceset/hip_r_coord_2_stop",
            "/forceset/knee_l_coord_0_stop",
            "/forceset/knee_r_coord_0_stop",
            "/forceset/ankle_l_coord_0_stop",
            "/forceset/ankle_r_coord_0_stop"
        ]

contact_forces = [
    "/forceset/left_heel_contact",
    "/forceset/left_lateralToe_contact",
    "/forceset/left_medialToe_contact",
    "/forceset/right_heel_contact",
    "/forceset/right_lateralToe_contact",
    "/forceset/right_medialToe_contact"
]

aggregators = {
    "limit_torques": limit_torques,
    "contact_forces": contact_forces,
}

aggregator_scales = {
    "limit_torques": 250.0,
    "contact_forces": 5000
}
model = OpenSimModel(model_path, visualize=False, accuracy=1e-5, step_size=0.01,
                     aggregators=aggregators, aggregator_scales=aggregator_scales)


num_controls = model.get_num_controls()
for i in range(10000):
    # print('step: ', i)
    actions = np.random.uniform(-1, 1, size=(num_controls,))
    model.actuate(actions)
    model.step()
    obs = model.get_observations()
