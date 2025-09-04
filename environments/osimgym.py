import os
import sys
from abc import ABC, abstractmethod
from typing import Optional
import opensim as osim

import gymnasium as gym
import numpy as np

from environments.model import OpenSimModel

class OpenSimGym(gym.Env, ABC):

    def __init__(self, model_filepath, visualize, accuracy, step_size, observations,
                 aggregators=dict(), aggregator_scales=dict()):
        super().__init__()

        self.model = OpenSimModel(model_filepath, visualize, accuracy, step_size,
                                  observation_list=observations,
                                  aggregators=aggregators,
                                  aggregator_scales=aggregator_scales)

        self.observation_space = self.model.get_observation_space()
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0,
                                           shape=(self.model.get_num_controls(),),
                                           dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.
        """
        self.model.reset()

        return self._get_obs(), self._get_info()

    def step(self, action):
        """Execute one timestep within the environment.
        """

        self.model.actuate(action)
        self.model.step()

        return self._get_obs(), self._get_reward(), self._get_done(), \
               self._get_truncated(), self._get_info()

    def _get_obs(self):
        return self.model.get_observations()

    @abstractmethod
    def _get_reward(self):
        pass

    @abstractmethod
    def _get_done(self):
        pass

    @abstractmethod
    def _get_truncated(self):
        pass

    @abstractmethod
    def _get_info(self):
        pass


class Gait3DEnv(OpenSimGym):

    def __init__(self, visualize=False, accuracy=1e-3, step_size=0.0025):

        observations = [
            "coordinate_values",
            "coordinate_speeds",
            "body_positions",
            # "body_velocities",
            # "body_accelerations",
            "body_orientations",
            # "body_angular_velocities",
            # "body_angular_accelerations",
            "center_of_mass_position",
            "center_of_mass_velocity",
            "center_of_mass_acceleration",
            # "whole_body_linear_momentum",
            # "whole_body_angular_momentum",
            "controls",
            "muscle_activations",
            "muscle_lengths",
            "muscle_velocities",
            "muscle_forces",
        ]

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

        model_filepath = os.path.join(os.getcwd(), 'models', 'Gait3DNoWrap.osim')

        super().__init__(model_filepath, visualize, accuracy, step_size, observations,
                         aggregators=aggregators, aggregator_scales=aggregator_scales)

        # reward function
        self.target_speed = 1.2 # m/s
        self.grf_threshold = 1.2 # BW
        self.body_weight = self.model.get_body_weight()
        self.weights = [
            0.1, # action smoothing
            0.1, # num active muscles
            0.001, # joint limit torque
            0.0005  # GRFs above threshold
        ]

        # reward adaptation
        self.reward_mean = 0.0
        self.schedule_mean = 0.0
        self.adaptation_rate = 1.0
        self.adaptation_rate_change = 9e-4
        self.threshold = 1000
        self.smoothing = 0.8
        self.decay = 0.9

        # early termination
        self.min_com_height = 0.5 # m

    def update_adaptation_rate(self):
        print('Updating adaptation rate...')
        reward = self._get_reward()
        print('--> Reward: ', reward)
        self.reward_mean = self.smoothing * self.reward_mean + \
                           (1 - self.smoothing) * reward
        print('--> Reward mean: ', self.reward_mean)

        if self.reward_mean > self.threshold and self.schedule_mean < 0.5:
            # performance is newly high, slow down adaptation
            self.adaptation_rate_change *= self.decay
            print('--> Slowing down adaptation: ', self.adaptation_rate_change)
        elif self.reward_mean > self.threshold and self.schedule_mean > 0.5:
            # performance high for too long
            self.adaptation_rate += self.adaptation_rate_change
            print('--> Adaptation rate increase: ', self.adaptation_rate)
        else:
            # performance too low
            self.adaptation_rate -= self.adaptation_rate_change
            print('--> Adaptation rate decrease: ', self.adaptation_rate)

        schedule_target = 1 if self.reward_mean > self.threshold else 0
        self.schedule_mean = self.smoothing * self.schedule_mean + \
                            (1 - self.smoothing) * schedule_target
        print('--> Schedule mean: ', self.schedule_mean)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.
        """
        print('Starting new episode...')
        reward = self._get_reward()
        print('--> Reward: ', reward)
        # self.update_adaptation_rate()
        self.model.reset()

        return self._get_obs(), self._get_info()

    def get_velocity_reward(self):
        outputs = self.model.get_outputs()
        speed = np.linalg.norm(outputs['center_of_mass_velocity'])
        diff = speed - self.target_speed
        if diff < 0:
            return np.exp(-1.0*diff*diff)
        else:
            return 1.0

    def get_effort(self):
        outputs = self.model.get_outputs()
        prev_outputs = self.model.get_previous_outputs()

        muscle_activity = np.sum(np.power(outputs['muscle_activations'], 3))
        excitation_smoothness = np.sum(
                np.power(outputs['controls'] - prev_outputs['controls'], 2))
        num_active = np.sum(outputs['muscle_activations'] > 0.15)
        # print('muscle activity: ', muscle_activity)
        # print('excitation smoothness: ', excitation_smoothness)
        # print('num active: ', num_active)

        return self.adaptation_rate * muscle_activity + \
               self.weights[0] * excitation_smoothness + \
               self.weights[1] * num_active

    def get_pain(self):
        outputs = self.model.get_outputs()
        limit_torques = np.sum(np.abs(outputs['limit_torques_generalized_forces']))
        grfs = outputs['contact_forces_body_forces'][-1] + \
               outputs['contact_forces_body_forces'][-2]

        # print('limit torques: ', limit_torques)
        # print('vertical GRF: ', grfs[1])

        pain = self.weights[2] * limit_torques
        if (grfs[1] / self.body_weight) > self.grf_threshold:
            pain += self.weights[3] * grfs[1]

        return pain

    def _get_reward(self):
        return self.get_velocity_reward() - self.get_effort() - self.get_pain()

    def _get_done(self):
        """
        The episode ends if the center of mass is below min_com_height.
        """
        outputs = self.model.get_outputs()
        return outputs['center_of_mass_position'][1] < self.min_com_height

    def _get_truncated(self):
        return False

    def _get_info(self):
        return {'time': self.model.state.getTime(),
                'steps': self.model.istep}
