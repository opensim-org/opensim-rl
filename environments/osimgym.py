import os
import sys
from abc import ABC, abstractmethod
from typing import Optional
import opensim as osim

import gymnasium as gym
import numpy as np

from environments.model import OpenSimModel

class OpenSimGym(gym.Env, ABC):

    def __init__(self, model_filepath, visualize, accuracy, step_size, observations):
        super().__init__()

        self.model = OpenSimModel(model_filepath, visualize, accuracy, step_size,
                                  observation_list=observations)

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

    @abstractmethod
    def _get_obs(self):
        pass

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
            "coordinate_kinematics",
            "body_kinematics",
            "center_of_mass_kinematics",
            "whole_body_momentum",
            "controls",
            "activations"
        ]

        model_filepath = os.path.join(os.getcwd(), 'models', 'Gait3DNoWrap.osim')

        super().__init__(model_filepath, visualize, accuracy, step_size, observations)

        self.observation_space = self.model.get_observation_space()

        # reward function
        self.target_speed = 1.2 # m/s
        self.grf_threshold = 1.2 # BW
        self.body_weight = self.model.get_body_weight()
        self.weights = [
            0.097, # action smoothing
            1.579, # num active muscles
            0.131, # joint limit torque
            0.073  # GRFs above threshold
        ]

        # reward adaptation
        self.reward_mean = 0.0
        self.schedule_mean = 0.0
        self.adaptation_rate = 0.0
        self.adaptation_rate_change = 9e-4
        self.threshold = 1000
        self.smoothing = 0.8
        self.decay = 0.9

        # early termination
        self.min_com_height = 0.5 # m

    def update_adaptation_rate(self):
        reward = self._get_reward()
        self.reward_mean = self.smoothing * self.reward_mean + \
                           (1 - self.smoothing) * reward

        if self.reward_mean > self.threshold and self.schedule_mean < 0.5:
            # performance is newly high, slow down adaptation
            self.adaptation_rate_change *= self.decay
        elif self.reward_mean > self.threshold and self.schedule_mean > 0.5:
            # performance high for too long
            self.adaptation_rate += self.adaptation_rate_change
        else:
            # performance too low
            self.adaptation_rate -= self.adaptation_rate_change

        schedule_target = 1 if self.reward_mean > self.threshold else 0
        self.schedule_mean = self.smoothing * self.schedule_mean + \
                            (1 - self.smoothing) * schedule_target

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.
        """
        self.update_adaptation_rate()
        self.model.reset()

        return self._get_obs(), self._get_info()

    def _get_obs(self):
        return self.model.get_scaled_observations()

    def get_velocity_reward(self):
        obs = self.model.get_observations()
        speed = np.linalg.norm(obs['center_of_mass_velocity'])
        diff = speed - self.target_speed
        if diff < 0:
            return np.exp(-1.0*diff*diff)
        else:
            return 1.0

    def get_effort(self):
        obs = self.model.get_observations()
        prev_obs = self.model.get_previous_observations()

        muscle_activity = np.sum(np.power(obs['activations'], 3))
        excitation_smoothness = np.sum(
                np.power(obs['controls'] - prev_obs['controls'], 2))
        num_active = np.sum(obs['activations'] > 0.15)

        return self.adaptation_rate * muscle_activity + \
               self.weights[0] * excitation_smoothness + \
               self.weights[1] * num_active

    def get_pain(self):
        obs = self.model.get_observations()
        generalized_forces = np.sum(np.abs(obs['CoordinateLinearStop_generalized_forces']))
        vertical_grf = obs['ExponentialContactForce_body_force'][1]

        pain = self.weights[2] * generalized_forces
        if (vertical_grf / self.body_weight) > self.grf_threshold:
            pain += self.weights[3] * vertical_grf

        return pain

    def _get_reward(self):
        return self.get_velocity_reward() - self.get_effort() - self.get_pain()

    def _get_done(self):
        """
        The episode ends if the center of mass is below min_com_height.
        """
        obs = self.model.get_observations()
        return obs['center_of_mass_position'][1] < self.min_com_height

    def _get_truncated(self):
        return False

    def _get_info(self):
        return {'time': self.model.state.getTime(),
                'steps': self.model.istep}
