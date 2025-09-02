import os
import sys
from abc import ABC, abstractmethod
from typing import Optional

import gymnasium as gym
import numpy as np

from environments.model import OpenSimModel

class OpenSimGym(gym.Env, ABC):

    def __init__(self, model_filepath, visualize, accuracy, step_size, observations,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.model_filepath = model_filepath
        self.visualize = visualize
        self.accuracy = accuracy
        self.step_size = step_size
        self.observations = observations
        self.model = OpenSimModel(self.model_filepath, self.visualize, self.accuracy,
                                  self.step_size, observations=self.observations)

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0,
                                           shape=(self.model.get_num_controls(),),
                                           dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.
        """

        return self._get_obs(), self._get_info

    def step(self, action):
        """Execute one timestep within the environment.
        """

        self.model.actuate(action)
        self.model.step()

        return self._get_obs(), self._get_reward(), self._get_terminated(), \
               self._get_truncated(), self._get_info()

    @abstractmethod
    def _get_obs(self):
        pass

    @abstractmethod
    def _get_reward(self):
        pass

    @abstractmethod
    def _get_terminated(self):
        pass

    @abstractmethod
    def _get_truncated(self):
        pass

    @abstractmethod
    def _get_info(self):
        pass


class Gait3DEnv(OpenSimGym):

    def __init__(self, model_filepath, visualize=False, accuracy=1e-3,
                 step_size=0.0025):

        observations = [
            "coordinate_kinematics",
            "body_kinematics",
            "center_of_mass_kinematics",
            "whole_body_momentum",
            "controls",
        ]

        super().__init__(model_filepath, visualize, accuracy, step_size, observations)

        self.observation_space = gym.spaces.Dict({
            "angles": gym.spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32),
            "velocities": gym.spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32),
            "body_pos": gym.spaces.Box(low=-1, high=1, shape=(12*2,), dtype=np.float32),
            "controls": gym.spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32),
        })



    def _get_obs(self):
        pass

    def _get_reward(self):
        pass

    def _get_terminated(self):
        pass

    def _get_truncated(self):
        return False

    def _get_info(self):
        pass