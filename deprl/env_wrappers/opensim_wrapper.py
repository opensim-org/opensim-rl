import numpy as np

from .wrappers import ExceptionWrapper


class CustomOpenSimException(Exception):
    """
    Custom exception class for OpenSim.
    """

    pass


class OpenSimWrapper(ExceptionWrapper):
    """Wrapper for OpenSim, compatible with gym=0.13.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.error = CustomOpenSimException

    def render(self, *args, **kwargs):
        pass

    def muscle_lengths(self):
        length = self.unwrapped.model.get_outputs()["muscle_lengths"]
        return length

    def muscle_forces(self):
        force = self.unwrapped.model.get_outputs()["muscle_forces"]
        return force

    def muscle_velocities(self):
        velocity = self.unwrapped.model.get_outputs()["muscle_velocities"]
        return velocity

    def muscle_activity(self):
        return self.unwrapped.model.get_outputs()["muscle_activations"]

    @property
    def _max_episode_steps(self):
        return 1000
