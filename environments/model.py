from xml.parsers.expat import model
import opensim as osim
import numpy as np
import gymnasium as gym
from copy import deepcopy

DEFAULT_OBSERVATION_LIST = [
    "coordinate_kinematics",
    "body_kinematics",
    "center_of_mass_kinematics",
    "whole_body_momentum",
    "controls",
    "activations"
]

OBSERVATION_SCALES = {
    "coordinate_values": 5.0*np.pi,
    "coordinate_speeds": 50.0*np.pi,
    "body_positions": 5.0,
    "body_velocities": 50.0,
    "body_accelerations": 500.0,
    "body_orientations": 50.0*np.pi,
    "body_angular_velocities": 500.0*np.pi,
    "body_angular_accelerations": 5000.0*np.pi,
    "center_of_mass_position": 5.0,
    "center_of_mass_velocity": 50.0,
    "center_of_mass_acceleration": 500.0,
    "whole_body_linear_momentum": 1000.0,
    "whole_body_angular_momentum": 1000.0,
    "controls": 1.0,
    "activations": 1.0
}

AGGREGATED_FORCE_LIST = [
    "ExponentialContactForce",
    "CoordinateLinearStop",
]

FORCE_SCALES = {
    "ExponentialContactForce": 25000.0,
    "CoordinateLinearStop": 5000.0,
}

class OpenSimModel:
    """
    A class to store an OpenSim model and convenient methods for reinforcement learning.
    """
    obs = dict()
    prev_obs = dict()

    def __init__(self, model_filepath, visualize, accuracy, step_size,
                 observation_list=DEFAULT_OBSERVATION_LIST,
                 force_list=AGGREGATED_FORCE_LIST,
                 force_scales=FORCE_SCALES):

        # Initialize the OpenSim model.
        self.model = osim.Model(model_filepath)
        self.model.initSystem()
        self.visualize = visualize
        self.model.setUseVisualizer(self.visualize)

        # Disable all logging except errors (or worse).
        osim.Logger.setLevelString('error')

        # Model controls.
        # ---------------
        # Check that the order of the controls in the controls vector matches the order
        # of the actuators in the model.
        osim.checkOrderSystemControls(self.model)

        # Get the list of control paths based on the order of the actuators in the
        # model.
        self.control_paths = osim.createControlNamesFromModel(self.model)

        # Compute the minimum and maximum control values.
        self.min_controls = []
        self.max_controls = []
        for control_path in self.control_paths:
            actu = osim.ScalarActuator.safeDownCast(self.model.getComponent(control_path))
            if actu:
                self.min_controls.append(actu.getMinControl())
                self.max_controls.append(actu.getMaxControl())
            else:
                raise Exception(f"Control {control_path} is not a ScalarActuator.")

        # Store the size of the controls vector and a SimTK::Vector to hold the controls
        # we will receive during policy training.
        self.num_controls = self.model.getNumControls()
        assert(len(self.min_controls) == self.num_controls)
        assert(len(self.max_controls) == self.num_controls)
        self.controls = osim.Vector(self.num_controls, 0.0)

        # Add a DiscreteController to the model. This allows us to update the controls
        # within the SimTK::State object.
        self.controller = osim.DiscreteController()
        self.controller.setName('brain')
        self.model.addController(self.controller)

        # Model activations.
        # -----------------
        self.num_activations = 0
        self.activation_muscles = []
        self.activation_coordinate_actuators = []
        for component in self.model.getComponentsList():
            if "Muscle" in component.getConcreteClassName():
                if "MuscleFixedWidthPennationModel" in component.getConcreteClassName():
                    continue
                if "MuscleFirstOrderActivationDynamicModel" in component.getConcreteClassName():
                    continue

                if not component.get_ignore_activation_dynamics():
                    self.num_activations += 1
                    self.activation_muscles.append(component)
            elif "ActivationCoordinateActuator" in component.getConcreteClassName():
                self.num_activations += 1
                self.activation_coordinate_actuators.append(component)

        # Force aggregators.
        # ------------------
        self.force_list = force_list
        self.force_scales = force_scales
        assert(len(force_list) == len(force_scales))
        for force_class in AGGREGATED_FORCE_LIST:
            force_aggregator = osim.ForceAggregator()
            force_aggregator.setName(f'{force_class}_aggregator')
            for component in self.model.getComponentsList():
                if force_class in component.getConcreteClassName():
                    force_aggregator.addForce(component)
            self.model.addComponent(force_aggregator)

        # Initialize the state.
        self.state = self.model.initSystem()

        # Integration settings.
        self.istep = 0
        self.accuracy = accuracy
        self.step_size = step_size

        # Initialize the manager.
        self.manager = osim.Manager(self.model)
        self.manager.setIntegratorMethod(osim.Manager.IntegratorMethod_CPodes)
        self.manager.setIntegratorAccuracy(self.accuracy)
        self.manager.setWriteToStorage(self.visualize)
        self.manager.setPerformAnalyses(self.visualize)

        # Observations.
        # -------------
        # Check that the requested observations are valid.
        for obs in observation_list:
            if obs not in DEFAULT_OBSERVATION_LIST:
                raise ValueError(f"Invalid observation: {obs}")

        # Store the observation keys.
        self.observation_list = {obs: obs in observation_list
                                 for obs in DEFAULT_OBSERVATION_LIST}

        self.obs = self.calc_observations()
        self.prev_obs = self.calc_observations()

    def get_num_controls(self):
        return self.num_controls

    def get_num_activations(self):
        return self.num_activations

    def get_num_bodies(self):
        return self.model.getNumBodies()

    def get_num_mobilities(self):
        return self.state.getNU()

    def get_num_coordinates(self):
        # TODO check if qdot =/= u
        return self.model.getNumCoordinates()

    def get_body_weight(self):
        mass = self.model.getTotalMass(self.state)
        gravity = self.model.getGravity()
        return mass * np.linalg.norm(gravity.to_numpy())

    def actuate(self, action):
        if np.any(np.isnan(action)):
            raise ValueError("Action contains NaN values.")

        # TODO: clip actions?
        range = zip(self.min_controls, self.max_controls)
        for ic, (c_min, c_max) in enumerate(range):
            self.controls[ic] = 0.5*(c_max - c_min) +  \
                                0.5*(c_max + c_min)*action[ic].item()

        self.controller.setDiscreteControls(self.state, self.controls)

    def step(self):
        self.manager.initialize(self.state)
        self.istep = self.istep + 1
        self.state = self.manager.integrate(self.step_size * self.istep)
        self.prev_obs = self.obs
        self.obs = self.calc_observations()

    def reset(self):
        self.state = self.model.initializeState()
        # TODO self.model.equilibrateMuscles(self.state)
        self.state.setTime(0.0)
        self.istep = 0

    def get_activations(self, state):
        activations = np.zeros((self.get_num_activations(),), dtype=np.float32)
        iact = 0
        for muscle in self.activation_muscles:
            activations[iact] = muscle.getActivation(state)
            iact += 1

        for actuator in self.activation_coordinate_actuators:
            activations[iact] = actuator.getActivation(state)
            iact += 1

        return activations

    def get_observation_space(self):
        obs_dict = {}

        # coordinate kinematics
        if self.observation_list["coordinate_kinematics"]:
            obs_dict["coordinate_values"] = gym.spaces.Box(low=-1, high=1,
                    shape=(self.get_num_coordinates(),), dtype=np.float32)
            obs_dict["coordinate_speeds"] = gym.spaces.Box(low=-1, high=1,
                    shape=(self.get_num_coordinates(),), dtype=np.float32)

        # body kinematics
        if self.observation_list['body_kinematics']:
            obs_dict["body_positions"] = gym.spaces.Box(low=-1, high=1,
                    shape=(self.get_num_bodies(), 3), dtype=np.float32)
            obs_dict["body_velocities"] = gym.spaces.Box(low=-1, high=1,
                    shape=(self.get_num_bodies(), 3), dtype=np.float32)
            obs_dict["body_accelerations"] = gym.spaces.Box(low=-1, high=1,
                    shape=(self.get_num_bodies(), 3), dtype=np.float32)
            obs_dict["body_orientations"] = gym.spaces.Box(low=-1, high=1,
                    shape=(self.get_num_bodies(), 3), dtype=np.float32)
            obs_dict["body_angular_velocities"] = gym.spaces.Box(low=-1, high=1,
                    shape=(self.get_num_bodies(), 3), dtype=np.float32)
            obs_dict["body_angular_accelerations"] = gym.spaces.Box(low=-1, high=1,
                    shape=(self.get_num_bodies(), 3), dtype=np.float32)

        # center of mass kinematics
        if self.observation_list["center_of_mass_kinematics"]:
            obs_dict["center_of_mass_position"] = gym.spaces.Box(low=-1, high=1,
                    shape=(3,), dtype=np.float32)
            obs_dict["center_of_mass_velocity"] = gym.spaces.Box(low=-1, high=1,
                    shape=(3,), dtype=np.float32)
            obs_dict["center_of_mass_acceleration"] = gym.spaces.Box(low=-1, high=1,
                    shape=(3,), dtype=np.float32)

        # whole body momentum
        if self.observation_list["whole_body_momentum"]:
            obs_dict["whole_body_linear_momentum"] = gym.spaces.Box(low=-1, high=1,
                    shape=(3,), dtype=np.float32)
            obs_dict["whole_body_angular_momentum"] = gym.spaces.Box(low=-1, high=1,
                    shape=(3,), dtype=np.float32)

        # controls
        if self.observation_list["controls"]:
            obs_dict["controls"] = gym.spaces.Box(low=-1, high=1,
                    shape=(self.get_num_controls(),), dtype=np.float32)

        # activations
        if self.observation_list["activations"]:
            obs_dict["activations"] = gym.spaces.Box(low=-1, high=1,
                    shape=(self.get_num_activations(),), dtype=np.float32)

        # forces
        for force_class in self.force_list:
            obs_dict[f'{force_class}_generalized_forces'] = gym.spaces.Box(low=-1, high=1,
                    shape=(self.get_num_mobilities(),), dtype=np.float32)
            obs_dict[f'{force_class}_body_torque'] = gym.spaces.Box(low=-1, high=1,
                    shape=(3,), dtype=np.float32)
            obs_dict[f'{force_class}_body_force'] = gym.spaces.Box(low=-1, high=1,
                    shape=(3,), dtype=np.float32)

        return gym.spaces.Dict(obs_dict)

    def get_observations(self):
        return self.obs

    def get_previous_observations(self):
        return self.prev_obs

    def calc_observations(self):
        self.model.realizeAcceleration(self.state)

        # coordinate kinematics
        if self.observation_list["coordinate_kinematics"]:
            self.obs["coordinate_values"] = np.zeros((self.get_num_coordinates(),),
                                                dtype=np.float32)
            self.obs["coordinate_speeds"] = np.zeros((self.get_num_coordinates(),),
                                                dtype=np.float32)
            coordinate_set = self.model.getCoordinateSet()
            for i in range(coordinate_set.getSize()):
                coord = coordinate_set.get(i)
                self.obs["coordinate_values"][i] = coord.getValue(self.state)
                self.obs["coordinate_speeds"][i] = coord.getSpeedValue(self.state)

        # body kinematics
        if self.observation_list['body_kinematics']:
            self.obs["body_positions"] = np.zeros((self.get_num_bodies(),3),
                                                dtype=np.float32)
            self.obs["body_velocities"] = np.zeros((self.get_num_bodies(),3),
                                                dtype=np.float32)
            self.obs["body_accelerations"] = np.zeros((self.get_num_bodies(),3),
                                                dtype=np.float32)
            self.obs["body_orientations"] = np.zeros((self.get_num_bodies(),3),
                                                dtype=np.float32)
            self.obs["body_angular_velocities"] = np.zeros((self.get_num_bodies(),3),
                                                dtype=np.float32)
            self.obs["body_angular_accelerations"] = np.zeros((self.get_num_bodies(),3),
                                                dtype=np.float32)
            body_set = self.model.getBodySet()
            for i in range(body_set.getSize()):
                body = body_set.get(i)
                transform = body.getTransformInGround(self.state)
                velocity = body.getVelocityInGround(self.state)
                acceleration = body.getAccelerationInGround(self.state)

                self.obs["body_positions"][i] = transform.p().to_numpy()
                self.obs["body_velocities"][i] = velocity.get(1).to_numpy()
                self.obs["body_accelerations"][i] = acceleration.get(1).to_numpy()

                self.obs["body_orientations"][i] = \
                        transform.R().convertRotationToBodyFixedXYZ().to_numpy()
                self.obs["body_angular_velocities"][i] = velocity.get(0).to_numpy()
                self.obs["body_angular_accelerations"][i] = acceleration.get(0).to_numpy()

        # center of mass kinematics
        if self.observation_list["center_of_mass_kinematics"]:
            self.obs["center_of_mass_position"] = np.array(
                    self.model.calcMassCenterPosition(self.state).to_numpy(),
                    dtype=np.float32)
            self.obs["center_of_mass_velocity"] = np.array(
                    self.model.calcMassCenterVelocity(self.state).to_numpy(),
                    dtype=np.float32)
            self.obs["center_of_mass_acceleration"] = np.array(
                    self.model.calcMassCenterAcceleration(self.state).to_numpy(),
                    dtype=np.float32)

        # whole body momentum
        if self.observation_list["whole_body_momentum"]:
            self.obs["whole_body_linear_momentum"] = np.array(
                    self.model.calcLinearMomentum(self.state).to_numpy(),
                    dtype=np.float32)
            self.obs["whole_body_angular_momentum"] = np.array(
                    self.model.calcAngularMomentum(self.state).to_numpy(),
                    dtype=np.float32)

        # controls
        if self.observation_list["controls"]:
            controls = self.controller.getDiscreteControls(self.state).to_numpy()
            self.obs["controls"] = np.array(controls, dtype=np.float32)

        # activations
        if self.observation_list["activations"]:
            self.obs["activations"] = self.get_activations(self.state)

        # forces
        for force_class in self.force_list:
            force_aggregator = osim.ForceAggregator.safeDownCast(
                self.model.getComponent(f'{force_class}_aggregator'))
            # TODO: summing body forces but not mobility forces
            generalized_forces = force_aggregator.getGeneralizedForces(self.state)
            body_force = force_aggregator.getBodyForcesSum(self.state)
            self.obs[f'{force_class}_generalized_forces'] = \
                    np.array(generalized_forces.to_numpy(), dtype=np.float32)
            self.obs[f'{force_class}_body_torque'] = \
                    np.array(body_force.get(0).to_numpy(), dtype=np.float32)
            self.obs[f'{force_class}_body_force'] = \
                    np.array(body_force.get(1).to_numpy(), dtype=np.float32)

        return self.obs

    def get_scaled_observations(self):

        obs = deepcopy(self.obs)

        # coordinate kinematics
        if self.observation_list["coordinate_kinematics"]:
            obs["coordinate_values"] /= OBSERVATION_SCALES["coordinate_values"]
            obs["coordinate_speeds"] /= OBSERVATION_SCALES["coordinate_speeds"]

        # body kinematics
        if self.observation_list['body_kinematics']:
            obs["body_positions"] /= OBSERVATION_SCALES["body_positions"]
            obs["body_velocities"] /= OBSERVATION_SCALES["body_velocities"]
            obs["body_accelerations"] /= OBSERVATION_SCALES["body_accelerations"]
            obs["body_orientations"] /= OBSERVATION_SCALES["body_orientations"]
            obs["body_angular_velocities"] /= \
                OBSERVATION_SCALES["body_angular_velocities"]
            obs["body_angular_accelerations"] /= \
                OBSERVATION_SCALES["body_angular_accelerations"]

        # center of mass kinematics
        if self.observation_list["center_of_mass_kinematics"]:
            obs["center_of_mass_position"] /= \
                OBSERVATION_SCALES["center_of_mass_position"]
            obs["center_of_mass_velocity"] /= \
                OBSERVATION_SCALES["center_of_mass_velocity"]
            obs["center_of_mass_acceleration"] /= \
                OBSERVATION_SCALES["center_of_mass_acceleration"]

        # whole body momentum
        if self.observation_list["whole_body_momentum"]:
            obs["whole_body_linear_momentum"] /= \
                OBSERVATION_SCALES["whole_body_linear_momentum"]
            obs["whole_body_angular_momentum"] /= \
                OBSERVATION_SCALES["whole_body_angular_momentum"]

        # controls
        if self.observation_list["controls"]:
            obs["controls"] /= OBSERVATION_SCALES["controls"]

        # activations
        if self.observation_list["activations"]:
            obs["activations"] /= OBSERVATION_SCALES["activations"]

        # force
        for force_class in self.force_list:
            obs[f'{force_class}_generalized_forces'] /= self.force_scales[force_class]
            obs[f'{force_class}_body_torque'] /= self.force_scales[force_class]
            obs[f'{force_class}_body_force'] /= self.force_scales[force_class]

        return obs
