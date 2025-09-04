from xml.parsers.expat import model
import opensim as osim
import numpy as np
import gymnasium as gym
from copy import deepcopy

DEFAULT_OBSERVATION_LIST = [
    "coordinate_values",
    "coordinate_speeds",
    "body_positions",
    "body_velocities",
    "body_accelerations",
    "body_orientations",
    "body_angular_velocities",
    "body_angular_accelerations",
    "center_of_mass_position",
    "center_of_mass_velocity",
    "center_of_mass_acceleration",
    "whole_body_linear_momentum",
    "whole_body_angular_momentum",
    "controls",
    "activations"
]

OBSERVATION_SCALES = {
    "coordinate_values": 2.0*np.pi,
    "coordinate_speeds": 20.0*np.pi,
    "body_positions": 5.0,
    "body_velocities": 20.0,
    "body_accelerations": 200.0,
    "body_orientations": 5.0*np.pi,
    "body_angular_velocities": 50.0*np.pi,
    "body_angular_accelerations": 500.0*np.pi,
    "center_of_mass_position": 5.0,
    "center_of_mass_velocity": 50.0,
    "center_of_mass_acceleration": 500.0,
    "whole_body_linear_momentum": 1000.0,
    "whole_body_angular_momentum": 1000.0,
    "controls": 1.0,
    "activations": 1.0
}

class OpenSimModel:
    """
    A class to store an OpenSim model and convenient methods for reinforcement learning.
    """
    outputs = dict()
    previous_outputs = dict()

    def __init__(self, model_filepath, visualize, accuracy, step_size,
                 observation_list=DEFAULT_OBSERVATION_LIST,
                 aggregators=dict(), aggregator_scales=dict()):

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
        # TODO: find a cleaner way to do this
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
        self.aggregators = aggregators
        self.aggregator_scales = aggregator_scales
        for key in aggregators.keys():
            force_aggregator = osim.ForceAggregator()
            force_aggregator.setName(f'{key}_aggregator')
            for force_path in aggregators[key]:
                force_aggregator.addForce(self.model.getComponent(force_path))
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

        self.outputs = self.calc_outputs()
        self.previous_outputs = self.calc_outputs()

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
        self.previous_outputs = self.outputs.copy()
        self.outputs = self.calc_outputs()

    def reset(self):
        self.state = self.model.initializeState()
        # TODO self.model.equilibrateMuscles(self.state)
        self.state.setTime(0.0)
        self.istep = 0
        self.outputs = self.calc_outputs()
        self.previous_outputs = self.calc_outputs()

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
        return gym.spaces.Box(
            low=-1, high=1, shape=self.get_observations().shape, dtype=np.float32
        )

    def calc_outputs(self):
        outputs = dict()
        self.model.realizeAcceleration(self.state)

        # coordinate kinematics
        outputs["coordinate_values"] = np.zeros((self.get_num_coordinates(),),
                                                dtype=np.float32)
        outputs["coordinate_speeds"] = np.zeros((self.get_num_coordinates(),),
                                            dtype=np.float32)
        coordinate_set = self.model.getCoordinateSet()
        for i in range(coordinate_set.getSize()):
            coord = coordinate_set.get(i)
            if self.observation_list["coordinate_values"]:
                outputs["coordinate_values"][i] = coord.getValue(self.state)
            if self.observation_list["coordinate_speeds"]:
                outputs["coordinate_speeds"][i] = coord.getSpeedValue(self.state)

        # body kinematics
        outputs["body_positions"] = np.zeros((self.get_num_bodies(), 3),
                dtype=np.float32)
        outputs["body_velocities"] = np.zeros((self.get_num_bodies(), 3),
                dtype=np.float32)
        outputs["body_accelerations"] = np.zeros((self.get_num_bodies(), 3),
                dtype=np.float32)
        outputs["body_orientations"] = np.zeros((self.get_num_bodies(), 3),
                dtype=np.float32)
        outputs["body_angular_velocities"] = np.zeros((self.get_num_bodies(), 3),
                dtype=np.float32)
        outputs["body_angular_accelerations"] = np.zeros((self.get_num_bodies(), 3),
                dtype=np.float32)
        body_set = self.model.getBodySet()
        for i in range(body_set.getSize()):
            body = body_set.get(i)
            transform = body.getTransformInGround(self.state)
            velocity = body.getVelocityInGround(self.state)
            acceleration = body.getAccelerationInGround(self.state)

            outputs["body_positions"][i] = transform.p().to_numpy()
            outputs["body_velocities"][i] = velocity.get(1).to_numpy()
            outputs["body_accelerations"][i] = acceleration.get(1).to_numpy()

            outputs["body_orientations"][i] = \
                    transform.R().convertRotationToBodyFixedXYZ().to_numpy()
            outputs["body_angular_velocities"][i] = velocity.get(0).to_numpy()
            outputs["body_angular_accelerations"][i] = \
                    acceleration.get(0).to_numpy()

        # center of mass kinematics
        outputs["center_of_mass_position"] = np.array(
                self.model.calcMassCenterPosition(self.state).to_numpy(),
                dtype=np.float32)
        outputs["center_of_mass_velocity"] = np.array(
                self.model.calcMassCenterVelocity(self.state).to_numpy(),
                dtype=np.float32)
        outputs["center_of_mass_acceleration"] = np.array(
                self.model.calcMassCenterAcceleration(self.state).to_numpy(),
                dtype=np.float32)

        # whole body momentum
        outputs["whole_body_linear_momentum"] = np.array(
                self.model.calcLinearMomentum(self.state).to_numpy(),
                dtype=np.float32)
        outputs["whole_body_angular_momentum"] = np.array(
                self.model.calcAngularMomentum(self.state).to_numpy(),
                dtype=np.float32)

        # controls
        controls = self.controller.getDiscreteControls(self.state).to_numpy()
        outputs["controls"] = np.array(controls, dtype=np.float32)

        # activations
        outputs["activations"] = self.get_activations(self.state)

        # forces
        for key in self.aggregators:
            outputs[f'{key}_generalized_forces'] = np.zeros(
                    (self.get_num_mobilities(),), dtype=np.float32)
            outputs[f'{key}_body_torques'] = np.zeros(
                    (self.get_num_bodies(), 3), dtype=np.float32)
            outputs[f'{key}_body_forces'] = np.zeros(
                    (self.get_num_bodies(), 3), dtype=np.float32)

            force_aggregator = osim.ForceAggregator.safeDownCast(
                self.model.getComponent(f'{key}_aggregator'))
            generalized_forces = force_aggregator.getGeneralizedForces(self.state)
            body_forces = force_aggregator.getBodyForces(self.state)
            outputs[f'{key}_generalized_forces'] = \
                    np.array(generalized_forces.to_numpy(), dtype=np.float32)
            for ibody in range(self.get_num_bodies()):
                outputs[f'{key}_body_torques'][ibody] = \
                        body_forces.get(ibody).get(0).to_numpy()
                outputs[f'{key}_body_forces'][ibody] = \
                        body_forces.get(ibody).get(1).to_numpy()

        return outputs

    def get_outputs(self):
        return self.outputs

    def get_previous_outputs(self):
        return self.previous_outputs

    def get_observations(self):
        outputs = self.get_outputs()
        obs = list()

        # coordinate kinematics
        if self.observation_list["coordinate_values"]:
            obs.append(outputs["coordinate_values"])
            obs[-1] /= OBSERVATION_SCALES["coordinate_values"]
        if self.observation_list["coordinate_speeds"]:
            obs.append(outputs["coordinate_speeds"])
            obs[-1] /= OBSERVATION_SCALES["coordinate_speeds"]

        # body kinematics
        if self.observation_list['body_positions']:
            obs.append(np.reshape(outputs["body_positions"],
                                  (3*self.get_num_bodies(),)))
            obs[-1] /= OBSERVATION_SCALES["body_positions"]
        if self.observation_list['body_velocities']:
            obs.append(np.reshape(outputs["body_velocities"],
                                  (3*self.get_num_bodies(),)))
            obs[-1] /= OBSERVATION_SCALES["body_velocities"]
        if self.observation_list['body_accelerations']:
            obs.append(np.reshape(outputs["body_accelerations"],
                                  (3*self.get_num_bodies(),)))
            obs[-1] /= OBSERVATION_SCALES["body_accelerations"]
        if self.observation_list['body_orientations']:
            obs.append(np.reshape(outputs["body_orientations"],
                                  (3*self.get_num_bodies(),)))
            obs[-1] /= OBSERVATION_SCALES["body_orientations"]
        if self.observation_list['body_angular_velocities']:
            obs.append(np.reshape(outputs["body_angular_velocities"],
                                  (3*self.get_num_bodies(),)))
            obs[-1] /= OBSERVATION_SCALES["body_angular_velocities"]
        if self.observation_list['body_angular_accelerations']:
            obs.append(np.reshape(outputs["body_angular_accelerations"],
                                  (3*self.get_num_bodies(),)))
            obs[-1] /= OBSERVATION_SCALES["body_angular_accelerations"]

        # center of mass kinematics
        if self.observation_list["center_of_mass_position"]:
            obs.append(outputs["center_of_mass_position"])
            obs[-1] /= OBSERVATION_SCALES["center_of_mass_position"]
        if self.observation_list["center_of_mass_velocity"]:
            obs.append(outputs["center_of_mass_velocity"])
            obs[-1] /= OBSERVATION_SCALES["center_of_mass_velocity"]
        if self.observation_list["center_of_mass_acceleration"]:
            obs.append(outputs["center_of_mass_acceleration"])
            obs[-1] /= OBSERVATION_SCALES["center_of_mass_acceleration"]

        # whole body momentum
        if self.observation_list["whole_body_linear_momentum"]:
            obs.append(outputs["whole_body_linear_momentum"])
            obs[-1] /= OBSERVATION_SCALES["whole_body_linear_momentum"]
        if self.observation_list["whole_body_angular_momentum"]:
            obs.append(outputs["whole_body_angular_momentum"])
            obs[-1] /= OBSERVATION_SCALES["whole_body_angular_momentum"]

        # controls
        if self.observation_list["controls"]:
            obs.append(outputs["controls"])
            obs[-1] /= OBSERVATION_SCALES["controls"]

        # activations
        if self.observation_list["activations"]:
            obs.append(outputs["activations"])
            obs[-1] /= OBSERVATION_SCALES["activations"]

        # force
        for key in self.aggregators:
            obs.append(outputs[f'{key}_generalized_forces'])
            obs[-1] /= self.aggregator_scales[key]
            obs.append(np.reshape(outputs[f'{key}_body_torques'],
                                  (3*self.get_num_bodies(),)))
            obs[-1] /= self.aggregator_scales[key]
            obs.append(np.reshape(outputs[f'{key}_body_forces'],
                                  (3*self.get_num_bodies(),)))
            obs[-1] /= self.aggregator_scales[key]

        return np.concatenate(obs, dtype=np.float32).copy()
