import opensim as osim
import numpy as np
import gymnasium as gym

DEFAULT_OBSERVATIONS = [
    "coordinate_kinematics",
    "body_kinematics",
    "center_of_mass_kinematics",
    "whole_body_momentum",
    "controls",
]

OBSERVATION_SCALES = {
    "coordinate_values": 2.0*np.pi,
    "coordinate_speeds": 20.0*np.pi,
    "body_positions": 5.0,
    "body_velocities": 5.0,
    "body_accelerations": 5.0,
    "body_orientations": 2.0*np.pi,
    "body_angular_velocities": 20.0*np.pi,
    "body_angular_accelerations": 200.0*np.pi,
    "center_of_mass_positions": 5.0,
    "center_of_mass_velocities": 25.0,
    "center_of_mass_accelerations": 100.0,
    "whole_body_linear_momentum": 20.0,
    "whole_body_angular_momentum": 20.0,
    "controls": 1.0,
}

class OpenSimModel:
    """
    A class to store an OpenSim model and convenient methods for reinforcement learning.
    """

    def __init__(self, model_filepath, visualize, accuracy, step_size,
                 observations=DEFAULT_OBSERVATIONS):

        # Initialize the OpenSim model.
        self.model = osim.Model(model_filepath)
        self.model.initSystem()
        self.visualize = visualize
        self.model.setUseVisualizer(self.visualize)

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
        self.ncontrols = self.model.getNumControls()
        assert(len(self.min_controls) == self.ncontrols)
        assert(len(self.max_controls) == self.ncontrols)
        self.controls = osim.Vector(self.ncontrols, 0.0)

        # Add a DiscreteController to the model. This allows us to update the controls
        # within the SimTK::State object.
        self.controller = osim.DiscreteController()
        self.controller.setName('brain')
        self.model.addController(self.controller)

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
        for obs in observations:
            if obs not in DEFAULT_OBSERVATIONS:
                raise ValueError(f"Invalid observation: {obs}")

        # Store the observation keys.
        self.observations = {obs: obs in observations for obs in DEFAULT_OBSERVATIONS}


    def get_num_controls(self):
        return self.ncontrols

    def get_num_bodies(self):
        return self.model.getNumBodies()

    def get_num_coordinates(self):
        # TODO check if qdot =/= u
        return self.model.getNumCoordinates()

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

    def reset(self):
        self.state = self.model.initializeState()
        # TODO self.model.equilibrateMuscles(self.state)
        self.state.setTime(0.0)
        self.istep = 0

    # def get_observation_space(self):

    #     obs_dict = {}

    #     # coordinate kinematics
    #     if self.observations["coordinate_kinematics"]:
    #         obs_dict["coordinate_values"] = gym.spaces.Box(low=-1, high=1,
    #                 shape=(self.get_num_coordinates,), dtype=np.float32)
    #         obs_dict["coordinate_speeds"] = gym.spaces.Box(low=-1, high=1,
    #                 shape=(self.get_num_coordinates,), dtype=np.float32)

    #     # body kinematics
    #     if self.observations['body_kinematics']:


    #     # center of mass kinematics
    #     if self.observations["center_of_mass_kinematics"]:


    #     # whole body momentum
    #     if self.observations["whole_body_momentum"]:


    #     # actuator controls
    #     if self.observations["controls"]:
    #         obs["controls"] = {}
    #         controls = self.controller.getDiscreteControls(self.state).to_numpy()
    #         for i in range(len(controls)):
    #             obs["controls"][f"control_{i}"] = controls[i]

    #     return gym.spaces.Dict(obs_dict)

    def compute_observations(self):
        self.model.realizeAcceleration(self.state)

        obs = {}

        # coordinate kinematics
        if self.observations["coordinate_kinematics"]:
            obs["coordinate_values"] = np.zeros((self.get_num_coordinates(),), dtype=np.float32)
            obs["coordinate_speeds"] = np.zeros((self.get_num_coordinates(),), dtype=np.float32)
            coordinate_set = self.model.getCoordinateSet()
            for i in range(coordinate_set.getSize()):
                coord = coordinate_set.get(i)
                name  = coord.getName()
                obs["coordinate_values"][coord.getAbsolutePathString() + '/value'] = \
                        coord.getValue(self.state)
                obs["coordinate_speeds"][coord.getAbsolutePathString() + '/speed'] = \
                        coord.getSpeedValue(self.state)

        # body kinematics
        if self.observations['body_kinematics']:
            obs["body_positions"] = {}
            obs["body_velocities"] = {}
            obs["body_accelerations"] = {}
            obs["body_orientations"] = {}
            obs["body_angular_velocities"] = {}
            obs["body_angular_accelerations"] = {}

            body_set = self.model.getBodySet()
            for i in range(body_set.getSize()):
                body = body_set.get(i)
                name = body.getName()
                transform = body.getTransformInGround(self.state)
                velocity = body.getVelocityInGround(self.state)
                acceleration = body.getAccelerationInGround(self.state)

                obs["body_positions"][name] = transform.p().to_numpy()
                obs["body_velocities"][name] = velocity.get(1).to_numpy()
                obs["body_accelerations"][name] = acceleration.get(1).to_numpy()

                obs["body_orientations"][name] = \
                        transform.R().convertRotationToBodyFixedXYZ().to_numpy()
                obs["body_angular_velocities"][name] = velocity.get(0).to_numpy()
                obs["body_angular_accelerations"][name] = acceleration.get(0).to_numpy()

        # center of mass kinematics
        if self.observations["center_of_mass_kinematics"]:
            obs["center_of_mass_position"] = \
                    self.model.calcMassCenterPosition(self.state).to_numpy()
            obs["center_of_mass_velocity"] = \
                    self.model.calcMassCenterVelocity(self.state).to_numpy()
            obs["center_of_mass_acceleration"] = \
                    self.model.calcMassCenterAcceleration(self.state).to_numpy()

        # whole body momentum
        if self.observations["whole_body_momentum"]:
            obs["whole_body_linear_momentum"] = \
                    self.model.calcLinearMomentum(self.state).to_numpy()
            obs["whole_body_angular_momentum"] = \
                    self.model.calcAngularMomentum(self.state).to_numpy()

        # actuator controls
        if self.observations["controls"]:
            obs["controls"] = {}
            controls = self.controller.getDiscreteControls(self.state).to_numpy()
            for ic, control_path in enumerate(self.control_paths):
                obs["controls"][control_path] = controls[ic]

        return obs

    def get_scaled_observations(self):
        obs = self.compute_observations()

        if self.observations["coordinate_kinematics"]:
            obs["coordinate_values"] /= OBSERVATION_SCALES["coordinate_values"]
            obs["coordinate_speeds"] /= OBSERVATION_SCALES["coordinate_speeds"]

        # body kinematics
        if self.observations['body_kinematics']:
            obs["body_positions"] /= OBSERVATION_SCALES["body_positions"]
            obs["body_velocities"] /= OBSERVATION_SCALES["body_velocities"]
            obs["body_accelerations"] /= OBSERVATION_SCALES["body_accelerations"]
            obs["body_orientations"] /= OBSERVATION_SCALES["body_orientations"]
            obs["body_angular_velocities"] /= \
                OBSERVATION_SCALES["body_angular_velocities"]
            obs["body_angular_accelerations"] /= \
                OBSERVATION_SCALES["body_angular_accelerations"]

        # center of mass kinematics
        if self.observations["center_of_mass_kinematics"]:
            obs["center_of_mass_position"] /= \
                OBSERVATION_SCALES["center_of_mass_position"]
            obs["center_of_mass_velocity"] /= \
                OBSERVATION_SCALES["center_of_mass_velocity"]
            obs["center_of_mass_acceleration"] /= \
                OBSERVATION_SCALES["center_of_mass_acceleration"]

        # whole body momentum
        if self.observations["whole_body_momentum"]:
            obs["whole_body_linear_momentum"] /= \
                OBSERVATION_SCALES["whole_body_linear_momentum"]
            obs["whole_body_angular_momentum"] /= \
                OBSERVATION_SCALES["whole_body_angular_momentum"]

        # actuator controls
        if self.observations["controls"]:
            obs["controls"] /= OBSERVATION_SCALES["controls"]

        return obs

    def get_observations(self):
        scaled_obs = self.get_scaled_observations()
        obs = dict()

        # coordinate kinematics
        if self.observations["coordinate_kinematics"]:
            obs["coordinate_values"] = np.zeros((self.get_num_coordinates(),),
                                                dtype=np.float32)
            for i, value in enumerate(scaled_obs["coordinate_values"]):
                import pdb; pdb.set_trace()
                obs["coordinate_values"][i] = value

            obs["coordinate_speeds"] = np.zeros((self.get_num_coordinates(),),
                                                dtype=np.float32)
            for i, speed in enumerate(scaled_obs["coordinate_speeds"]):
                obs["coordinate_speeds"][i] = speed

        # body kinematics
        if self.observations['body_kinematics']:
            obs["body_positions"] = np.zeros((self.get_num_bodies(), 3),
                                             dtype=np.float32)
            for i, pos in enumerate(scaled_obs["body_positions"]):
                obs["body_positions"][i] = pos

            obs["body_velocities"] = np.zeros((self.get_num_bodies(), 3),
                                              dtype=np.float32)
            for i, vel in enumerate(scaled_obs["body_velocities"]):
                obs["body_velocities"][i] = vel

            obs["body_accelerations"] = np.zeros((self.get_num_bodies(), 3),
                                                 dtype=np.float32)
            for i, acc in enumerate(scaled_obs["body_accelerations"]):
                obs["body_accelerations"][i] = acc

            obs["body_orientations"] = np.zeros((self.get_num_bodies(), 3),
                                                dtype=np.float32)
            for i, orientation in enumerate(scaled_obs["body_orientations"]):
                obs["body_orientations"][i] = orientation

            obs["body_angular_velocities"] = np.zeros((self.get_num_bodies(), 3),
                                                      dtype=np.float32)
            for i, vel in enumerate(scaled_obs["body_angular_velocities"]):
                obs["body_angular_velocities"][i] = vel

            obs["body_angular_accelerations"] = np.zeros((self.get_num_bodies(), 3),
                                                         dtype=np.float32)
            for i, acc in enumerate(scaled_obs["body_angular_accelerations"]):
                obs["body_angular_accelerations"][i] = acc

        # center of mass kinematics
        if self.observations["center_of_mass_kinematics"]:
            obs["center_of_mass_position"] = np.zeros((3,), dtype=np.float32)
            obs["center_of_mass_position"] = scaled_obs["center_of_mass_position"]

            obs["center_of_mass_velocity"] = np.zeros((3,), dtype=np.float32)
            for i, vel in enumerate(scaled_obs["center_of_mass_velocity"]):
                obs["center_of_mass_velocity"][i] = vel

            obs["center_of_mass_acceleration"] = np.zeros((3,), dtype=np.float32)
            for i, acc in enumerate(scaled_obs["center_of_mass_acceleration"]):
                obs["center_of_mass_acceleration"][i] = acc

        # whole body momentum
        if self.observations["whole_body_momentum"]:
            obs["whole_body_linear_momentum"] = np.zeros((3,), dtype=np.float32)
            for i, lin_mom in enumerate(scaled_obs["whole_body_linear_momentum"]):
                obs["whole_body_linear_momentum"][i] = lin_mom

            obs["whole_body_angular_momentum"] = np.zeros((3,), dtype=np.float32)
            for i, ang_mom in enumerate(scaled_obs["whole_body_angular_momentum"]):
                obs["whole_body_angular_momentum"][i] = ang_mom


        # actuator controls
        if self.observations["controls"]:
            pass

        return scaled_obs
