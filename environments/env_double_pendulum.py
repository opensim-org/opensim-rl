import os
import opensim as osim
import numpy as np
import gymnasium as gym

class osimModel:

    stepsize = 0.0025

    model  = None
    state  = None
    state0 = None
    joints = []
    bodies = []
    brain  = None

    verbose = False

    istep = 0

    state_desc_istep = None
    prev_state_desc  = None
    state_desc       = None
    
    integrator_accuracy = None

    def __init__(self, model_path, visualize=False, integrator_accuracy=1e-5):
        self.integrator_accuracy = integrator_accuracy
        self.model = osim.Model(model_path)
        self.model_state = self.model.initSystem()
        self.state_names = self.model.getStateVariableNames()
        self.brain = osim.PrescribedController()

        self.model.setUseVisualizer(visualize)

        self.actuatorSet = self.model.getActuators()
        self.bodySet  = self.model.getBodySet()
        self.jointSet = self.model.getJointSet()

        for j in range(self.actuatorSet.getSize()):
            func = osim.Constant(0.5)
            self.brain.addActuator(self.actuatorSet.get(j))
            self.brain.prescribeControlForActuator(self.actuatorSet.get(j).getName(), func)

        self.noutput = self.actuatorSet.getSize()

        self.model.addController(self.brain)

        self.model_state = self.model.initSystem()

    def actuate(self, action):
        if np.any(np.isnan(action)):
            raise ValueError("Action contains NaN values.")
        
        # forces the action to be in the range of -1 to 1
        action = np.clip(action, -1.0, 1.0)

        brain = osim.PrescribedController.safeDownCast(self.model.getControllerSet().get(0))
        functionSet = brain.get_ControlFunctions()

        for j in range(functionSet.getSize()):
            func = osim.Constant.safeDownCast(functionSet.get(j))
            func.setValue(action[j].item())

    def compute_state_desc(self):
        self.model.realizeAcceleration(self.state)

        res = {}

        ## joints
        res["joint_pos"] = {}
        res["joint_vel"] = {}
        res["joint_acc"] = {}
        for i in range(self.jointSet.getSize()):
            joint = self.jointSet.get(i)
            name = joint.getName()
            res["joint_pos"][name] = [joint.get_coordinates(j).getValue(self.state) for j in range(joint.numCoordinates())]
            res["joint_vel"][name] = [joint.get_coordinates(j).getSpeedValue(self.state) for j in range(joint.numCoordinates())]
            res["joint_acc"][name] = [joint.get_coordinates(j).getAccelerationValue(self.state) for j in range(joint.numCoordinates())]

        ## bodies
        res["body_pos"] = {}
        res["body_vel"] = {}
        res["body_acc"] = {}
        res["body_pos_rot"] = {}
        res["body_vel_rot"] = {}
        res["body_acc_rot"] = {}
        for i in range(self.bodySet.getSize()):
            body = self.bodySet.get(i)
            name = body.getName()
            res["body_pos"][name] = [body.getTransformInGround(self.state).p()[i] for i in range(3)]
            res["body_vel"][name] = [body.getVelocityInGround(self.state).get(1).get(i) for i in range(3)]
            res["body_acc"][name] = [body.getAccelerationInGround(self.state).get(1).get(i) for i in range(3)]
            
            res["body_pos_rot"][name] = [body.getTransformInGround(self.state).R().convertRotationToBodyFixedXYZ().get(i) for i in range(3)]
            res["body_vel_rot"][name] = [body.getVelocityInGround(self.state).get(0).get(i) for i in range(3)]
            res["body_acc_rot"][name] = [body.getAccelerationInGround(self.state).get(0).get(i) for i in range(3)]

        ## actuators
        res["controls"] = {}
        for i in range(self.actuatorSet.getSize()):
            actuator = self.actuatorSet.get(i)
            opt_force = osim.CoordinateActuator.safeDownCast(actuator).get_optimal_force()
            name = actuator.getName()
            res["controls"][name] = actuator.getRecordValues(self.state).get(0) / opt_force

        return res
    
    def get_state_desc(self):
        if self.state_desc_istep != self.istep:
            self.prev_state_desc = self.state_desc
            self.state_desc = self.compute_state_desc()
            self.state_desc_istep = self.istep
        return self.state_desc
    
    def get_action_space_size(self):
        return self.noutput
    
    def reset_manager(self):
        self.manager = osim.Manager(self.model)
        self.manager.setIntegratorAccuracy(self.integrator_accuracy)
        self.manager.initialize(self.state)

    def reset(self, pos0=None):
        self.state = self.model.initializeState()

        if pos0 is None:
            self.model.setStateVariableValue(self.state, '/jointset/pin1/q1/value', np.pi/4)
            self.model.setStateVariableValue(self.state, '/jointset/pin2/q2/value', -np.pi/2)

        else:
            self.model.setStateVariableValue(self.state, '/jointset/pin1/q1/value', pos0["q1"].item())
            self.model.setStateVariableValue(self.state, '/jointset/pin2/q2/value', pos0["q2"].item())

        self.state.setTime(0.0)
        self.istep = 0
        self.reset_manager()

        print(f"Initial reset state: {self.model.getStateVariableValues(self.state)}")

    def integrate(self):
        self.istep = self.istep + 1
        self.state = self.manager.integrate(self.stepsize * self.istep)

    def show_visualizer(self):
        visual = self.model.updVisualizer()
        visual.show(self.state)

        #visual = self.model.updVisualizer().updSimboxVisualizer()
        #visual.setShowSimTime(True)
        #visual.zoomCameraToShowAllGeometry()
        #visual.setCameraTransform(osim.Transform( osim.Vec3(0,1,1) ))
        #visual.setCameraFieldOfView(1.7)
        #self.model.getVisualizer().show(self.state)

class osimEnv(gym.Env):
    action_space = None
    observation_space = None
    osim_model = None
    istep = 0
    verbose = False

    visualize = False
    spec = None
    time_limit = 1.0

    prev_state_desc = None

    model_path = None

    metadata = {
        'render_modes': ['human'],
        'video.frames_per_second' : None
    }

    def get_reward(self):
        raise NotImplementedError
    
    def __init__(self, visualize=False, integrator_accuracy=1e-5, model_path=None):
        self.visualize = visualize
        self.integrator_accuracy = integrator_accuracy
        if model_path is not None:
            self.load_model(model_path)
    
    def load_model(self, model_path=None):
        if model_path:
            self.model_path = model_path

        self.osim_model = osimModel(self.model_path,self.visualize,self.integrator_accuracy)

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.osim_model.get_action_space_size(),), dtype=np.float32)

    def get_state_desc(self):
        return self.osim_model.get_state_desc()
    
    def get_prev_state_desc(self):
        return self.osim_model.prev_state_desc
    
    def get_observation(self):
        return self.osim_model.get_state_desc()
    
    def get_observation_dict(self):
        return self.osim_model.get_state_desc()
    
    def get_observation_space_size(self):
        return 0
    
    def get_action_space_size(self):
        return self.osim_model.get_action_space_size()
    
    def reset(self):
        self.osim_model.reset(pos0={"q1": 0.0, "q2": 0.0})

        return self.get_observation()
    
    def get_time(self):
        return self.osim_model.state.getTime()
    
    def get_istep(self):
        return self.osim_model.istep
    
    def _get_obs(self):
        NotImplementedError("This method should be implemented in the subclass.")
    
    def _get_info(self):
        NotImplementedError("This method should be implemented in the subclass.")
    
    def step(self, action):
        self.prev_state_desc = self.get_state_desc()
        self.osim_model.actuate(action)
        self.osim_model.integrate()
        if self.visualize == True:
            self.osim_model.show_visualizer()

        obs = self.get_observation()

        if self.get_time() < self.time_limit:
            self.is_done = False
        else:
            self.is_done = True

        truncated = False

        return self._get_obs(), self.get_reward(), self.is_done, truncated, self._get_info()


class doublePendulumEnv(osimEnv):

    current_dir = os.getcwd()
    model_name = "double_pendulum.osim"
    model_path = os.path.join(current_dir,"msk_models", model_name)

    target_q1 = np.pi
    target_q2 = 0.0

    print(f"Loading model from: {model_path}")

    def __init__(self, visualize=False, rand_pos0=False):
        super().__init__(visualize=visualize, model_path=self.model_path)

        self.rand_pos0 = rand_pos0

        self.observation_space = gym.spaces.Dict({
            "angles": gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "velocities": gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "body_pos": gym.spaces.Box(low=-1, high=1, shape=(2*2,), dtype=np.float32),
            "controls": gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "target_q1_diff": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            "target_q2_diff": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        })

    def get_observation(self):

        state_desc = self.osim_model.get_state_desc()

        res = {
        "angles": np.squeeze(np.array([state_desc["joint_pos"]["pin1"],
                                       state_desc["joint_pos"]["pin2"]], dtype=np.float32), axis=1) / (2*np.pi),

        "velocities": np.squeeze(np.array([state_desc["joint_vel"]["pin1"],
                                           state_desc["joint_vel"]["pin2"]], dtype=np.float32), axis=1) / 50.0,
                                           
        "body_pos": np.array([state_desc["body_pos"]["rod1"][0], state_desc["body_pos"]["rod1"][1],
                                         state_desc["body_pos"]["rod2"][0], state_desc["body_pos"]["rod2"][1]], dtype=np.float32) / 6.0,

        "controls": np.array([state_desc["controls"][actuator_name]
                                            for actuator_name in sorted(state_desc["controls"].keys())], dtype=np.float32),

        "target_q1_diff": np.array([self.target_q1 - state_desc["joint_pos"]["pin1"][0]], dtype=np.float32) / 6.0,

        "target_q2_diff": np.array([self.target_q2 - state_desc["joint_pos"]["pin2"][0]], dtype=np.float32) / 6.0
        }

        # check whether the observations in the res dict contain NaN values
        for key, value in res.items():
            if isinstance(value, np.ndarray) and np.isnan(value).any():
                raise ValueError(f"Observation {key} contains NaN values: {value}")        
        
        return res
    
    def _get_obs(self):
        return self.get_observation()
    
    def _get_info(self):
        return {}
    
    def get_observation_space_size(self):
        obs_size = 0

        obs_size += self.osim_model.jointSet.getSize() * 2  # joint positions, velocities, accelerations
        obs_size += self.osim_model.bodySet.getSize() * 2  # body positions
        obs_size += self.osim_model.actuatorSet.getSize()  # actuator controls
        obs_size += 2  # target_x and target_y    

        return obs_size
    
    def get_observation_space(self):
        return self.observation_space
    
    def get_reward(self):
        state_desc = self.get_state_desc()

        diff_q1 = state_desc["joint_pos"]["pin1"][0] - self.target_q1
        diff_q2 = state_desc["joint_pos"]["pin2"][0] - self.target_q2

        reward_angle_diff = 1/((diff_q1 ** 2 + diff_q2 ** 2) + 0.01)

        controls_array = np.array(list(state_desc["controls"].values()))
        reward_controls = -np.sum(np.square(controls_array))

        return reward_angle_diff + 0.01*reward_controls

    def reset(self, seed=None, options=None):

        self.np_random, seed = gym.utils.seeding.np_random(seed)

        if self.rand_pos0:
            rand_q = self.np_random.uniform(low=-0.14,high=0.14, size=(2,)).astype(np.float32)
            pos0 = {"q1": rand_q[0] + np.pi/4, "q2": rand_q[1] - np.pi/2}
            self.osim_model.reset(pos0=pos0)
        else:
            self.osim_model.reset()

        return self._get_obs(), self._get_info()
