import os
import opensim as osim
import numpy as np
import gymnasium as gym
from utilities import load_sto_file
import random

class osimModel:

    stepsize = 0.0015

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
        self.coordinateSet = self.model.getCoordinateSet()

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
        
        # if self.istep == 0:
        #     print(f"Initial obs action: {action}")

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
            # make a catch statement for welded joints
            if not self.jointSet.get(i).numCoordinates() == 0:
                joint = self.jointSet.get(i)
                name = joint.getName()
                res["joint_pos"][name] = [joint.get_coordinates(j).getValue(self.state) for j in range(joint.numCoordinates())]
                res["joint_vel"][name] = [joint.get_coordinates(j).getSpeedValue(self.state) for j in range(joint.numCoordinates())]
                res["joint_acc"][name] = [joint.get_coordinates(j).getAccelerationValue(self.state) for j in range(joint.numCoordinates())]

        ## coordinates
        # this really is doing the same as above... should remove the above, this approach is better for mapping to the coordinate names in the data
        res["coord_pos"] = {}
        res["coord_vel"] = {}
        for i in range(self.coordinateSet.getSize()):
            coord = self.coordinateSet.get(i)
            name  = coord.getName()
            res["coord_pos"][coord.getAbsolutePathString() + '/value'] = coord.getValue(self.state)
            res["coord_vel"][coord.getAbsolutePathString() + '/speed'] = coord.getSpeedValue(self.state)

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

            if self.istep == 0:
                res["controls"][name] = 0.05
            else:
                res["controls"][name] = actuator.getRecordValues(self.state).get(0) / opt_force

        # if self.istep == 0:
        #     print(f"Initial obs action: {res['controls']}")

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

    def reset(self, pos0=None, start_index=0, start_time=0.0):
        self.state = self.model.initializeState()

        # if pos0 is None:
        #     self.model.setStateVariableValue(self.state, '/jointset/pin1/q1/value', np.pi/4)
        #     self.model.setStateVariableValue(self.state, '/jointset/pin2/q2/value', -np.pi/2)

        # else:
        #     self.model.setStateVariableValue(self.state, '/jointset/pin1/q1/value', pos0["q1"].item())
        #     self.model.setStateVariableValue(self.state, '/jointset/pin2/q2/value', pos0["q2"].item())

        if pos0 is not None:
            for joint_name, value in pos0.items():
                self.model.setStateVariableValue(self.state, joint_name, value.item())

        self.state.setTime(start_time)
        self.istep = start_index
        self.reset_manager()

        print(f"Initial reset state: {self.model.getStateVariableValues(self.state)}")

    def integrate(self):
        self.istep = self.istep + 1
        self.state = self.manager.integrate(self.stepsize * self.istep)

    def show_visualizer(self):
        visual = self.model.updVisualizer()
        visual.show(self.state)

    def write_states_to_file(self):
        raise NotImplementedError

class osimEnv(gym.Env):
    action_space = None
    observation_space = None
    osim_model = None
    istep = 0
    verbose = False

    visualize = False
    spec = None
    time_limit = 0.4

    writeFile = False

    prev_state_desc = None

    model_path = None

    metadata = {
        'render_modes': ['human'],
        'video.frames_per_second' : None
    }

    def get_reward(self):
        raise NotImplementedError
        #return 0.0

    def __init__(self, visualize=False, integrator_accuracy=1e-5, model_path=None, writeFile=False):
        self.visualize = visualize
        self.integrator_accuracy = integrator_accuracy
        self.writeFile = writeFile
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
        #self.osim_model.reset(pos0={"q1": 0.0, "q2": 0.0})
        self.osim_model.reset()

        return self.get_observation()
    
    def get_time(self):
        return self.osim_model.state.getTime()
    
    def get_istep(self):
        return self.osim_model.istep
    
    def _get_obs(self):
        NotImplementedError("This method should be implemented in the subclass.")
    
    def _get_info(self):
        NotImplementedError("This method should be implemented in the subclass.")

    def write_states_to_file(self):

        self.osim_model.model.setUseVisualizer(False)

        file = self.osim_model.manager.getStatesTable()
        osim.STOFileAdapter.write(file, 'C:/Users/Nicos/Documents/RL_Opensim/mock_file.sto')

        #table = opensim.TimeSeriesTable("subject01_states.sto")
        states = osim.StatesTrajectory.createFromStatesTable(self.osim_model.model, file)

        contact_r = osim.StdVectorString()
        contact_l = osim.StdVectorString()

        contact_r.append('contactHeel_r')
        contact_r.append('contactFront_r')
        contact_l.append('contactHeel_l')
        contact_l.append('contactFront_l')

        file_grf = osim.createExternalLoadsTableForGait(self.osim_model.model, states, contact_r, contact_l)
        osim.STOFileAdapter.write(file_grf, 'C:/Users/Nicos/Documents/RL_Opensim/mock_file_grf.sto')

        # plays it as a movie that you can control
        vizzi = osim.VisualizerUtilities()
        vizzi.showMotion(self.osim_model.model, file)


    def step(self, action):
        self.prev_state_desc = self.get_state_desc()
        self.osim_model.actuate(action)
        self.osim_model.integrate()
        if self.visualize == True:
            self.osim_model.show_visualizer()

        # these are the observations that have been normalized
        #obs = self.get_observation()

        # these are not normalized observations
        state_desc = self.get_state_desc()

        #if self.get_istep() < 300:
        #    self.is_done = False
        #else:
        #    self.is_done = True
        #    print(f"Episode finished at step {self.get_istep()} with time {self.get_time()}")

        if self.get_time() >= self.time_limit or state_desc["joint_pos"]["groundPelvis"][2] < 0.84:
            self.is_done = True
            print(f"Episode finished at step {self.get_istep()} with time {self.get_time()}")
            print(f"Pelvis height: {state_desc['joint_pos']['groundPelvis'][2]}")

            if self.writeFile:
                self.write_states_to_file()
        
        else:
            self.is_done = False

        # if state_desc["joint_pos"]["groundPelvis"][2] < 0.84:
        #     self.is_done = True
        # else:
        #     self.is_done = False

        # if self.is_done:
        #     print(f"action: {action}")

        truncated = False

        return self._get_obs(), self.get_reward(), self.is_done, truncated, self._get_info()

    def set_time_limit(self, time_limit):
        self.time_limit = time_limit

class Gait_2D_Env(osimEnv):

    current_dir = os.getcwd()
    model_name = "2D_gait_torque.osim"
    model_path = os.path.join(current_dir,"msk_models", model_name)

    target_q1 = 0.0 # try pelvis tilt?
    target_q2 = 0.0 # try lumbar extension?

    print(f"Loading model from: {model_path}")

    def __init__(self, visualize=False, rand_pos0=False):
        super().__init__(visualize=visualize, model_path=self.model_path)

        self.rand_pos0 = rand_pos0

        self.observation_space = gym.spaces.Dict({
            "angles": gym.spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32),
            "velocities": gym.spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32),
            "body_pos": gym.spaces.Box(low=-1, high=1, shape=(12*2,), dtype=np.float32),
            "controls": gym.spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32),
            "target_q1_diff": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            "target_q2_diff": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        })

    def get_observation(self):

        state_desc = self.osim_model.get_state_desc()

        res = {
        "angles": np.array([state_desc["joint_pos"]["groundPelvis"][0],
                                       state_desc["joint_pos"]["groundPelvis"][1],
                                       state_desc["joint_pos"]["groundPelvis"][2],
                                       state_desc["joint_pos"]["hip_l"][0],
                                       state_desc["joint_pos"]["hip_r"][0],
                                       state_desc["joint_pos"]["knee_l"][0],
                                       state_desc["joint_pos"]["knee_r"][0],
                                       state_desc["joint_pos"]["ankle_l"][0],
                                       state_desc["joint_pos"]["ankle_r"][0],
                                       state_desc["joint_pos"]["lumbar"][0],
                                       ], dtype=np.float32) / (2*np.pi),

        "velocities": np.array([state_desc["joint_vel"]["groundPelvis"][0],
                                       state_desc["joint_vel"]["groundPelvis"][1],
                                       state_desc["joint_vel"]["groundPelvis"][2],
                                       state_desc["joint_vel"]["hip_l"][0],
                                       state_desc["joint_vel"]["hip_r"][0],
                                       state_desc["joint_vel"]["knee_l"][0],
                                       state_desc["joint_vel"]["knee_r"][0],
                                       state_desc["joint_vel"]["ankle_l"][0],
                                       state_desc["joint_vel"]["ankle_r"][0],
                                       state_desc["joint_vel"]["lumbar"][0],
                                       ], dtype=np.float32) / 50.0,
                                           
        "body_pos": np.array([state_desc["body_pos"]["pelvis"][0], state_desc["body_pos"]["pelvis"][1],
                                         state_desc["body_pos"]["femur_l"][0], state_desc["body_pos"]["femur_l"][1],
                                         state_desc["body_pos"]["femur_r"][0], state_desc["body_pos"]["femur_r"][1],
                                         state_desc["body_pos"]["tibia_l"][0], state_desc["body_pos"]["tibia_l"][1],
                                         state_desc["body_pos"]["tibia_r"][0], state_desc["body_pos"]["tibia_r"][1],
                                         state_desc["body_pos"]["talus_l"][0], state_desc["body_pos"]["talus_l"][1],
                                         state_desc["body_pos"]["talus_r"][0], state_desc["body_pos"]["talus_r"][1],
                                         state_desc["body_pos"]["calcn_l"][0], state_desc["body_pos"]["calcn_l"][1],
                                         state_desc["body_pos"]["calcn_r"][0], state_desc["body_pos"]["calcn_r"][1],
                                         state_desc["body_pos"]["toes_l"][0], state_desc["body_pos"]["toes_l"][1],
                                         state_desc["body_pos"]["toes_r"][0], state_desc["body_pos"]["toes_r"][1],
                                         state_desc["body_pos"]["torso"][0], state_desc["body_pos"]["torso"][1]  
                                         ], dtype=np.float32) / 6.0,

        "controls": np.array([state_desc["controls"][actuator_name]
                                            for actuator_name in sorted(state_desc["controls"].keys())], dtype=np.float32),

        "target_q1_diff": np.array([self.target_q1 - state_desc["joint_pos"]["groundPelvis"][0]], dtype=np.float32) / 6.0,

        "target_q2_diff": np.array([self.target_q2 - state_desc["joint_pos"]["lumbar"][0]], dtype=np.float32) / 6.0
        }

        # if self.istep == 0 and self.:
        #     print(f"Initial observation action: {res['controls']}")

        # check whether the observations in the res dict contain NaN values
        for key, value in res.items():
            if isinstance(value, np.ndarray) and np.isnan(value).any():
                raise ValueError(f"Observation {key} contains NaN values: {value}")        
        
        # these are observations that have beeen normalized
        return res
    
    def _get_obs(self):
        return self.get_observation()
    
    def _get_info(self):
        return {'time': super().get_time(), 'steps': super().get_istep()}
    
    def get_observation_space_size(self):
        obs_size = 0

        obs_size += self.osim_model.jointSet.getSize() * 2  # joint positions, velocities, accelerations
        obs_size += self.osim_model.bodySet.getSize() * 2  # body positions
        obs_size += self.osim_model.actuatorSet.getSize()  # actuator controls
        obs_size += 2  # target_x and target_y    

        return obs_size
    
    def get_observation_space(self):
        return self.observation_space
    
    def get_time(self):
        return self.osim_model.state.getTime()

    def get_reward(self):
        state_desc = self.get_state_desc()

        diff_q1 = state_desc["joint_pos"]["groundPelvis"][0] - self.target_q1
        diff_q2 = state_desc["joint_pos"]["lumbar"][0] - self.target_q2

        reward_angle_diff = 1/((diff_q1 ** 2 + diff_q2 ** 2) + 0.01)

        reward_angle_height = state_desc["joint_pos"]["groundPelvis"][2] - 0.90

        controls_array = np.array(list(state_desc["controls"].values()))
        reward_controls = -np.sum(np.square(controls_array))

        return 5*reward_angle_diff + 0.01*reward_controls #+ 10*reward_angle_height
    
    def reset(self, seed=None, options=None):

        # likely doesn't work as expected as parent class has no parameters for seed
        #super().reset(seed=seed)

        self.np_random, seed = gym.utils.seeding.np_random(seed)

        # print(f"Pseudo random number: {self.np_random.uniform(low=-0.1, high=0.1, size=(2,))}")

        if self.rand_pos0:
            rand_q = self.np_random.uniform(low=-0.14,high=0.14, size=(2,)).astype(np.float32)
            pos0 = {"q1": rand_q[0] + np.pi/4, "q2": rand_q[1] - np.pi/2}
            self.osim_model.reset(pos0=pos0)
        else:
            self.osim_model.reset()

        return self._get_obs(), self._get_info()


    



class Gait_2D_Env_Track(osimEnv):

    current_dir = os.getcwd()
    model_name = "2D_gait_torque.osim"
    model_path = os.path.join(current_dir,"msk_models", model_name)

    # put the "data" to track in the reset method
    data_to_track = None

    def __init__(self, visualize=False, rand_pos0=False, writeFile=False):
        super().__init__(visualize=visualize, model_path=self.model_path, writeFile=writeFile)

        self.rand_pos0 = rand_pos0

        self.observation_space = gym.spaces.Dict({
            "angles": gym.spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32),
            "velocities": gym.spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32),
            "body_pos": gym.spaces.Box(low=-1, high=1, shape=(12*2,), dtype=np.float32),
            "controls": gym.spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32),
            "target_q1_diff": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            "target_q2_diff": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            "target_q3_diff": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            "target_q4_diff": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            "target_q5_diff": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            "target_q6_diff": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            "target_q7_diff": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            "target_q8_diff": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            "target_q9_diff": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            "target_q10_diff": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        })

    def get_observation(self):

        try:
            value_at_index = self.data_to_track[self.data_keys[0]][self.osim_model.istep+1]
        except IndexError:
            test = 1

        state_desc = self.osim_model.get_state_desc()

        res = {
        "angles": np.array([state_desc["joint_pos"]["groundPelvis"][0],
                                       state_desc["joint_pos"]["groundPelvis"][1],
                                       state_desc["joint_pos"]["groundPelvis"][2],
                                       state_desc["joint_pos"]["hip_l"][0],
                                       state_desc["joint_pos"]["hip_r"][0],
                                       state_desc["joint_pos"]["knee_l"][0],
                                       state_desc["joint_pos"]["knee_r"][0],
                                       state_desc["joint_pos"]["ankle_l"][0],
                                       state_desc["joint_pos"]["ankle_r"][0],
                                       state_desc["joint_pos"]["lumbar"][0],
                                       ], dtype=np.float32) / (2*np.pi),

        "velocities": np.array([state_desc["joint_vel"]["groundPelvis"][0],
                                       state_desc["joint_vel"]["groundPelvis"][1],
                                       state_desc["joint_vel"]["groundPelvis"][2],
                                       state_desc["joint_vel"]["hip_l"][0],
                                       state_desc["joint_vel"]["hip_r"][0],
                                       state_desc["joint_vel"]["knee_l"][0],
                                       state_desc["joint_vel"]["knee_r"][0],
                                       state_desc["joint_vel"]["ankle_l"][0],
                                       state_desc["joint_vel"]["ankle_r"][0],
                                       state_desc["joint_vel"]["lumbar"][0],
                                       ], dtype=np.float32) / 50.0,
                                           
        "body_pos": np.array([state_desc["body_pos"]["pelvis"][0], state_desc["body_pos"]["pelvis"][1],
                                         state_desc["body_pos"]["femur_l"][0], state_desc["body_pos"]["femur_l"][1],
                                         state_desc["body_pos"]["femur_r"][0], state_desc["body_pos"]["femur_r"][1],
                                         state_desc["body_pos"]["tibia_l"][0], state_desc["body_pos"]["tibia_l"][1],
                                         state_desc["body_pos"]["tibia_r"][0], state_desc["body_pos"]["tibia_r"][1],
                                         state_desc["body_pos"]["talus_l"][0], state_desc["body_pos"]["talus_l"][1],
                                         state_desc["body_pos"]["talus_r"][0], state_desc["body_pos"]["talus_r"][1],
                                         state_desc["body_pos"]["calcn_l"][0], state_desc["body_pos"]["calcn_l"][1],
                                         state_desc["body_pos"]["calcn_r"][0], state_desc["body_pos"]["calcn_r"][1],
                                         state_desc["body_pos"]["toes_l"][0], state_desc["body_pos"]["toes_l"][1],
                                         state_desc["body_pos"]["toes_r"][0], state_desc["body_pos"]["toes_r"][1],
                                         state_desc["body_pos"]["torso"][0], state_desc["body_pos"]["torso"][1]  
                                         ], dtype=np.float32) / 6.0,

        "controls": np.array([state_desc["controls"][actuator_name]
                                            for actuator_name in sorted(state_desc["controls"].keys())], dtype=np.float32),


        # CHANGE THE INDICES!
        "target_q1_diff": np.array([self.data_to_track[self.data_keys[0]][self.osim_model.istep+1] - state_desc["coord_pos"][self.data_keys[0]]], dtype=np.float32) / 6.0,

        "target_q2_diff": np.array([self.data_to_track[self.data_keys[1]][self.osim_model.istep+1] - state_desc["coord_pos"][self.data_keys[1]]], dtype=np.float32) / 6.0,

        "target_q3_diff": np.array([self.data_to_track[self.data_keys[2]][self.osim_model.istep+1] - state_desc["coord_pos"][self.data_keys[2]]], dtype=np.float32) / 6.0,

        "target_q4_diff": np.array([self.data_to_track[self.data_keys[3]][self.osim_model.istep+1] - state_desc["coord_pos"][self.data_keys[3]]], dtype=np.float32) / 6.0,

        "target_q5_diff": np.array([self.data_to_track[self.data_keys[4]][self.osim_model.istep+1] - state_desc["coord_pos"][self.data_keys[4]]], dtype=np.float32) / 6.0,

        "target_q6_diff": np.array([self.data_to_track[self.data_keys[5]][self.osim_model.istep+1] - state_desc["coord_pos"][self.data_keys[5]]], dtype=np.float32) / 6.0,

        "target_q7_diff": np.array([self.data_to_track[self.data_keys[6]][self.osim_model.istep+1] - state_desc["coord_pos"][self.data_keys[6]]], dtype=np.float32) / 6.0,

        "target_q8_diff": np.array([self.data_to_track[self.data_keys[7]][self.osim_model.istep+1] - state_desc["coord_pos"][self.data_keys[7]]], dtype=np.float32) / 6.0,

        "target_q9_diff": np.array([self.data_to_track[self.data_keys[8]][self.osim_model.istep+1] - state_desc["coord_pos"][self.data_keys[8]]], dtype=np.float32) / 6.0,

        "target_q10_diff": np.array([self.data_to_track[self.data_keys[9]][self.osim_model.istep+1] - state_desc["coord_pos"][self.data_keys[9]]], dtype=np.float32) / 6.0

        }

        # if self.istep == 0 and self.:
        #     print(f"Initial observation action: {res['controls']}")

        # check whether the observations in the res dict contain NaN values
        for key, value in res.items():
            if isinstance(value, np.ndarray) and np.isnan(value).any():
                raise ValueError(f"Observation {key} contains NaN values: {value}")        
        
        # these are observations that have beeen normalized
        return res
    
    def _get_obs(self):
        return self.get_observation()
    
    def _get_info(self):
        return {'time': super().get_time(), 'steps': super().get_istep()}
    
    def get_observation_space_size(self):
        obs_size = 0

        obs_size += self.osim_model.jointSet.getSize() * 2  # joint positions, velocities, accelerations
        obs_size += self.osim_model.bodySet.getSize() * 2  # body positions
        obs_size += self.osim_model.actuatorSet.getSize()  # actuator controls
        obs_size += 2  # target_x and target_y    

        return obs_size
    
    def get_observation_space(self):
        return self.observation_space
    
    def get_time(self):
        return self.osim_model.state.getTime()

    def get_reward(self):
        state_desc = self.get_state_desc()

        coord_pos = state_desc['coord_pos']

        #self.osim_model.istep

        diff_q1 = coord_pos[self.data_keys[0]] - self.data_to_track[self.data_keys[0]][self.osim_model.istep]
        diff_q2 = coord_pos[self.data_keys[1]] - self.data_to_track[self.data_keys[1]][self.osim_model.istep]
        diff_q3 = coord_pos[self.data_keys[2]] - self.data_to_track[self.data_keys[2]][self.osim_model.istep]
        diff_q4 = coord_pos[self.data_keys[3]] - self.data_to_track[self.data_keys[3]][self.osim_model.istep]
        diff_q5 = coord_pos[self.data_keys[4]] - self.data_to_track[self.data_keys[4]][self.osim_model.istep]
        diff_q6 = coord_pos[self.data_keys[5]] - self.data_to_track[self.data_keys[5]][self.osim_model.istep]
        diff_q7 = coord_pos[self.data_keys[6]] - self.data_to_track[self.data_keys[6]][self.osim_model.istep]
        diff_q8 = coord_pos[self.data_keys[7]] - self.data_to_track[self.data_keys[7]][self.osim_model.istep]
        diff_q9 = coord_pos[self.data_keys[8]] - self.data_to_track[self.data_keys[8]][self.osim_model.istep]
        diff_q10 = coord_pos[self.data_keys[9]] - self.data_to_track[self.data_keys[9]][self.osim_model.istep]   

        #diff_q1 = state_desc["joint_pos"]["groundPelvis"][0] - self.target_q1
        #diff_q2 = state_desc["joint_pos"]["lumbar"][0] - self.target_q2

        reward_angle_diff = 1/((diff_q1 ** 2 + diff_q2 ** 2 + diff_q3 ** 2 + diff_q4 ** 2 + 
                                diff_q5 ** 2 + diff_q6 ** 2 + diff_q7 ** 2 + diff_q8 ** 2 +
                                diff_q9 ** 2 + diff_q10 ** 2) + 0.01)

        #reward_angle_height = state_desc["joint_pos"]["groundPelvis"][2] - 0.90

        controls_array = np.array(list(state_desc["controls"].values()))
        reward_controls = -np.sum(np.square(controls_array))

        return 0.05*reward_angle_diff + 0.01*reward_controls #+ 10*reward_angle_height

    def reset(self, seed=None, options=None):

        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        random.seed(seed)

        data_name = "referenceCoordinatesStride.sto"
        self.data_time, self.data_to_track, self.data_keys = load_sto_file(os.path.join(self.current_dir, "msk_models", data_name), self.osim_model.stepsize)

        # pull a random integer from random_integer
        random_integer = self.np_random.integers(0, 5, 1) # low, high, number of samples

        data_index = None
        if random_integer == 0:
            data_index = 0
        elif random_integer == 1:
            data_index = 20
        elif random_integer == 2:
            data_index = 40
        elif random_integer == 3:
            data_index = 60
        elif random_integer == 4:
            data_index = 80

        # check if the data_index is < size of the data
        if data_index is not None and data_index > len(self.data_time[:-2]):
            raise AssertionError("Data index when initializing is out of bounds")

        super().set_time_limit(self.data_time[-2])

        pos0 = {}
        if self.data_to_track is not None:
            for i, key in enumerate(self.data_keys):
                pos0[key] = self.data_to_track[key][data_index]

        #if self.rand_pos0:
        #    rand_q = self.np_random.uniform(low=-0.14,high=0.14, size=(2,)).astype(np.float32)
        #    pos0 = {"q1": rand_q[0] + np.pi/4, "q2": rand_q[1] - np.pi/2}
        #    self.osim_model.reset(pos0=pos0)
        #else:
        #    self.osim_model.reset()
        self.osim_model.reset(pos0=pos0, start_index=data_index, start_time=self.data_time[data_index])

        return self._get_obs(), self._get_info()
