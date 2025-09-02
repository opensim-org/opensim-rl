import opensim as osim
import numpy as np
import gymnasium as gym

class osimModel:

    stepsize = 0.0025

    model = None
    state = None
    state0 = None
    joints = []
    bodies = []
    brain = None
    verbose = False
    istep = 0

    state_desc_istep = None
    prev_state_desc = None
    state_desc = None
    integrator_accuracy = None

    #target_x = []
    #target_y = []

    def __init__(self, model_path, visualize, integrator_accuracy=1e-5):
        self.integrator_accuracy = integrator_accuracy
        self.model = osim.Model(model_path)
        self.model_state = self.model.initSystem()
        self.brain = osim.PrescribedController()

        target = osim.Body('targ', 0.0001 , osim.Vec3(0), osim.Inertia(1,1,.0001,0,0,0) )
        self.target_joint = osim.FreeJoint('target-joint',
                                  self.model.getGround(), # PhysicalFrame
                                  osim.Vec3(0, 0, 0),
                                  osim.Vec3(0, 0, 0),
                                  target, # PhysicalFrame
                                  osim.Vec3(0, 0, -0.25),
                                  osim.Vec3(0, 0, 0))

        geometry = osim.Ellipsoid(0.02, 0.02, 0.02)
        geometry.setColor(osim.Green)
        target.attachGeometry(geometry)

        self.model.addJoint(self.target_joint)
        self.model.addBody(target)

        self.model.setUseVisualizer(visualize)

        self.muscleSet = self.model.getMuscles()
        self.forceSet  = self.model.getForceSet()
        self.bodySet   = self.model.getBodySet()
        self.jointSet  = self.model.getJointSet()
        self.markerSet = self.model.getMarkerSet()

        self.contactGeometrySet = self.model.getContactGeometrySet()

        #if self.verbose:
        #    self.list_elements()

        for j in range(self.muscleSet.getSize()):
            func = osim.Constant(0.5)
            self.brain.addActuator(self.muscleSet.get(j))
            self.brain.prescribeControlForActuator(self.muscleSet.get(j).getName(), func)

        self.noutput = self.muscleSet.getSize()

        self.model.addController(self.brain)

        self.model_state = self.model.initSystem()

    def actuate(self, action):
        if np.any(np.isnan(action)):
            raise ValueError("Action contains NaN values.")

        # commented out for now, we will try to build policies that obey this already
        #action = np.clip(np.array(action), 0.0, 1.0)

        brain = osim.PrescribedController.safeDownCast(self.model.getControllerSet().get(0))
        functionSet = brain.get_ControlFunctions()

        for j in range(functionSet.getSize()):
            func = osim.Constant.safeDownCast(functionSet.get(j))
            func.setValue(0.5 + 0.5*action[j].item())

    def get_activations(self):
        return [self.muscleSet.get(j).getActivation(self.state) for j in range(self.muscleSet.getSize())]

    def compute_state_desc(self):
        self.model.realizeAcceleration(self.state)
        #print(self.model.getStateVariableValues(self.state))
        #print(self.model.getStateVariableNames().get(15))
        res = {}

        ## Joints
        res["joint_pos"] = {}
        res["joint_vel"] = {}
        res["joint_acc"] = {}
        for i in range(self.jointSet.getSize()):
            joint = self.jointSet.get(i)
            name = joint.getName()
            res["joint_pos"][name] = [joint.get_coordinates(i).getValue(self.state) for i in range(joint.numCoordinates())]
            res["joint_vel"][name] = [joint.get_coordinates(i).getSpeedValue(self.state) for i in range(joint.numCoordinates())]
            res["joint_acc"][name] = [joint.get_coordinates(i).getAccelerationValue(self.state) for i in range(joint.numCoordinates())]

        ## Bodies
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

        ## Forces
        # res["forces"] = {}
        # for i in range(self.forceSet.getSize()):
        #     force = self.forceSet.get(i)
        #     name = force.getName()
        #     values = force.getRecordValues(self.state)
        #     res["forces"][name] = [values.get(i) for i in range(values.size())]

        ## Muscles
        res["muscles"] = {}
        for i in range(self.muscleSet.getSize()):
            muscle = self.muscleSet.get(i)
            name = muscle.getName()
            res["muscles"][name] = {}
            res["muscles"][name]["activation"] = muscle.getActivation(self.state)
            res["muscles"][name]["fiber_length"] = muscle.getFiberLength(self.state)
            res["muscles"][name]["fiber_velocity"] = muscle.getFiberVelocity(self.state)
            res["muscles"][name]["fiber_force"] = muscle.getFiberForce(self.state)

        ## Markers
        res["markers"] = {}
        for i in range(self.markerSet.getSize()):
            marker = self.markerSet.get(i)
            name = marker.getName()
            res["markers"][name] = {}
            res["markers"][name]["pos"] = [marker.getLocationInGround(self.state)[i] for i in range(3)]
            res["markers"][name]["vel"] = [marker.getVelocityInGround(self.state)[i] for i in range(3)]
            res["markers"][name]["acc"] = [marker.getAccelerationInGround(self.state)[i] for i in range(3)]

        return res

    def get_state_desc(self):
        #print(self.state_desc_istep, self.istep)
        if self.state_desc_istep != self.istep:
            self.prev_state_desc = self.state_desc
            self.state_desc = self.compute_state_desc()
            self.state_desc_istep = self.istep
        return self.state_desc

    def get_marker(self, name):
        return self.markerSet.get(name)

    def get_action_space_size(self):
        return self.noutput

    def set_integrator_accuracy(self, integrator_accuracy):
        self.integrator_accuracy = integrator_accuracy

    def reset_manager(self):
        self.manager = osim.Manager(self.model)
        self.manager.setIntegratorAccuracy(self.integrator_accuracy)
        self.manager.initialize(self.state)

    def reset(self, target_pos=None):

        if target_pos is None:
            target_pos = np.array([0.2, 0.2])

        # target = osim.Body('targ', 0.0001 , osim.Vec3(0), osim.Inertia(1,1,.0001,0,0,0) )
        # self.target_joint = osim.FreeJoint('target-joint',
        #                           self.model.getGround(), # PhysicalFrame
        #                           osim.Vec3(0, 0, 0),
        #                           osim.Vec3(0, 0, 0),
        #                           target, # PhysicalFrame
        #                           osim.Vec3(0, 0, -0.25),
        #                           osim.Vec3(0, 0, 0))

        # geometry = osim.Ellipsoid(0.02, 0.02, 0.02)
        # geometry.setColor(osim.Green)
        # target.attachGeometry(geometry)

        # self.model.addJoint(self.target_joint)
        # self.model.addBody(target)
        # self.model.finalizeConnections()
        # self.model.buildSystem()

        self.state = self.model.initializeState()

        target_joint = self.model.getJointSet().get('target-joint')

        target_joint.get_coordinates(3).setValue(self.state, target_pos[0].item(), False)
        target_joint.get_coordinates(4).setValue(self.state, target_pos[1].item(), False)
        #target_joint.getCoordinate(2).setValue(self.state, 0, False)

        target_joint.get_coordinates(3).setLocked(self.state, True)
        target_joint.get_coordinates(4).setLocked(self.state, True)


        self.model.equilibrateMuscles(self.state)
        self.state.setTime(0.0)
        self.istep = 0
        print("resetting")

        #print("simulation time:", self.state.getTime(), "istep:", self.istep)


        self.reset_manager()

    def get_state(self):
        return osim.State(self.state)

    def set_state(self, state):
        self.state = state
        #####self.istep = int(self.state.getTime() / self.stepsize) # ???
        self.reset_manager()

    def integrate(self):
        self.istep = self.istep + 1
        #print("istep:", self.istep)
        self.state = self.manager.integrate(self.stepsize * self.istep)

    def get_states_table(self):
        return self.manager.getStatesTable()

    def write_and_viz(self):
        file = self.get_states_table()
        osim.STOFileAdapter.write(file, 'C:/Users/Nicos/Documents/OpenSim/4.5-2024-01-10-3b63585/Models/Arm26/output_states.sto')
        osim.VisualizerUtilities.showMotion(self.model,file)

    def testing_viz(self):
        visual = self.model.updVisualizer()
        visual.show(self.state)

class osimEnv(gym.Env):
    action_space = None
    observation_space = None
    osim_model = None
    istep = 0
    verbose = False

    visualize = False
    spec = None
    time_limit = 1.5

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

        #self.spec = Spec()
        #self.spec.timestep_limit = self.time_limit

        # define this later?
        #self.observation_space = ( [0] * self.get_observation_space_size(), [0] * self.get_observation_space_size() )

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

    def reset(self, project=True, obs_as_dict=True):
        self.osim_model.reset()

        if not project:
            return self.get_state_desc()
        if obs_as_dict:
            return self.get_observation_dict()
        return self.get_observation()

    def get_time(self):
        return self.osim_model.state.getTime()

    def get_istep(self):
        return self.osim_model.istep

    def _get_obs(self):
        NotImplementedError("This method should be implemented in the subclass.")

    def _get_info(self):
        NotImplementedError("This method should be implemented in the subclass.")

    def step(self, action, project=True, obs_as_dict=False):

        #print("simulation time:", self.get_time(), "istep:", self.get_istep())

        self.prev_state_desc = self.get_state_desc()
        self.osim_model.actuate(action)
        self.osim_model.integrate()
        if self.visualize == True:
            self.osim_model.testing_viz()
        #print("time:", self.osim_model.state.getTime(), "istep:", self.osim_model.istep)

        if project:
            if obs_as_dict:
                obs = self.get_observation_dict()
            else:
                obs = self.get_observation()
        else:
            obs = self.get_state_desc()

        if self.get_time() < self.time_limit:
            self.is_done = False
        else:
            self.is_done = True
            #self.osim_model.write_and_viz()

        if obs['angles'][1] < -0.2/np.pi:
            self.is_done = True

        # check whether the observations contain NaN values
        for key, value in obs.items():
            if isinstance(value, np.ndarray) and np.isnan(value).any():
                self.is_done = True
                break

        truncated = False

        #return [ obs, self.get_reward(), self.is_done() or (self.osim_model.istep >= self.spec.timestep_limit), {} ]
        return self._get_obs(), self.get_reward(), self.is_done, truncated, self._get_info()

class armEnv(osimEnv):

    model_path = 'C:/Users/Nicos/Documents/OpenSim/4.5-2024-01-10-3b63585/Models/Arm26/arm26_4_PC.osim'

    target_x = 0.83
    target_y = 0.63

    def __init__(self, visualize=False):
        super().__init__(model_path=self.model_path, visualize=visualize)

        self.observation_space = gym.spaces.Dict({
            "angles": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(3-1,), dtype=np.float32),
            "velocities": gym.spaces.Box(low=-20, high=20, shape=(3-1,), dtype=np.float32),
            "body_pos": gym.spaces.Box(low=-3, high=3, shape=(2*2,), dtype=np.float32),
            "activations": gym.spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32),
            "fiber_length": gym.spaces.Box(low=0, high=2, shape=(6,), dtype=np.float32),
            "fiber_velocity": gym.spaces.Box(low=-20, high=20, shape=(6,), dtype=np.float32),
            "marker_pos": gym.spaces.Box(low=-3, high=3, shape=(2,), dtype=np.float32),
            "target_x": gym.spaces.Box(low=-3, high=3, shape=(1,), dtype=np.float32),
            "target_y": gym.spaces.Box(low=-3, high=3, shape=(1,), dtype=np.float32)
        })

    def get_observation(self):

        # res = {}

        # ## Joints
        # res["angles"] = {}
        # res["velocities"] = {}

        state_desc = self.osim_model.get_state_desc()

        # res["angles"]["r_shoulder"] = state_desc["joint_pos"]["r_shoulder"]
        # res["angles"]["r_elbow"] = state_desc["joint_pos"]["r_elbow"]

        res = {
        "angles": np.squeeze(np.array([state_desc["joint_pos"]["r_shoulder"],
                                       state_desc["joint_pos"]["r_elbow"]], dtype=np.float32), axis=1) / np.pi,
        "velocities": np.squeeze(np.array([state_desc["joint_vel"]["r_shoulder"],
                                           state_desc["joint_vel"]["r_elbow"]], dtype=np.float32), axis=1) / 20.0,
        "body_pos": np.array([state_desc["body_pos"]["r_humerus"][0], state_desc["body_pos"]["r_humerus"][1],
                                         state_desc["body_pos"]["r_ulna_radius_hand"][0], state_desc["body_pos"]["r_ulna_radius_hand"][1]], dtype=np.float32) / 3.0,
        "activations": np.array([state_desc["muscles"][muscle_name]["activation"]
                                            for muscle_name in sorted(state_desc["muscles"].keys())], dtype=np.float32),
        "fiber_length": np.array([state_desc["muscles"][muscle_name]["fiber_length"]
                                             for muscle_name in sorted(state_desc["muscles"].keys())], dtype=np.float32) / 2.0,
        "fiber_velocity": np.array([state_desc["muscles"][muscle_name]["fiber_velocity"]
                                               for muscle_name in sorted(state_desc["muscles"].keys())], dtype=np.float32) / 20.0,
        "marker_pos": np.array(state_desc["markers"]["r_radius_styloid"]["pos"][0:2], dtype=np.float32) / 3.0,
        "target_x": np.array([self.target_x], dtype=np.float32) / 3.0,
        "target_y": np.array([self.target_y], dtype=np.float32) / 3.0
        }

        # check whether the observations in the res dict contain NaN values
        # for key, value in res.items():
        #     if isinstance(value, np.ndarray) and np.isnan(value).any():
        #         raise ValueError(f"Observation {key} contains NaN values: {value}")

        # I UNCOMMENTED THE ABOVE CODE, AS THIS WOULD TERMINATE ALGORITHM HERE
        # NOW THIS IS CHECKED IN THE STEP FUNCTION AND THE ENVIRONMENT IS TERMINATED IF ANY OBSERVATION CONTAINS NaN VALUES

        # res = []

        # for joint_name in ["r_shoulder", "r_elbow"]:
        #     res += state_desc["joint_pos"][joint_name]
        #     res += state_desc["joint_vel"][joint_name]

        # for body_name in ["r_humerus", "r_ulna_radius_hand"]:
        #     res += state_desc["body_pos"][body_name][0:2]

        # for muscle_name in sorted(state_desc["muscles"].keys()):
        #     res += [state_desc["muscles"][muscle_name]["activation"]]
        #     res += [state_desc["muscles"][muscle_name]["fiber_length"]]
        #     res += [state_desc["muscles"][muscle_name]["fiber_velocity"]]

        # res += state_desc["markers"]["r_radius_styloid"]["pos"][0:2]

        #obs["target_x"] = self.target_x
        #obs["target_y"] = self.target_y

        # res += [self.target_x]
        # res += [self.target_y]

        return res

    def _get_obs(self):
        return self.get_observation()

    def _get_info(self):
        return {
            "distance": self.reward_distance()
        }

    def get_observation_space_size(self):
        obs_size = 0

        obs_size += (self.osim_model.jointSet.getSize() - 1) * 2  # joint positions, velocities, accelerations
        obs_size += (self.osim_model.bodySet.getSize() - 1) * 2  # body positions, velocities, accelerations (3 for position, 3 for rotation)
        obs_size += self.osim_model.muscleSet.getSize() * 3  # muscle activations, fiber lengths, fiber velocities, fiber forces
        obs_size += (self.osim_model.markerSet.getSize() -2) * 2  # marker positions
        obs_size += 2  # target_x and target_y

        return obs_size

    def get_observation_space(self):

        return self.observation_space

    def reward_distance(self):
        state_desc = self.get_state_desc()
        #penalty = (state_desc["markers"]["r_radius_styloid"]["pos"][0] - self.target_x)**2 + (state_desc["markers"]["r_radius_styloid"]["pos"][1] - self.target_y)**2
        penalty = np.linalg.norm(np.array([state_desc["markers"]["r_radius_styloid"]["pos"][0], state_desc["markers"]["r_radius_styloid"]["pos"][1]])
                                - np.array([self.target_x, self.target_y]))

        #print("penalty:", penalty, "target_x:", self.target_x, "target_y:", self.target_y)
        #print("r_radius_styloid pos:", state_desc["markers"]["r_radius_styloid"]["pos"][0], state_desc["markers"]["r_radius_styloid"]["pos"][1])

        #print("penalty:", penalty, "target_x:", self.target_x, "target_y:", self.target_y)

        #randomness= self.np_random.uniform(low=-0.2, high=0.2, size=(2,)).astype(np.float32)

        #test_time = np.linspace(0, 1.5, np.int32(1.5/self.osim_model.stepsize) + 1)
        #test_angle1 = np.linspace(0, np.pi/3, np.int32(1.5/self.osim_model.stepsize) + 1)
        #test_angle2 = np.linspace(0, np.pi/2, np.int32(1.5/self.osim_model.stepsize) + 1)

        #print("test_time:", test_time[self.get_istep()], "istep:", self.get_istep())

        #pen1 = (state_desc["joint_pos"]["r_shoulder"][0] - test_angle1[self.get_istep()])**2
        #pen2 = (state_desc["joint_pos"]["r_elbow"][0] - test_angle2[self.get_istep()])**2
        #penalty = pen1 + pen2

        if np.isnan(penalty):
            penalty = 1
        #return 50*np.exp(-penalty*10)

        addtional_reward = 0.0
        if penalty < 0.1:
            addtional_reward = 50.0
            #print("reached target, adding reward:", addtional_reward)

        return np.exp(-penalty*10) + addtional_reward

    def reward_r_elbow(self):
        state_desc = self.get_state_desc()
        r_elbow_angle = state_desc["joint_pos"]["r_elbow"][0]
        #print("r_elbow_angle:", r_elbow_angle, "istep:", self.osim_model.istep)
        if r_elbow_angle > np.pi - (15/(180/np.pi)):
            return -5.0
        else:
            return 0.0

    def reward_effort(self):
        state_desc = self.get_state_desc()
        effort = 0.0
        for muscle_name in sorted(state_desc["muscles"].keys()):
            effort += state_desc["muscles"][muscle_name]["activation"]**2

        return -0.001*effort # 1.0 - effort / self.osim_model.muscleSet.getSize()

    # def reward_termination(self):
    #         if self.osim_model.get_time() <= self.time_limit and self.is_done:
    #             pen = -100
    #         else:
    #             pen = 0

    #         return pen

    def get_reward(self):
        return self.reward_distance() + self.reward_effort() + self.reward_r_elbow() #+ self.reward_termination()

    def reset(self, project=True, seed=None, options=None):

        self.np_random, seed = gym.utils.seeding.np_random(seed)

        # Use the RNG to generate a random initial position for the target
        target_pos = self.np_random.uniform(low=-0.1, high=0.1, size=(2,)).astype(np.float32)

        #print("target_pos:", target_pos)

        self.target_x = 0.41 #target_pos[0]
        self.target_y = 0.63 + target_pos[1]

        self.osim_model.reset(target_pos=np.array([self.target_x, self.target_y], dtype=np.float32))
        # super.reset() # will call the reset method of the parent class

        if not project:
            return self.get_state_desc()
        if options is not None and options.get("obs_as_dict", True):
            return self._get_obs(), self._get_info()
        return self._get_obs(), self._get_info()


#model_path = 'C:/Users/Nicos/Documents/OpenSim/4.5-2024-01-10-3b63585/Models/Arm26/arm26_4_PC.osim'

#class_test = osimModel(model_path=model_path, visualize=False)

#class_test.actuate(np.ones(class_test.noutput) * 0.2)
#class_test.reset()
#class_test.get_state_desc()



#osim_env = osimEnv()
#osim_env.load_model(model_path=model_path)

#print("testing2")
