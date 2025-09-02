import opensim as osim
import os
from env_2d_torque_gait import osimEnv

current_dir = os.getcwd()
model_name = "2d_gait_torque.osim"
model_path = os.path.join(current_dir,"msk_models", model_name)

print(f"Loading model from: {model_path}")

#model = osimModel(model_path=model_path, visualize=True, integrator_accuracy=1e-5)

# model = osim.Model(model_path)
# state = model.initSystem()

# forceSet  = model.getActuators()

# brain = osim.PrescribedController()

# for j in range(forceSet.getSize()):
#     func = osim.Constant(0.5)
#     brain.addActuator(forceSet.get(j))
#     brain.prescribeControlForActuator(forceSet.get(j).getName(), func)

# model.addController(brain)

# state = model.initSystem()

# brain0 = osim.PrescribedController.safeDownCast(model.getControllerSet().get(0))
# functionSet0 = brain0.get_ControlFunctions()

# for j in range(functionSet0.getSize()):
#     func = osim.Constant.safeDownCast(functionSet0.get(j))
    
#     if j <= 0:  
#         func.setValue(-10.0)
#     else:
#         func.setValue(0.0)

# model.realizeAcceleration(state)

# for i in range(model.getJointSet().getSize()):
#     joint = model.getJointSet().get(i)
#     name = joint.getName()
#     [print(joint.get_coordinates(i).getValue(state)) for i in range(joint.numCoordinates())]
#     [print(joint.get_coordinates(i).getSpeedValue(state)) for i in range(joint.numCoordinates())]
#     [print(joint.get_coordinates(i).getAccelerationValue(state)) for i in range(joint.numCoordinates())]

#modeling = doublePendulumEnv(visualize=True)
#modeling.reset(rand_pos0=True)

modeling = osimEnv(model_path=model_path, visualize=True)
modeling.reset()#(pos0={"q1": 0.0, "q2": 1.0})

for i in range(500):
    action = [1.0, 0.8, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    modeling.step(action=action)  # Integrate the model for one step

#     state_desc = modeling.get_state_desc()
#     print(f"Step {i+1}:")
#     print("Joint Positions:", state_desc["joint_pos"])
#     print("Joint Velocities:", state_desc["joint_vel"])
#     print("Joint Accelerations:", state_desc["joint_acc"])
#     print("time:", modeling.get_time())
#     print("Controls:", state_desc["controls"])