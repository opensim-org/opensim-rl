import opensim as osim

model_path = 'C:/Users/Nicos/Documents/OpenSim/4.5-2024-01-10-3b63585/Models/Arm26/arm26_4_PC.osim'

model = osim.Model(model_path)


state = model.initSystem()
#model.realizeVelocity(state)

#osim.VisualizerUtilities.showModel(model)

#visualizer = model.setUseVisualizer(True)


brain = osim.PrescribedController.safeDownCast(model.getControllerSet().get(0))
functionSet = brain.get_ControlFunctions()

for j in range(functionSet.getSize()):
    func = osim.Constant.safeDownCast(functionSet.get(j))
    
    if j < 5:  
        func.setValue(0.01)
    else:
        func.setValue(0.6)

model.realizeAcceleration(state)

for i in range(model.getJointSet().getSize()):
    joint = model.getJointSet().get(i)
    name = joint.getName()
    [print(joint.get_coordinates(i).getValue(state)) for i in range(joint.numCoordinates())]
    [print(joint.get_coordinates(i).getSpeedValue(state)) for i in range(joint.numCoordinates())]
    [print(joint.get_coordinates(i).getAccelerationValue(state)) for i in range(joint.numCoordinates())]

#print(model.getActuators().get(4).getName())

#storage_file = osim.Storage()

manager = osim.Manager(model)
state.setTime(0.0)
#manager.setStateStorage(storage_file)
manager.initialize(state)

for j in range(250):
    state = manager.integrate(0.005 * (j+1))
    for i in range(model.getJointSet().getSize()):
        joint = model.getJointSet().get(i)
        name = joint.getName()
        #[print(joint.get_coordinates(i).getValue(state)) for i in range(joint.numCoordinates())]
        #[print(joint.get_coordinates(i).getSpeedValue(state)) for i in range(joint.numCoordinates())]
        #[print(state.getTime())]

    if 0.005*(j+1) >= 1.0:
        func0 = osim.Constant.safeDownCast(functionSet.get(0))
        func0.setValue(0.95)
    model.realizeAcceleration(state)
    print(model.getControls(state))

#cont4 = osim.PrescribedController.safeDownCast(model.getControllerSet().get(4))
#cont5 = osim.PrescribedController.safeDownCast(model.getControllerSet().get(5))

#u_cont4 = cont4.get_ControlFunctions()

#print(u_cont4)
#print(C2)

#manager.integrate(5.0)

#f1 = manager.getStateStorage()

file = manager.getStatesTable()

osim.STOFileAdapter.write(file, 'C:/Users/Nicos/Documents/OpenSim/4.5-2024-01-10-3b63585/Models/Arm26/output_states.sto')

#f1_table = osim.STOFileAdapter().read(f1)
#fd.write(file, 'C:/Users/Nicos/Documents/OpenSim/4.5-2024-01-10-3b63585/Models/Arm26/arm26_4_PC_states.sto')

osim.VisualizerUtilities.showMotion(model,file)
