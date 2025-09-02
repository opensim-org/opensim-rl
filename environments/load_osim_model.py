import opensim as osim

model_path = 'C:/Users/Nicos/Documents/OpenSim/4.5-2024-01-10-3b63585/Models/Arm26/arm26_4.osim'

model = osim.Model(model_path)

for actuator in model.getActuators():
    print(f"Name: {actuator.getName()}")