import opensim as osim

model_path = 'C:/Users/Nicos/Documents/OpenSim/4.5-2024-01-10-3b63585/Models/Arm26/arm26_4.osim'

model = osim.Model(model_path)

brain = osim.PrescribedController()

brain.setName("brain")
for actuator in model.getActuators():
    brain.addActuator(actuator)
    brain.prescribeControlForActuator(actuator.getName(), osim.Constant(0.05))
    
model.addController(brain)
model.finalizeConnections()

#model.addController(prescibeController)
#model.finalizeConnections()

model.printToXML('C:/Users/Nicos/Documents/OpenSim/4.5-2024-01-10-3b63585/Models/Arm26/arm26_4_PC.osim')