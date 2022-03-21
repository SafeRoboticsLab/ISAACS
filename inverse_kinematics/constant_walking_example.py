import pybullet as p
import numpy as np
import pybullet_data
from kinematic_model import robotKinematics
from gaitPlanner import trotGait
import os
import time

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-9.8)

p.loadURDF("plane.urdf")
f_name = os.path.join(os.path.dirname(__file__), 'spirit40.urdf')
robotId = p.loadURDF(f_name, [0,0,0.5])

robotKinematics = robotKinematics()
trot = trotGait() 

#robot properties
maxForce = 40
maxVel = 30

"""initial foot position"""
#foot separation (Ydist = 0.16 -> tetta=0) and distance to floor
Xdist = 0.39
Ydist = 0.28
height = 0.3

#body frame to foot frame vector
bodytoFeet0 = np.matrix([[ Xdist/2 , -Ydist/2 , -height],
                         [ Xdist/2 ,  Ydist/2 , -height],
                         [-Xdist/2 , -Ydist/2 , -height],
                         [-Xdist/2 ,  Ydist/2 , -height]])

offset = np.array([0.5, 0., 0., 0.5]) #defines the offset between each foot step in this order (FR,FL,BR,BL)

p.setRealTimeSimulation(0)
p.setTimeStep(1./240.)

pos = np.zeros([3])
orn = np.zeros([3])
Lrot = 0
angle = 180
L = 1.2
T = 1.0

def make_joint_list():
    damaged_legs = []
    joint_names = [
        b'hip0', b'upper0', b'lower0',
        b'hip1', b'upper1', b'lower1',
        b'hip2', b'upper2', b'lower2',
        b'hip3', b'upper3', b'lower3'
    ]
    joint_list = []
    for n in joint_names:
        joint_found = False
        for joint in range(p.getNumJoints(robotId, physicsClientId = physicsClient)):
            name = p.getJointInfo(robotId, joint, physicsClientId = physicsClient)[12]
            if name == n and name not in damaged_legs:
                joint_list += [joint]
                joint_found = True
            elif name == n and name in damaged_legs:
                p.changeVisualShape(
                    1, joint, rgbaColor=[0.5, 0.5, 0.5, 0.5], physicsClientId = physicsClient)
        if joint_found is False:
            # if the joint is not here (aka broken leg case) put 1000
            # joint_list += [1000]
            continue
    return joint_list

joint_index = make_joint_list()
for i in range(len(joint_index)):
    info = p.getJointInfo(robotId, joint_index[i], physicsClientId = physicsClient)
    lower_limit = info[8]
    upper_limit = info[9]
    max_force = info[10]
    max_velocity = info[11]
    
    print(max_force, max_velocity)

iteration = 0
bodytoFeet = trot.loop(L , angle , Lrot , T , offset , bodytoFeet0)
FR_angles, FL_angles, BR_angles, BL_angles , transformedBodytoFeet = robotKinematics.solve(orn, pos, bodytoFeet)

while True:
    if iteration % (0.05 / (1./240.)) == 0:
        bodytoFeet = trot.loop(L , angle , Lrot , T , offset , bodytoFeet0)
        FR_angles, FL_angles, BR_angles, BL_angles , transformedBodytoFeet = robotKinematics.solve(orn, pos, bodytoFeet)

    p.setJointMotorControl2(robotId, 0, p.POSITION_CONTROL, 
                                targetPosition = FL_angles[0], positionGain=1./12., velocityGain=0.4, force = maxForce , maxVelocity = maxVel)
    p.setJointMotorControl2(robotId, 1, p.POSITION_CONTROL, 
                                targetPosition = FL_angles[1], positionGain=1./12., velocityGain=0.4, force = maxForce , maxVelocity = maxVel)
    p.setJointMotorControl2(robotId, 2, p.POSITION_CONTROL, 
                                targetPosition = FL_angles[2] + 3.14, positionGain=1./12., velocityGain=0.4, force = maxForce , maxVelocity = maxVel)

    p.setJointMotorControl2(robotId, 4, p.POSITION_CONTROL, 
                                targetPosition = BL_angles[0], positionGain=1./12., velocityGain=0.4, force = maxForce , maxVelocity = maxVel)
    p.setJointMotorControl2(robotId, 5, p.POSITION_CONTROL, 
                                targetPosition = BL_angles[1], positionGain=1./12., velocityGain=0.4, force = maxForce , maxVelocity = maxVel)
    p.setJointMotorControl2(robotId, 6, p.POSITION_CONTROL, 
                                targetPosition = BL_angles[2] + 3.14, positionGain=1./12., velocityGain=0.4, force = maxForce , maxVelocity = maxVel)

    p.setJointMotorControl2(robotId, 8, p.POSITION_CONTROL, 
                                targetPosition = FR_angles[0], positionGain=1./12., velocityGain=0.4, force = maxForce , maxVelocity = maxVel)
    p.setJointMotorControl2(robotId, 9, p.POSITION_CONTROL, 
                                targetPosition = FR_angles[1], positionGain=1./12., velocityGain=0.4, force = maxForce , maxVelocity = maxVel)
    p.setJointMotorControl2(robotId, 10, p.POSITION_CONTROL, 
                                targetPosition = FR_angles[2] + 3.14, positionGain=1./12., velocityGain=0.4, force = maxForce , maxVelocity = maxVel)

    p.setJointMotorControl2(robotId, 12, p.POSITION_CONTROL, 
                                targetPosition = BR_angles[0], positionGain=1./12., velocityGain=0.4, force = maxForce , maxVelocity = maxVel)
    p.setJointMotorControl2(robotId, 13, p.POSITION_CONTROL, 
                                targetPosition = BR_angles[1], positionGain=1./12., velocityGain=0.4, force = maxForce , maxVelocity = maxVel)
    p.setJointMotorControl2(robotId, 14, p.POSITION_CONTROL, 
                                targetPosition = BR_angles[2] + 3.14, positionGain=1./12., velocityGain=0.4, force = maxForce , maxVelocity = maxVel)
    
    p.stepSimulation(physicsClientId = physicsClient)
    iteration+=1
    
    time.sleep(1./240.)

p.disconnect()