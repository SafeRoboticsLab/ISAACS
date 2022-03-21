import pybullet as p 
import pybullet_data 
import time
import math

client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10, physicsClientId=client)

planeId = p.loadURDF("plane.urdf")
robotId = p.loadURDF("/media/buzi/New Volume/Ubuntu/robotics/bullet3-master/examples/pybullet/gym/pybullet_data/quadruped/spirit40.urdf",[0,0,.5], useFixedBase=False)

leg0HipTorque = p.addUserDebugParameter("leg0 Hip Torque", -5, 5, 0)
leg0UpperTorque = p.addUserDebugParameter("leg0 Upper Torque", -5, 5, 0)
leg0LowerTorque = p.addUserDebugParameter("leg0 Lower Torque", -5, 5, 0)

leg1HipTorque = p.addUserDebugParameter("leg1 Hip Torque", -1.57, 1.57, 0)
leg1UpperTorque = p.addUserDebugParameter("leg1 Upper Torque", -5, 5, 0)
leg1LowerTorque = p.addUserDebugParameter("leg1 Lower Torque", -5, 5, 0)

leg2HipTorque = p.addUserDebugParameter("leg2 Hip Torque", -1.57, 1.57, 0)
leg2UpperTorque = p.addUserDebugParameter("leg2 Upper Torque", -5, 5, 0)
leg2LowerTorque = p.addUserDebugParameter("leg2 Lower Torque", -5, 5, 0)

leg3HipTorque = p.addUserDebugParameter("leg3 Hip Torque", -1.57, 1.57, 0)
leg3UpperTorque = p.addUserDebugParameter("leg3 Upper Torque", -5, 5, 0)
leg3LowerTorque = p.addUserDebugParameter("leg3 Lower Torque", -5, 5, 0)

toes = [3, 7, 11, 15]
revolute_joints=[1, 2, 5, 6, 9, 10, 13, 14, 0, 4, 8, 12]

while True:
    pos, ori = p.getBasePositionAndOrientation(robotId, client)
    # print(ori)
    ang = p.getEulerFromQuaternion(ori)
    # print("{}\t{}\t{}".format(round(math.degrees(ang[0])), round(math.degrees(ang[1])), round(math.degrees(ang[2])))) 
    vel = p.getBaseVelocity(robotId, client)
    # print(round(vel[0][0]))

    userLeg0HipTorque = p.readUserDebugParameter(leg0HipTorque)
    userLeg0UpperTorque = p.readUserDebugParameter(leg0UpperTorque)
    userLeg0LowerTorque = p.readUserDebugParameter(leg0LowerTorque)

    userLeg1HipTorque = p.readUserDebugParameter(leg1HipTorque)
    userLeg1UpperTorque = p.readUserDebugParameter(leg1UpperTorque)
    userLeg1LowerTorque = p.readUserDebugParameter(leg1LowerTorque)

    userLeg2HipTorque = p.readUserDebugParameter(leg2HipTorque)
    userLeg2UpperTorque = p.readUserDebugParameter(leg2UpperTorque)
    userLeg2LowerTorque = p.readUserDebugParameter(leg2LowerTorque)

    userLeg3HipTorque = p.readUserDebugParameter(leg3HipTorque)
    userLeg3UpperTorque = p.readUserDebugParameter(leg3UpperTorque)
    userLeg3LowerTorque = p.readUserDebugParameter(leg3LowerTorque)
    
    for joint in revolute_joints:
        p.setJointMotorControl2(robotId, joint, p.VELOCITY_CONTROL, force = 0)
    
    p.setJointMotorControl2(robotId, 0, p.TORQUE_CONTROL, force = userLeg0HipTorque)
    p.setJointMotorControl2(robotId, 1, p.TORQUE_CONTROL, force = userLeg0UpperTorque)
    p.setJointMotorControl2(robotId, 2, p.TORQUE_CONTROL, force = userLeg0LowerTorque)

    p.setJointMotorControl2(robotId, 4, p.TORQUE_CONTROL, force = userLeg1HipTorque)
    p.setJointMotorControl2(robotId, 5, p.TORQUE_CONTROL, force = userLeg1UpperTorque)
    p.setJointMotorControl2(robotId, 6, p.TORQUE_CONTROL, force = userLeg1LowerTorque)

    p.setJointMotorControl2(robotId, 8, p.TORQUE_CONTROL, force = userLeg2HipTorque)
    p.setJointMotorControl2(robotId, 9, p.TORQUE_CONTROL, force = userLeg2UpperTorque)
    p.setJointMotorControl2(robotId, 10, p.TORQUE_CONTROL, force = userLeg2LowerTorque)

    p.setJointMotorControl2(robotId, 12, p.TORQUE_CONTROL, force = userLeg3HipTorque)
    p.setJointMotorControl2(robotId, 13, p.TORQUE_CONTROL, force = userLeg3UpperTorque)
    p.setJointMotorControl2(robotId, 14, p.TORQUE_CONTROL, force = userLeg3LowerTorque)
    
    p.stepSimulation()

    time.sleep(1./240.)
p.disconnect()