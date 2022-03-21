import pybullet as p 
import pybullet_data 
import time
import math

client = p.connect(p.GUI)
p.configureDebugVisualizer(
    p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(
    p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
p.configureDebugVisualizer(
    p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
p.configureDebugVisualizer(
    p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
# p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
p.resetDebugVisualizerCamera(
    cameraDistance=1,
    cameraYaw=20,
    cameraPitch=-20,
    cameraTargetPosition=[1, -0.5, 0.8])
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10, physicsClientId=client)

planeId = p.loadURDF("plane.urdf")
robotId = p.loadURDF("spirit40.urdf",[0,0,.5], useFixedBase=False)

leg0HipAngle = p.addUserDebugParameter("leg0 Hip Angle", -1.57, 1.57, 0)
leg0UpperAngle = p.addUserDebugParameter("leg0 Upper Angle", -1.57, 1.57, 0.75)
# leg0UpperAngle = p.addUserDebugParameter("leg0 Upper Angle", 0, 3.14, 0.5)
leg0LowerAngle = p.addUserDebugParameter("leg0 Lower Angle", 0, 3.14, 1.5)

leg1HipAngle = p.addUserDebugParameter("leg1 Hip Angle", -1.57, 1.57, 0)
leg1UpperAngle = p.addUserDebugParameter("leg1 Upper Angle", -1.57, 1.57, 0.75)
leg1LowerAngle = p.addUserDebugParameter("leg1 Lower Angle", 0, 3.14, 1.5)

leg2HipAngle = p.addUserDebugParameter("leg2 Hip Angle", -1.57, 1.57, 0)
leg2UpperAngle = p.addUserDebugParameter("leg2 Upper Angle", -1.57, 1.57, 0.75)
leg2LowerAngle = p.addUserDebugParameter("leg2 Lower Angle", 0, 3.14, 1.5)

leg3HipAngle = p.addUserDebugParameter("leg3 Hip Angle", -1.57, 1.57, 0)
leg3UpperAngle = p.addUserDebugParameter("leg3 Upper Angle", -1.57, 1.57, 0.75)
leg3LowerAngle = p.addUserDebugParameter("leg3 Lower Angle", 0, 3.14, 1.5)

toes = [3, 7, 11, 15]
revolute_joints=[1, 2, 5, 6, 9, 10, 13, 14, 0, 4, 8, 12]

while True:
    pos, ori = p.getBasePositionAndOrientation(robotId, client)
    # print(ori)
    ang = p.getEulerFromQuaternion(ori)
    # print("{}\t{}\t{}".format(round(math.degrees(ang[0])), round(math.degrees(ang[1])), round(math.degrees(ang[2])))) 
    vel = p.getBaseVelocity(robotId, client)
    # print(round(vel[0][0]))

    userLeg0HipAngle = p.readUserDebugParameter(leg0HipAngle)
    userLeg0UpperAngle = p.readUserDebugParameter(leg0UpperAngle)
    userLeg0LowerAngle = p.readUserDebugParameter(leg0LowerAngle)

    userLeg1HipAngle = p.readUserDebugParameter(leg1HipAngle)
    userLeg1UpperAngle = p.readUserDebugParameter(leg1UpperAngle)
    userLeg1LowerAngle = p.readUserDebugParameter(leg1LowerAngle)

    userLeg2HipAngle = p.readUserDebugParameter(leg2HipAngle)
    userLeg2UpperAngle = p.readUserDebugParameter(leg2UpperAngle)
    userLeg2LowerAngle = p.readUserDebugParameter(leg2LowerAngle)

    userLeg3HipAngle = p.readUserDebugParameter(leg3HipAngle)
    userLeg3UpperAngle = p.readUserDebugParameter(leg3UpperAngle)
    userLeg3LowerAngle = p.readUserDebugParameter(leg3LowerAngle)
    
    p.setJointMotorControl2(robotId, 0, p.POSITION_CONTROL, targetPosition = userLeg0HipAngle)
    p.setJointMotorControl2(robotId, 1, p.POSITION_CONTROL, targetPosition = userLeg0UpperAngle)
    p.setJointMotorControl2(robotId, 2, p.POSITION_CONTROL, targetPosition = userLeg0LowerAngle)

    p.setJointMotorControl2(robotId, 4, p.POSITION_CONTROL, targetPosition = userLeg1HipAngle)
    p.setJointMotorControl2(robotId, 5, p.POSITION_CONTROL, targetPosition = userLeg1UpperAngle)
    p.setJointMotorControl2(robotId, 6, p.POSITION_CONTROL, targetPosition = userLeg1LowerAngle)

    p.setJointMotorControl2(robotId, 8, p.POSITION_CONTROL, targetPosition = userLeg2HipAngle)
    p.setJointMotorControl2(robotId, 9, p.POSITION_CONTROL, targetPosition = userLeg2UpperAngle)
    p.setJointMotorControl2(robotId, 10, p.POSITION_CONTROL, targetPosition = userLeg2LowerAngle)

    p.setJointMotorControl2(robotId, 12, p.POSITION_CONTROL, targetPosition = userLeg3HipAngle)
    p.setJointMotorControl2(robotId, 13, p.POSITION_CONTROL, targetPosition = userLeg3UpperAngle)
    p.setJointMotorControl2(robotId, 14, p.POSITION_CONTROL, targetPosition = userLeg3LowerAngle)
    
    p.stepSimulation()

    joint_state = p.getJointStates(robotId, jointIndices = revolute_joints)
    position = [state[3] for state in joint_state]
    print(sum(position))

    time.sleep(1./240.)
p.disconnect()