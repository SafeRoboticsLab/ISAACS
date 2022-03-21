import pybullet as p 
import pybullet_data 
import time
import math
import numpy as np

client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8, physicsClientId=client)
p.setTimeStep(1.0/240.)
p.setPhysicsEngineParameter(fixedTimeStep = 1.0/240.)

planeId = p.loadURDF("plane.urdf")
robotId = p.loadURDF("spirit40.urdf",[0,0,.5], p.getQuaternionFromEuler([0.0, 0.0, 0.0]), useFixedBase=False)

debug_param = dict()
for joint in range(p.getNumJoints(robotId)):
    debug_param[joint] = p.addUserDebugParameter("joint {}".format(joint), -3.14, 3.14, 0)

toes = [3, 7, 11, 15]
revolute_joints=[1, 2, 5, 6, 9, 10, 13, 14, 0, 4, 8, 12]

for i in range(100):
    p.stepSimulation()

jointFrictionForce = 1
for joint in range(p.getNumJoints(robotId)):
    p.setJointMotorControl2(
        robotId,
        joint,
        p.POSITION_CONTROL,
        force=jointFrictionForce)

forceAppliedResetTime = 200
forceAppliedTimeCounter = 0
forcePosition = [0, 0, 0]
forceMagnitude = [0, 0, 0]

while True:
    pos, ori = p.getBasePositionAndOrientation(robotId, client)
    # print(pos[2])
    # print(ori)
    ang = p.getEulerFromQuaternion(ori)
    # print("{}\t{}\t{}".format(round(math.degrees(ang[0])), round(math.degrees(ang[1])), round(math.degrees(ang[2])))) 
    # print("{}\t{}\t{}".format(ang[0], ang[1], ang[2]))
    vel = p.getBaseVelocity(robotId, client)
    # print(vel[0][2])

    for dp in list(debug_param.keys()):
        info = p.getJointInfo(robotId, dp)
        p.setJointMotorControl2(
            robotId, 
            dp, 
            p.POSITION_CONTROL, 
            targetPosition = p.readUserDebugParameter(debug_param[dp]), 
            positionGain=1./12.,
            velocityGain=0.4,
            force=info[10],
            maxVelocity=info[11]
        )
    
    if forceAppliedTimeCounter > forceAppliedResetTime:
        forceAppliedTimeCounter = 0
        forceMagnitude = [np.random.uniform(-10, 10), np.random.uniform(-10, 10), np.random.uniform(-500, 50)]
        # forceMagnitude = [0, 0, np.random.uniform(-500, 50)]
        forcePosition = [np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), np.random.uniform(0, 0.5)]
        print(forceMagnitude, forcePosition)
    forceAppliedTimeCounter += 1

    p.applyExternalForce(robotId, -1, forceMagnitude, forcePosition, p.LINK_FRAME)
    
    p.stepSimulation()
    p.setGravity(0, 0, -9.8)

    # joint_state = p.getJointStates(robotId, jointIndices = revolute_joints)
    # position = [state[3] for state in joint_state]
    # print(sum(position))

    # find contact points of robot
    # contact_points = p.getContactPoints(robotId)
    # contact_joints = [cp[3] for cp in contact_points]
    # print(contact_joints)

    time.sleep(1./240.)
p.disconnect()