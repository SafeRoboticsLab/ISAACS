import pybullet as p
import pybullet_data
import time
import numpy as np
import math

from spirit_rl.resources.spirit import Spirit

client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
dt = 1./240.
# dt = 1.

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
plane = p.loadURDF("plane.urdf")
# robot = p.loadURDF("spirit40.urdf",[0,0,1.0], p.getQuaternionFromEuler([1.0, 1.0, 1.0]), useFixedBase=False)
# robot = p.loadURDF("spirit40.urdf",[0,0,0.5], p.getQuaternionFromEuler([0.0, 0.0, 0.0]), useFixedBase=False)
robot = Spirit(client, 0.5, p.getQuaternionFromEuler([0.0, 0.0, 0.0]))
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
p.setGravity(0,0,-9.8, physicsClientId = client)

toes=[3, 7, 11, 15]
revolute_joints=[1, 2, 5, 6, 9, 10, 13, 14, 0, 4, 8, 12]

# joint_positions = [
#   0.74833727,  1.4664032,   
#   0.76712584,  1.4589894,   
#   0.74642015,  1.4782903,
#   0.7488482,   1.4464638 , 
#   -0.01239824 ,-0.02952504,  0.02939725,  0.05764484
# ]

# joint_positions = [
#   1, 1.9, 
#   1, 1.9, 
#   1, 1.9, 
#   1, 1.9, 
#   0, 0, 0, 0
# ]

# joint_positions = [
#   0.75, 1.45,
#   0.75, 1.45,
#   0.75, 1.45,
#   0.75, 1.45,
#   0, 0, 0, 0
# ]

# joint_positions = [
#   0.3, 1.2,
#   0.3, 1.2,
#   0.3, 1.2,
#   0.3, 1.2,
#   0.3, 0.3, -0.3, -0.3
# ]

joint_positions = [
  1.57, 3.14,
  1.57, 3.14,
  1.57, 3.14,
  1.57, 3.14,
  0, 0, 0, 0
]

# joint_positions = [
#   0.4, 1.0,
#   0.4, 1.0,
#   0.4, 1.0,
#   0.4, 1.0,
#   0, 0, 0, 0
# ]

for j in range (p.getNumJoints(robot.robot)):
  print("j=", p.getJointInfo(robot.robot,j))

for j in range (12):
  joint_index = revolute_joints[j]
  print("revolute_joint index=", joint_index)
  p.resetJointState(robot.robot, joint_index, joint_positions[j])  
  # p.setJointMotorControl2(robot,joint_index,p.POSITION_CONTROL, joint_positions[j], force = 10)
  info = p.getJointInfo(robot.robot, joint_index)
  p.setJointMotorControl2(
    robot.robot, 
    joint_index, 
    p.POSITION_CONTROL, 
    targetPosition = joint_positions[j], 
    positionGain=1./12.,
    velocityGain=0.4,
    force=info[10],
    maxVelocity=info[11]
  )
  print(info[10], info[11])

count = 0
while p.isConnected():
  # p.applyExternalForce(robot, -1, np.random.rand(3)*50, np.random.rand(3), p.LINK_FRAME)
  # get height of robot from ground
  pos, ang = p.getBasePositionAndOrientation(robot.robot, client)
  print("Height: {:.5f}\tTarget: {:.5f}\tSafe: {:.5f}".format(pos[2], robot.target_margin(), robot.safety_margin()))

  # # get joint positions from ground
  # joint_state = p.getJointStates(robot, jointIndices = revolute_joints)
  # position = [state[0] for state in joint_state]
  # print(position)

  # print robot orientation
  # pos, ang = p.getBasePositionAndOrientation(robot, client)
  # ang = p.getEulerFromQuaternion(ang)
  # print(pos[2])
  # print("{:.2f}\t{:.2f}\t{:.2f}".format(ang[0], ang[1], ang[2]))

  # find contact points of robot
  # contact_points = p.getContactPoints(robot)
  # contact_joints = [cp[3] for cp in contact_points]
  # print(contact_joints)

  # print("{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(
  #   pos[2] - 0.175 * math.sin(ang[0]) - 0.13 * math.sin(ang[1]),
  #   pos[2] - 0.175 * math.sin(-ang[0]) - 0.13 * math.sin(ang[1]),
  #   pos[2] - 0.175 * math.sin(ang[0]) - 0.13 * math.sin(-ang[1]),
  #   pos[2] - 0.175 * math.sin(-ang[0]) - 0.13 * math.sin(-ang[1]))
  # )

  # closest_shoulder_height = pos[2] - max(
  #     0.175 * math.sin(ang[0]) + 0.13 * math.sin(ang[1]),
  #     0.175 * math.sin(-ang[0]) + 0.13 * math.sin(ang[1]),
  #     0.175 * math.sin(ang[0]) + 0.13 * math.sin(-ang[1]),
  #     0.175 * math.sin(-ang[0]) + 0.13 * math.sin(-ang[1])
  # )
  
  # print(closest_shoulder_height)

  vel = p.getBaseVelocity(robot.robot, client)[0][:]
  # print(np.array(vel) > 1.0)

  count+=1
  p.stepSimulation()
  time.sleep(dt)
print("sitting")
