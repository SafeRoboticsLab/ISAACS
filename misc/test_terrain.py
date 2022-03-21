import pybullet as p
from numpy import random
import pybullet_data

client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)

plane = p.loadURDF("plane.urdf")

heightPerturbationRange = 0.08
numHeightfieldRows = 256
numHeightfieldColumns = 256

hf_id = 0
terrainShape = 0
heightfieldData = [0] * numHeightfieldRows * numHeightfieldColumns

heightPerturbationRange = heightPerturbationRange
for j in range(int(numHeightfieldColumns / 2)):
    for i in range(int(numHeightfieldRows / 2)):
        height = random.uniform(0, heightPerturbationRange)
        heightfieldData[2 * i +
                                2 * j * numHeightfieldRows] = height
        heightfieldData[2 * i + 1 +
                                2 * j * numHeightfieldRows] = height
        heightfieldData[2 * i + (2 * j + 1) *
                                numHeightfieldRows] = height
        heightfieldData[2 * i + 1 + (2 * j + 1) *
                                numHeightfieldRows] = height

terrainShape = p.createCollisionShape(
    shapeType=p.GEOM_HEIGHTFIELD,
    meshScale=[.07, .07, 1.6],
    heightfieldTextureScaling=(numHeightfieldRows - 1) / 2,
    heightfieldData = heightfieldData,
    numHeightfieldRows=numHeightfieldRows,
    numHeightfieldColumns=numHeightfieldColumns)
terrain = p.createMultiBody(0, terrainShape)
p.resetBasePositionAndOrientation(
    terrain, [0, 0, 0.0], [0, 0, 0, 1])
p.changeDynamics(terrain, -1, lateralFriction=1.0)

p.changeVisualShape(terrain, -1, rgbaColor=[1, 1, 1, 1])

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

while p.isConnected():
    p.stepSimulation()