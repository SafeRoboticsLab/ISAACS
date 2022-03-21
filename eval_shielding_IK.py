########################################
# FOR SAC RA POLICY
########################################
print("\n\n ** START INITIALIZING SAC RA POLICY ** \n\n")

import gym
import torch
from config.utils import load_config
from inverse_kinematics.inverse_kinematics_controller import (
    InverseKinematicsController
)
import spirit_rl
import time
import os
import numpy as np

from RARL.naive_rl import NaiveRL

import argparse

timestr = time.strftime("%Y-%m-%d-%H_%M")

from intelligent_trial_error.utils.arg_processing import Range, bool_type

#== ARGS ==
parser = argparse.ArgumentParser()

# additional env config information from boca/pexod gym
parser.add_argument(
    "-pg", "--pybulletGUI", help="PyBullet GUI", action="store_true"
)
parser.add_argument(
    "-pv", "--pybulletVerbose", help="PyBullet verbose", action="store_true"
)

# use -fa to manual control the force applied. If -fr is not chosen, then only
# a few force directions will be used (see the environment).
# If -fr is chosen, then random force will be applied, and magnitude will be
# proportional to -fa (uniform random * force)
parser.add_argument(
    "-fa", "--forceApplied", help="force applied", default=0.0, type=float
)
parser.add_argument(
    "-frt", "--forceResetTime", help="force reset time", default=20.0,
    type=float
)
parser.add_argument(
    "-fr", "--forceRandom", help="toggle random force", action="store_true"
)
# if this is chosen, it overwrites -fa. Current body mass is (2.495 +
# payload * 10.0), the force applied will be calculated as:
# force = (2.495 + payload * 10.0) * GRAVITY * args.forceBodyRatio, for GRAVITY
# = -9.81 m/s^2
parser.add_argument(
    "-fbr", "--forceBodyRatio", help="force based on body mass", default=0.0,
    type=float
)

parser.add_argument(
    "-rr", "--rotateReset", help="rotate sampling at reset",
    action="store_true"
)
parser.add_argument(
    "-hr", "--heightReset", help="height sampling at reset",
    action="store_true"
)
parser.add_argument(
    "-sh", "--shielding", help="apply shielding", action="store_true"
)
parser.add_argument(
    "-ov", "--override", help="apply override for shielding",
    action="store_true"
)
parser.add_argument(
    "-ep", "--epsilon", help="epsilon to apply shielding", default=0.25,
    type=float
)

parser.add_argument(
    "-i", "--iteration",
    help="iteration to run among all the trained iterations", type=int
)

# rollout
parser.add_argument("-ro", "--rollout", default=False, type=bool_type)
parser.add_argument(
    "-ros", "--rolloutStep", help="rollout steps to take", default=50, type=int
)

parser.add_argument('--agentPath', required=False, default="SAC_preTrained")

parser.add_argument(
    "--device", help="device to be used", default='cpu', type=str
)
parser.add_argument(
    "--terrain", help="terrain type: rough/normal", default="normal",
    choices=['normal', 'rough'], type=str
)
parser.add_argument(
    "--terrainHeight", help="terrain height perturbation range", default=0.05,
    type=float
)
parser.add_argument(
    "--terrainFriction", help="terrain friction", default=1.0, type=float
)

args = parser.parse_args()

CONFIG = load_config(os.path.join("config", "sample_naive.yaml"))
CONFIG_UPDATE = CONFIG["update"]
CONFIG_TRAIN = CONFIG["training"]
CONFIG_ARCH = CONFIG["arch"]
CONFIG_ENV = CONFIG["environment"]

#== CONFIGURATION ==
env_name = CONFIG_ENV.ENV_NAME

if args.device is not None:
  device = args.device
  CONFIG_UPDATE.DEVICE = args.device
  CONFIG_TRAIN.DEVICE = args.device

if CONFIG_UPDATE.DEVICE == "cpu":
  device = CONFIG_UPDATE.DEVICE
else:
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parent_dir = CONFIG_TRAIN.OUT_FOLDER

#== Environment ==
print("\n== Environment Information ==")
# OVERWRITE FORCE APPLIED
if args.forceBodyRatio != 0:
  force = (2.495 + args.payload * 10.0) * 9.81 * args.forceBodyRatio
  forceRandom = True
else:
  force = args.forceApplied
  forceRandom = args.forceRandom

env = gym.make(
    env_name, device=device, mode="RA", doneType="fail", force=force,
    gui=args.pybulletGUI, verbose=args.pybulletVerbose,
    rotateReset=args.rotateReset, heightReset=args.heightReset,
    forceResetTime=args.forceResetTime, forceRandom=forceRandom,
    terrain=args.terrain, terrainHeight=args.terrainHeight,
    terrainFriction=args.terrainFriction
)

#== Setting in this Environment ==
env.seed(CONFIG_TRAIN.SEED)

# If there is previously trained data, get checkpoint and resume
train_iteration_list = os.listdir(parent_dir)
print("Found latest trained iterations for evaluation")
latest_iteration = sorted(train_iteration_list)[-1]
if args.iteration is not None:
  latest_iteration = latest_iteration.split("_")[0] + "_" + str(
      int(args.iteration)
  ).zfill(2)
print("\tLatest iteration: {}".format(latest_iteration))
model_list = os.listdir(parent_dir + "/" + latest_iteration + "/model/critic/")
highest_number = sorted([
    int(a.split("-")[1].split(".")[0]) for a in model_list
])[-1]
print("\tHighest training number: {}".format(highest_number))

outFolder = parent_dir + "/" + latest_iteration
CONFIG_TRAIN.OUT_FOLDER = outFolder

agent = NaiveRL(CONFIG_TRAIN, CONFIG_UPDATE, CONFIG_ARCH, CONFIG_ENV)
agent.restore(highest_number, outFolder + "/model")

########################################
# FOR PERFORMANCE CONTROLLER
########################################
print("\n\n ** START INITIALIZING PERFORMANCE CONTROL POLICY ** \n\n")

# Select which solution to evaluate
controller = InverseKinematicsController()

while True:
  if args.shielding:
    # if shielding but not override, then the system will only print out when
    # then system needs shielding
    if args.rollout:
      raise NotImplementedError
    else:
      success = agent.shielding_with_IK(
          env, controller, epsilon=args.epsilon, override=args.override
      )
  else:
    # Run without shielding
    success = agent.shielding_with_IK(
        env, controller, epsilon=np.inf, override=args.override
    )
