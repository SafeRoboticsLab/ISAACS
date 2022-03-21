import gym
import torch
from config.utils import load_config
import spirit_rl
import time
import os

from RARL.naive_rl import NaiveRL

import argparse

timestr = time.strftime("%Y-%m-%d-%H_%M")

from utils import Range, bool_type

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
    "-ep", "--epsilon", help="epsilon to apply shielding", default=0.25,
    type=float
)

parser.add_argument(
    "-i", "--iteration",
    help="iteration to run among all the trained iterations", type=int
)

parser.add_argument("--device", help="device to be used", type=str)
parser.add_argument('--agentPath', required=False, default="SAC_preTrained")

# FOR LINC
# WHILE TRAINING SAFETY POLICY, USE THE ENVTYPE AND PAYLOAD TO CREATE THE
# SHIELDING POLICY FOR THE RESPECTIVE ENV
parser.add_argument('--loadpath', required=False)
parser.add_argument(
    '--envtype', default=None,
    choices=['normal', 'damaged', 'payload', 'spring']
)
parser.add_argument(
    '--payload', default=None, type=float, choices=Range(0.0, 1.0)
)
parser.add_argument(
    '--damage', default=None, nargs='*', help=[
        b'body_leg_0',
        b'leg_0_1_2',
        b'leg_0_2_3',
        b'body_leg_1',
        b'leg_1_1_2',
        b'leg_1_2_3',
        b'body_leg_2',
        b'leg_2_1_2',
        b'leg_2_2_3',
        b'body_leg_3',
        b'leg_3_1_2',
        b'leg_3_2_3',
        b'body_leg_4',
        b'leg_4_1_2',
        b'leg_4_2_3',
        b'body_leg_5',
        b'leg_5_1_2',
        b'leg_5_2_3',
    ]
)
parser.add_argument('--eplen', default=None, type=float)
parser.add_argument('--video', default=False, type=bool_type)
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
    payload=args.payload, envtype=args.envtype,
    forceResetTime=args.forceResetTime, forceRandom=forceRandom,
    video_output_file="evaluation.avi" if args.video else None
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

while True:
  agent.evaluate(env, args.epsilon)
