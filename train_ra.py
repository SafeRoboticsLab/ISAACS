import gym
import torch
from config.utils import load_config
import spirit_rl
import time
import os

from RARL.naive_rl import NaiveRL

import argparse

timestr = time.strftime("%Y-%m-%d-%H_%M")

import wandb
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
parser.add_argument(
    "-fa", "--forceApplied", help="force applied", default=0.0, type=float
)
parser.add_argument(
    "-rr", "--rotateReset", help="rotate sampling at reset",
    action="store_true"
)
parser.add_argument(
    "-hr", "--heightReset", help="height sampling at reset",
    action="store_true"
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
if not os.path.exists(parent_dir):
  os.makedirs(parent_dir)

iteration_count = len(os.listdir(parent_dir))

# WANDB
if CONFIG_TRAIN.USE_WANDB:
  wandb_project = "spirit-ra"
  wandb_name = "spirit-ra-{}-{}".format(iteration_count, timestr)

  wandb.init(entity='safe-princeton', project=wandb_project, name=wandb_name)

directory = parent_dir + '/' + env_name + '_{}/'.format(
    str(int(args.forceApplied)).zfill(2)
)
if not os.path.exists(directory):
  os.makedirs(directory)

outFolder = directory
figureFolder = os.path.join(outFolder, 'figure/')
os.makedirs(figureFolder, exist_ok=True)

# New OUT_FOLDER based on the current run
CONFIG_TRAIN.OUT_FOLDER = outFolder

#== Environment ==
env = gym.make(
    env_name, device=device, mode="RA", doneType=CONFIG_ENV.DONE_TYPE,
    force=args.forceApplied, gui=args.pybulletGUI,
    verbose=args.pybulletVerbose, rotateReset=args.rotateReset,
    heightReset=args.heightReset, payload=args.payload, envtype=args.envtype,
    max_train_steps=CONFIG_ENV.MAX_TRAIN_STEPS
)

#== Setting in this Environment ==
env.seed(CONFIG_TRAIN.SEED)

agent = NaiveRL(CONFIG_TRAIN, CONFIG_UPDATE, CONFIG_ARCH, CONFIG_ENV)

# If there is previously trained data, get checkpoint and resume
train_iteration_list = os.listdir(parent_dir)
if len(train_iteration_list) > 1:
  print("Found previously trained iterations, get the latest one")
  latest_iteration = sorted(train_iteration_list)[-2]
  print("\tLatest iteration: {}".format(latest_iteration))
  model_list = os.listdir(
      parent_dir + "/" + latest_iteration + "/model/critic/"
  )
  highest_number = sorted([
      int(a.split("-")[1].split(".")[0]) for a in model_list
  ])[-1]
  print("\tHighest training number: {}".format(highest_number))
  agent.restore(highest_number, parent_dir + "/" + latest_iteration + "/model")

trainRecords, trainProgress, _, _ = agent.learn(env)
