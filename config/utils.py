import yaml


class Struct:

  def __init__(self, **kwargs):
    for key, value in kwargs.items():
      setattr(self, key, value)


KEYS = [
    'environment', 'training', 'arch_performance', 'arch_backup',
    'update_performance', 'update_backup'
]


def dict2object(dictionary, key):
  return Struct(**dictionary[key])


def load_config(file_path):
  with open(file_path) as f:
    data = yaml.safe_load(f)
  config_dict = {}
  for key, value in data.items():
    config_dict[key] = Struct(**value)
  return config_dict


def dump_config(file_path, objects, keys=KEYS):
  data = {}
  for key, object in zip(keys, objects):
    data[key] = object.__dict__
  with open(file_path, "w") as f:
    yaml.dump(data, f)


if __name__ == '__main__':
  import os
  import numpy as np
  import torch

  file_path = os.path.join("sample_naive.yaml")
  config_dict = load_config(file_path)
  config_env = config_dict['environment']
  config_training = config_dict['training']
  config_arch = config_dict['arch']
  config_update = config_dict['update']

  action_range = np.array(config_env.ACTION_RANGE)
  action_range = torch.from_numpy(action_range)
  if action_range.ndim == 1:
    action_range = action_range.unsqueeze(0)
  print(action_range[:, 0], action_range[:, 1])
