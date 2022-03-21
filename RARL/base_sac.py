# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )
#          Allen Z. Ren (allen.ren@princeton.edu)

from abc import ABC, abstractmethod
import os
import copy
import numpy as np
import torch
from torch.optim import Adam
from torch.optim import lr_scheduler

from .model import GaussianPolicy, TwinnedQNetwork
from .scheduler import StepLRMargin
from .utils import soft_update, save_model


class BaseSAC(ABC):

  def __init__(self, CONFIG, CONFIG_ARCH, CONFIG_ENV):
    """
    __init__: initialization.

    Args:
        CONFIG (object): update-related hyper-parameter configuration.
        CONFIG_ARCH (object): NN architecture configuration.
        CONFIG_ENV (object): environment configuration.
    """
    self.CONFIG = CONFIG
    self.CONFIG_ARCH = CONFIG_ARCH
    self.EVAL = CONFIG.EVAL
    self.mode = CONFIG.MODE

    # == ENV PARAM ==
    self.action_range = np.array(CONFIG_ENV.ACTION_RANGE)
    self.action_dim = CONFIG_ENV.ACTION_DIM
    self.state_dim = CONFIG_ENV.STATE_DIM

    # NN: device, action indicators
    self.device = CONFIG.DEVICE
    self.critic_has_act_ind = CONFIG_ARCH.CRITIC_HAS_ACT_IND
    if CONFIG_ARCH.ACT_IND is not None:
      self.act_ind = torch.FloatTensor(CONFIG_ARCH.ACT_IND).to(self.device)
      self.act_ind_dim = self.act_ind.shape[0]

    # == PARAM FOR TRAINING ==
    if not self.EVAL:
      self.terminal_type = CONFIG.TERMINAL_TYPE

      # NN
      self.batch_size = CONFIG.BATCH_SIZE

      # Learning Rate
      self.LR_A_SCHEDULE = CONFIG.LR_A_SCHEDULE
      self.LR_C_SCHEDULE = CONFIG.LR_C_SCHEDULE
      if self.LR_A_SCHEDULE:
        self.LR_A_PERIOD = CONFIG.LR_A_PERIOD
        self.LR_A_DECAY = CONFIG.LR_A_DECAY
        self.LR_A_END = CONFIG.LR_A_END
      if self.LR_C_SCHEDULE:
        self.LR_C_PERIOD = CONFIG.LR_C_PERIOD
        self.LR_C_DECAY = CONFIG.LR_C_DECAY
        self.LR_C_END = CONFIG.LR_C_END
      self.LR_C = CONFIG.LR_C
      self.LR_A = CONFIG.LR_A

      # Discount factor
      self.GAMMA_SCHEDULE = CONFIG.GAMMA_SCHEDULE
      if self.GAMMA_SCHEDULE:
        self.gamma_scheduler = StepLRMargin(
            init_value=CONFIG.GAMMA, period=CONFIG.GAMMA_PERIOD,
            decay=CONFIG.GAMMA_DECAY, end_value=CONFIG.GAMMA_END, goal_value=1.
        )
        self.GAMMA = self.gamma_scheduler.get_variable()
      else:
        self.GAMMA = CONFIG.GAMMA

      # Target Network Update
      self.TAU = CONFIG.TAU

      # alpha-related hyper-parameters
      self.init_alpha = CONFIG.ALPHA
      self.LEARN_ALPHA = CONFIG.LEARN_ALPHA
      self.log_alpha = torch.tensor(np.log(self.init_alpha)).to(self.device)
      self.target_entropy = -self.action_dim
      if self.LEARN_ALPHA:
        self.log_alpha.requires_grad = True
        self.LR_Al = CONFIG.LR_Al
        self.LR_Al_SCHEDULE = CONFIG.LR_Al_SCHEDULE
        if self.LR_Al_SCHEDULE:
          self.LR_Al_PERIOD = CONFIG.LR_Al_PERIOD
          self.LR_Al_DECAY = CONFIG.LR_Al_DECAY
          self.LR_Al_END = CONFIG.LR_Al_END

  # region: property
  @property
  def alpha(self):
    return self.log_alpha.exp()

  @property
  @abstractmethod
  def has_latent(self):
    raise NotImplementedError

  @property
  @abstractmethod
  def latent_dist(self):
    raise NotImplementedError

  # endregion

  # region: build models and optimizers
  def build_network(self, verbose=True, actor_path=None, critic_path=None):
    if self.critic_has_act_ind:
      critic_action_dim = self.action_dim + self.act_ind_dim
    else:
      critic_action_dim = self.action_dim

    self.critic = TwinnedQNetwork(
        state_dim=self.state_dim, mlp_dim=self.CONFIG_ARCH.DIM_LIST['critic'],
        action_dim=critic_action_dim, append_dim=self.CONFIG_ARCH.APPEND_DIM,
        latent_dim=self.CONFIG.LATENT_DIM,
        activation_type=self.CONFIG_ARCH.ACTIVATION['critic'],
        device=self.device, verbose=verbose
    )
    if verbose:
      print("\nThe actor shares the same encoder with the critic.")
    self.actor = GaussianPolicy(
        state_dim=self.state_dim, mlp_dim=self.CONFIG_ARCH.DIM_LIST['actor'],
        action_dim=self.action_dim, action_range=self.action_range,
        append_dim=self.CONFIG_ARCH.APPEND_DIM,
        latent_dim=self.CONFIG.LATENT_DIM,
        activation_type=self.CONFIG_ARCH.ACTIVATION['actor'],
        device=self.device, verbose=verbose
    )

    # Load model if specified
    if critic_path is not None:
      self.critic.load_state_dict(
          torch.load(critic_path, map_location=self.device)
      )
      print("--> Load critic wights from {}".format(critic_path))

    if actor_path is not None:
      self.actor.load_state_dict(
          torch.load(actor_path, map_location=self.device)
      )
      print("--> Load actor wights from {}".format(actor_path))

    # Copy for critic targer
    self.critic_target = copy.deepcopy(self.critic)

  def build_optimizer(self):
    print("Build basic optimizers.")
    self.critic_optimizer = Adam(self.critic.parameters(), lr=self.LR_C)
    self.actor_optimizer = Adam(self.actor.parameters(), lr=self.LR_A)

    if self.LR_C_SCHEDULE:
      self.critic_scheduler = lr_scheduler.StepLR(
          self.critic_optimizer, step_size=self.LR_C_PERIOD,
          gamma=self.LR_C_DECAY
      )
    if self.LR_A_SCHEDULE:
      self.actor_scheduler = lr_scheduler.StepLR(
          self.actor_optimizer, step_size=self.LR_A_PERIOD,
          gamma=self.LR_A_DECAY
      )

    if self.LEARN_ALPHA:
      self.log_alpha_optimizer = Adam([self.log_alpha], lr=self.LR_Al)
      if self.LR_Al_SCHEDULE:
        self.log_alpha_scheduler = lr_scheduler.StepLR(
            self.log_alpha_optimizer, step_size=self.LR_Al_PERIOD,
            gamma=self.LR_Al_DECAY
        )

  # endregion

  # region: update functions
  def reset_alpha(self):
    self.log_alpha = torch.tensor(np.log(self.init_alpha)).to(self.device)
    self.log_alpha.requires_grad = True
    self.log_alpha_optimizer = Adam([self.log_alpha], lr=self.LR_Al)
    self.log_alpha_scheduler = lr_scheduler.StepLR(
        self.log_alpha_optimizer, step_size=self.LR_Al_PERIOD,
        gamma=self.LR_Al_DECAY
    )

  def update_alpha_hyper_param(self):
    if self.LR_Al_SCHEDULE:
      lr = self.log_alpha_optimizer.state_dict()['param_groups'][0]['lr']
      if lr <= self.LR_Al_END:
        for param_group in self.log_alpha_optimizer.param_groups:
          param_group['lr'] = self.LR_Al_END
      else:
        self.log_alpha_scheduler.step()

  def update_critic_hyper_param(self):
    if self.LR_C_SCHEDULE:
      lr = self.critic_optimizer.state_dict()['param_groups'][0]['lr']
      if lr <= self.LR_C_END:
        for param_group in self.critic_optimizer.param_groups:
          param_group['lr'] = self.LR_C_END
      else:
        self.critic_scheduler.step()
    if self.GAMMA_SCHEDULE:
      old_gamma = self.gamma_scheduler.get_variable()
      self.gamma_scheduler.step()
      self.GAMMA = self.gamma_scheduler.get_variable()
      if self.GAMMA != old_gamma:
        self.reset_alpha()

  def update_actor_hyper_param(self):
    if self.LR_A_SCHEDULE:
      lr = self.actor_optimizer.state_dict()['param_groups'][0]['lr']
      if lr <= self.LR_A_END:
        for param_group in self.actor_optimizer.param_groups:
          param_group['lr'] = self.LR_A_END
      else:
        self.actor_scheduler.step()

  def update_hyper_param(self):
    self.update_critic_hyper_param()
    self.update_actor_hyper_param()
    if self.LEARN_ALPHA:
      self.update_alpha_hyper_param()

  def update_target_networks(self):
    soft_update(self.critic_target, self.critic, self.TAU)

  @abstractmethod
  def update_actor(self, batch):
    raise NotImplementedError

  @abstractmethod
  def update_critic(self, batch):
    raise NotImplementedError

  @abstractmethod
  def update(self, batch, timer, update_period=2):
    raise NotImplementedError

  # endregion

  # region: utils
  def save(self, step, logs_path, max_model=None):
    path_c = os.path.join(logs_path, 'critic')
    path_a = os.path.join(logs_path, 'actor')
    save_model(self.critic, step, path_c, 'critic', max_model)
    save_model(self.actor, step, path_a, 'actor', max_model)

  def remove(self, step, logs_path):
    path_c = os.path.join(logs_path, 'critic', 'critic-{}.pth'.format(step))
    path_a = os.path.join(logs_path, 'actor', 'actor-{}.pth'.format(step))
    print("Remove", path_a)
    print("Remove", path_c)
    if os.path.exists(path_c):
      os.remove(path_c)
    if os.path.exists(path_a):
      os.remove(path_a)

  @abstractmethod
  def value(self, obs, append):
    raise NotImplementedError

  @abstractmethod
  def check(self, env, cnt_step, states, verbose=True, **kwargs):
    raise NotImplementedError

  # endregion
