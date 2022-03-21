# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )
#          Duy

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


class BaseMASAC(ABC):
  """
  Implements the multi-agent SAC base class.
  """

  def __init__(self, CONFIG, CONFIG_ARCH, CONFIG_ENV):
    """
    __init__: initialization.

    Args:
        CONFIG (object): update-rekated hyper-parameter configuration.
        CONFIG_ARCH (object): NN architecture configuration.
        CONFIG_ENV (object): environment configuration.
    """
    self.CONFIG = CONFIG
    self.CONFIG_ARCH = CONFIG_ARCH
    self.EVAL = CONFIG.EVAL
    self.mode = CONFIG.MODE

    # == ENV PARAM ==
    self.action_range = np.array(CONFIG_ENV.ACTION_RANGE)  # np.ndarray
    self.action_dim = np.array(CONFIG_ENV.ACTION_DIM)  # vector
    self.num_agents = len(self.action_dim)
    self.action_dim_all = np.sum(self.action_dim)
    assert len(self.action_range) == self.num_agents, \
        "the number of agents is not consistent!"
    self.state_dim = CONFIG_ENV.STATE_DIM

    # NN: device, action indicators
    self.device = CONFIG.DEVICE
    self.critic_has_act_ind = CONFIG_ARCH.CRITIC_HAS_ACT_IND
    self.actor_type = CONFIG.ACTOR_TYPE  # a list of "min" or "max"
    assert len(self.actor_type) == self.num_agents, \
        "the number of agents is not consistent!"
    if CONFIG_ARCH.ACT_IND is not None:
      self.act_ind = torch.FloatTensor(CONFIG_ARCH.ACT_IND).to(self.device)
      self.act_ind_dim = self.act_ind.shape[0]

    # == PARAM FOR TRAINING ==
    # Assumes each agent has the same training hyper-parameters except alpha
    # for now.
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
      self.init_alpha = torch.ones(self.num_agents) * CONFIG.ALPHA
      self.LEARN_ALPHA = CONFIG.LEARN_ALPHA
      self.log_alpha = torch.log(self.init_alpha).to(self.device)
      self.target_entropy = -torch.tensor(self.action_dim).to(self.device)
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
  def build_actor(
      self, mlp_dim, action_dim, action_range, latent_dim, activation_type,
      verbose=True
  ):
    actor = GaussianPolicy(
        state_dim=self.state_dim, mlp_dim=mlp_dim, action_dim=action_dim,
        action_range=action_range, append_dim=self.CONFIG_ARCH.APPEND_DIM,
        latent_dim=latent_dim, activation_type=activation_type,
        device=self.device, verbose=verbose
    )

    return actor

  def build_network(self, verbose=True, actor_paths=None, critic_path=None):
    if self.critic_has_act_ind:
      critic_action_dim = self.action_dim_all + self.act_ind_dim
    else:
      critic_action_dim = self.action_dim_all

    self.critic = TwinnedQNetwork(
        state_dim=self.state_dim, mlp_dim=self.CONFIG_ARCH.DIM_LIST['critic'],
        action_dim=critic_action_dim, append_dim=self.CONFIG_ARCH.APPEND_DIM,
        latent_dim=self.CONFIG.LATENT_DIM,
        activation_type=self.CONFIG_ARCH.ACTIVATION['critic'],
        device=self.device, verbose=verbose
    )

    # Load model if specified
    if critic_path is not None:
      self.critic.load_state_dict(
          torch.load(critic_path, map_location=self.device)
      )
      print("--> Load central critic wights from {}".format(critic_path))

    # Copy for critic targer
    self.critic_target = copy.deepcopy(self.critic)

    if verbose:
      print("\nThe actor shares the same encoder with the critic.")
    self.actors = []
    for i in range(self.num_agents):
      actor = self.build_actor(
          mlp_dim=self.CONFIG_ARCH.MLP_DIM['actor'][i],
          action_dim=self.action_dim[i],
          action_range=self.action_range[i],
          #! below two args are assumed the same for now
          latent_dim=self.CONFIG.LATENT_DIM,
          activation_type=self.CONFIG_ARCH.ACTIVATION['actor'],
          verbose=verbose
      )
      if actor_paths[i] is not None:
        actor_path = actor_paths[i]
        actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        print("--> Load actor {} wights from {}".format(i, actor_path))
      self.actors.append(actor)

  def build_optimizer(self):
    print("Build basic optimizers.")
    # central critic
    self.critic_optimizer = Adam(self.critic.parameters(), lr=self.LR_C)
    if self.LR_C_SCHEDULE:
      self.critic_scheduler = lr_scheduler.StepLR(
          self.critic_optimizer, step_size=self.LR_C_PERIOD,
          gamma=self.LR_C_DECAY
      )

    # actors
    self.actor_optimizers = []
    if self.LR_A_SCHEDULE:
      self.actor_schedulers = []
    for i in range(self.num_agents):
      actor_optimizer = Adam(self.actors[i].parameters(), lr=self.LR_A)
      self.actor_optimizers.append(actor_optimizer)

      if self.LR_A_SCHEDULE:
        self.actor_schedulers.append(
            lr_scheduler.StepLR(
                actor_optimizer, step_size=self.LR_A_PERIOD,
                gamma=self.LR_A_DECAY
            )
        )

    # entropy temperature parameters
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
    self.log_alphaOptimizer = Adam([self.log_alpha], lr=self.LR_Al)
    self.log_alpha_scheduler = lr_scheduler.StepLR(
        self.log_alphaOptimizer, step_size=self.LR_Al_PERIOD,
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
      self.gamma_scheduler.step()
      self.GAMMA = self.gamma_scheduler.get_variable()

  def update_actor_hyper_param(self):
    if self.LR_A_SCHEDULE:
      for i in range(self.num_agents):
        actor_optimizer = self.actor_optimizers[i]
        lr = actor_optimizer.state_dict()['param_groups'][0]['lr']
        if lr <= self.LR_A_END:
          for param_group in actor_optimizer.param_groups:
            param_group['lr'] = self.LR_A_END
        else:
          self.actor_schedulers[i].step()

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
    save_model(self.critic, step, path_c, 'critic', max_model)
    for i in range(self.num_agents):
      path_a = os.path.join(logs_path, 'actor_' + str(i))
      save_model(self.actors[i], step, path_a, 'actor', max_model)

  def remove(self, step, logs_path):
    path_c = os.path.join(logs_path, 'critic', 'critic-{}.pth'.format(step))
    print("Remove", path_c)
    if os.path.exists(path_c):
      os.remove(path_c)

    for i in range(self.num_agents):
      path_a = os.path.join(
          logs_path, 'actor_' + str(i), 'actor-{}.pth'.format(step)
      )
      print("Remove", path_a)
      if os.path.exists(path_a):
        os.remove(path_a)

  @abstractmethod
  def value(self, obs, append):
    raise NotImplementedError

  @abstractmethod
  def check(self, env, cnt_step, states, verbose=True, **kwargs):
    raise NotImplementedError

  # endregion
