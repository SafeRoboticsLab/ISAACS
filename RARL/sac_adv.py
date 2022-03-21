# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )
#          Duy
import copy
import numpy as np
import torch
from torch.optim import Adam, lr_scheduler
from torch.nn.functional import mse_loss

from .base_ma_sac import BaseMASAC
from .model import TwinnedQNetwork
from .utils import get_bellman_update


class SAC_adv(BaseMASAC):

  def __init__(self, CONFIG, CONFIG_ARCH, CONFIG_ENV):
    super().__init__(CONFIG, CONFIG_ARCH, CONFIG_ENV)
    self.action_dim_ctrl = self.action_dim[0]
    self.action_dim_dstb = self.action_dim[1]
    assert self.actor_type[0] == "min", "Ctrl needs to minimize the cost!"
    assert self.actor_type[1] == "max", "Dstb needs to maximize the cost!"

  # region: property
  @property
  def has_latent(self):
    return False

  @property
  def latent_dist(self):
    return None

  # endregion

  # region: build models and optimizers
  def build_network(
      self, build_optimizer=True, verbose=True, ctrl_path=None, dstb_path=None,
      adv_critic_path=None, mean_critic_path=None
  ):

    # Builds central critic, ctrl actor, and dstb actor.
    actor_paths = [ctrl_path, dstb_path]
    super().build_network(
        self, verbose=verbose, actor_paths=actor_paths,
        critic_path=adv_critic_path
    )
    self.adv_critic = self.critic  # alias
    self.ctrl = self.actors[0]  # alias
    self.dstb = self.actors[1]  # alias

    # Builds an auxiliary critic (if no dstb for deployment). Assumes the same
    # architecture and activation functions as ones in the central critic.
    if self.critic_has_act_ind:
      mean_critic_action_dim = self.action_dim_ctrl + self.act_ind_dim
    else:
      mean_critic_action_dim = self.action_dim_ctrl
    self.mean_critic = TwinnedQNetwork(
        state_dim=self.state_dim, mlp_dim=self.CONFIG_ARCH.DIM_LIST['critic'],
        action_dim=mean_critic_action_dim,
        append_dim=self.CONFIG_ARCH.APPEND_DIM,
        latent_dim=self.CONFIG.LATENT_DIM,
        activation_type=self.CONFIG_ARCH.ACTIVATION['critic'],
        device=self.device, verbose=verbose
    )

    # Load model if specified
    if mean_critic_path is not None:
      self.mean_critic.load_state_dict(
          torch.load(mean_critic_path, map_location=self.device)
      )
      print("--> Load mean critic weights from {}".format(mean_critic_path))

    # Copy for critic targer
    self.adv_critic_target = copy.deepcopy(self.adv_critic)
    self.mean_critic_target = copy.deepcopy(self.mean_critic)

    # Set up optimizer
    if build_optimizer:
      self.build_optimizer()

  def build_optimizer(self):
    super().build_optimizer()
    self.adv_critic_optimizer = self.critic_optimizer  # alias
    self.mean_critic_optimizer = Adam(
        self.mean_critic.parameters(), lr=self.LR_C
    )

    self.ctrl_optimizer = self.actor_optimizers[0]  # alias
    self.dstb_optimizer = self.actor_optimizers[1]  # alias

    if self.LR_C_SCHEDULE:
      self.adv_critic_scheduler = self.critic_scheduler  # alias
      self.mean_critic_scheduler = lr_scheduler.StepLR(
          self.mean_critic_optimizer, step_size=self.LR_C_PERIOD,
          gamma=self.LR_C_DECAY
      )
    if self.LR_A_SCHEDULE:
      self.ctrl_scheduler = self.actor_schedulers[0]  # alias
      self.dstb_scheduler = self.actor_schedulers[1]  # alias

  # endregion

  # region: main update functions
  def _update_mean_critic(self, batch):
    # Gets transition information from the batch. Action is the concatenation
    # of the control and disturbance.
    (
        non_final_mask, non_final_state_nxt, state, action, reward, g_x, l_x,
        _, append, non_final_append_nxt, binary_cost
    ) = batch
    self.mean_critic.train()
    self.mean_critic_target.eval()
    self.ctrl.eval()

    # Gets Q(s, a).
    action = action[:, :self.action_dim_ctrl]
    q1, q2 = self.mean_critic(state, action, append=append)

    # Computes actor next_actions and feed to critic_target
    with torch.no_grad():
      next_actions, _ = self.ctrl.sample(
          non_final_state_nxt, append=non_final_append_nxt
      )
      # Appends action indicator if required.
      if self.critic_has_act_ind:
        act_ind_rep = self.act_ind.repeat(next_actions.shape[0], 1)
        next_actions = torch.cat((next_actions, act_ind_rep), dim=-1)
      next_q1, next_q2 = self.mean_critic_target(
          non_final_state_nxt, next_actions, append=non_final_append_nxt
      )

    #! Bellman update includes entropy from ctrl?
    # Gets Bellman update.
    y = get_bellman_update(
        self.mode, self.batch_size, self.device, next_q1, next_q2,
        non_final_mask, reward, g_x, l_x, binary_cost, self.GAMMA,
        terminal_type=self.terminal_type
    )

    # Regresses MSE loss for both Q1 and Q2.
    loss_q1 = mse_loss(input=q1.view(-1), target=y)
    loss_q2 = mse_loss(input=q2.view(-1), target=y)
    loss_q = loss_q1 + loss_q2

    # Backpropagates.
    self.mean_critic_optimizer.zero_grad()
    loss_q.backward()
    self.mean_critic_optimizer.step()

    self.ctrl.train()

    return loss_q.item()

  def _update_adv_critic(self, batch):
    # Gets transition information from the batch. Action is the concatenation
    # of the control and disturbance.
    (
        non_final_mask, non_final_state_nxt, state, action, reward, g_x, l_x,
        _, append, non_final_append_nxt, binary_cost
    ) = batch
    self.adv_critic.train()
    self.adv_critic_target.eval()
    self.ctrl.eval()
    self.dstb.eval()

    # Gets Q(s, a)
    q1, q2 = self.adv_critic(state, action, append=append)

    # Computes actor next_actions and feed to critic_target
    with torch.no_grad():
      next_actions_ctrl, _ = self.ctrl.sample(
          non_final_state_nxt, append=non_final_append_nxt
      )
      # Appends ctrl action after state for dstb.
      non_final_state_nxt_dstb = torch.cat(
          (non_final_state_nxt, next_actions_ctrl), dim=-1
      )
      next_actions_dstb, _ = self.dstb.sample(
          non_final_state_nxt_dstb, append=non_final_append_nxt
      )
      next_actions = torch.cat((next_actions_ctrl, next_actions_dstb), dim=-1)
      # Appends action indicator if required.
      if self.critic_has_act_ind:
        act_ind_rep = self.act_ind.repeat(next_actions.shape[0], 1)
        next_actions = torch.cat((next_actions, act_ind_rep), dim=-1)
      next_q1, next_q2 = self.adv_critic_target(
          non_final_state_nxt, next_actions, append=non_final_append_nxt
      )

    #! Bellman update includes entropy from ctrl and dstb?
    # Gets Bellman update.
    y = get_bellman_update(
        self.mode, self.batch_size, self.device, next_q1, next_q2,
        non_final_mask, reward, g_x, l_x, binary_cost, self.GAMMA,
        terminal_type=self.terminal_type
    )

    # Regresses MSE loss for both Q1 and Q2.
    loss_q1 = mse_loss(input=q1.view(-1), target=y)
    loss_q2 = mse_loss(input=q2.view(-1), target=y)
    loss_q = loss_q1 + loss_q2

    # Backpropagates.
    self.adv_critic_optimizer.zero_grad()
    loss_q.backward()
    self.adv_critic_optimizer.step()

    self.ctrl.train()
    self.dstb.train()

    return loss_q.item()

  def update_critic(self, batch):
    loss_q_adv = self._update_adv_critic(batch)
    loss_q_mean = self._update_mean_critic(batch)
    return loss_q_adv, loss_q_mean

  def update_actor(self, batch):
    state = batch[2]
    append = batch[8]

    self.critic.eval()
    self.ctrl.train()
    self.dstb.train()

    # Gets actions.
    action_ctrl, log_prob_ctrl = self.ctrl.sample(
        state, append=append, detach_encoder=True
    )
    # Appends ctrl action after state for dstb.
    state_dstb = torch.cat((state, action_ctrl), dim=-1)
    action_dstb, log_prob_dstb = self.dstb.sample(
        state_dstb, append=append, detach_encoder=True
    )
    action_sample = torch.cat((action_ctrl, action_dstb), dim=-1)
    if self.critic_has_act_ind:
      act_ind_rep = self.act_ind.repeat(action_sample.shape[0], 1)
      action_sample = torch.cat((action_sample, act_ind_rep), dim=-1)

    # Gets target values.
    q_pi_1, q_pi_2 = self.critic(
        state, action_sample, append=append, detach_encoder=True
    )

    if self.mode == 'RA' or self.mode == 'safety' or self.mode == 'risk':
      q_pi = torch.max(q_pi_1, q_pi_2)
    elif self.mode == 'performance':
      q_pi = torch.min(q_pi_1, q_pi_2)

    loss_q = q_pi.mean()
    loss_ent_ctrl = log_prob_ctrl.view(-1).mean()
    loss_ent_dstb = log_prob_dstb.view(-1).mean()
    if self.mode == 'RA' or self.mode == 'safety' or self.mode == 'risk':
      # cost: min_u max_d E[Q + alpha_u * log pi_u - alpha_d * log pi_d]
      loss_pi_ctrl = loss_q + self.alpha[0] * loss_ent_ctrl
      loss_pi_dstb = -loss_q + self.alpha[1] * loss_ent_dstb
    elif self.mode == 'performance':
      # reward: max_u min_d E[Q - alpha_u * log pi_u + alpha_d * log pi_d]
      loss_pi_ctrl = -loss_q + self.alpha[0] * loss_ent_ctrl
      loss_pi_dstb = loss_q + self.alpha[1] * loss_ent_dstb

    # Backpropagates.
    self.ctrl_optimizer.zero_grad()
    loss_pi_ctrl.backward()
    self.ctrl_optimizer.step()

    self.dstb_optimizer.zero_grad()
    loss_pi_dstb.backward()
    self.dstb_optimizer.step()

    # Tunes entropy temperature automatically.
    log_prob = torch.cat((log_prob_ctrl, log_prob_dstb), dim=-1).detach()
    loss_alpha = (self.alpha * (-log_prob - self.target_entropy)).mean()
    if self.LEARN_ALPHA:
      self.log_alpha_optimizer.zero_grad()
      loss_alpha.backward()
      self.log_alpha_optimizer.step()

    self.critic.train()
    return (
        loss_pi_ctrl.item(), loss_pi_dstb.item(), loss_ent_ctrl.item(),
        loss_ent_dstb.item(), loss_alpha.item()
    )

  def update(self, batch, timer, update_period=2):
    self.critic.train()
    for actor in self.actors:
      actor.train()

    loss_q = self.update_critic(batch)
    loss_pi = np.zeros(self.num_agents)
    loss_entropy = np.zeros(self.num_agents)
    loss_alpha = np.zeros(self.num_agents)
    if timer % update_period == 0:
      loss_pi, loss_entropy, loss_alpha = self.update_actor(batch)
      self.update_target_networks()

    self.critic.eval()
    for actor in self.actors:
      actor.eval()

    return loss_q, loss_pi, loss_entropy, loss_alpha

  # endregion

  # region: utils
  def value(self, state, append, use_adv=True):
    action_ctrl = self.ctrl(state, append=append)
    action_ctrl = torch.from_numpy(action_ctrl).to(self.device)

    if use_adv:
      state_dstb = torch.cat((state, action_ctrl), dim=-1)
      action_dstb = self.dstb(state_dstb, append=append)
      action = torch.cat((action_ctrl, action_dstb), dim=-1)
    else:
      action = action_ctrl
    if self.critic_has_act_ind:
      act_ind_rep = self.act_ind.repeat(action.shape[0], 1)
      action = torch.cat((action, act_ind_rep), dim=-1)

    if use_adv:
      q_pi_1, q_pi_2 = self.adv_critic(state, action, append=append)
    else:
      q_pi_1, q_pi_2 = self.mean_critic(state, action, append=append)
    value = (q_pi_1+q_pi_2) / 2
    return value

  # TODO
  def check(self, env, cnt_step, states, verbose=True, **kwargs):
    raise NotImplementedError

  # endregion
