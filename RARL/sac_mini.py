# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )
#          Allen Z. Ren (allen.ren@princeton.edu)

from threading import active_count
import numpy as np
import torch
from torch.nn.functional import mse_loss

from .base_sac import BaseSAC
from .utils import get_bellman_update


class SAC_mini(BaseSAC):

  def __init__(self, CONFIG, CONFIG_ARCH, CONFIG_ENV):
    super().__init__(CONFIG, CONFIG_ARCH, CONFIG_ENV)

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
      self, build_optimizer=True, verbose=True, actor_path=None,
      critic_path=None
  ):
    super().build_network(
        verbose, actor_path=actor_path, critic_path=critic_path
    )

    # Set up optimizer
    if build_optimizer:
      super().build_optimizer()
    else:
      for _, param in self.actor.named_parameters():
        param.requires_grad = False
      for _, param in self.critic.named_parameters():
        param.requires_grad = False
      self.actor.eval()
      self.critic.eval()

  # endregion

  # region: main update functions
  def update_critic(self, batch):
    (
        non_final_mask, non_final_state_nxt, state, action, reward, g_x, l_x,
        _, append, non_final_append_nxt, binary_cost
    ) = batch

    # print(non_final_state_nxt.dtype)
    # print(action.dtype)
    # print(state.dtype)

    # input()

    self.critic.train()
    self.critic_target.eval()
    self.actor.eval()

    # Gets Q(s, a).
    q1, q2 = self.critic(
        state, action, append=append
    )  # Used to compute loss (non-target part).

    # Computes actor next_actions and feed to critic_target
    with torch.no_grad():
      next_actions, next_log_prob = self.actor.sample(
          non_final_state_nxt, append=non_final_append_nxt
      )
      if self.critic_has_act_ind:
        act_ind_rep = self.act_ind.repeat(next_actions.shape[0], 1)
        next_actions = torch.cat((next_actions, act_ind_rep), dim=-1)

      next_actions = next_actions.float()

      # print(non_final_append_nxt)

      next_q1, next_q2 = self.critic_target(
          non_final_state_nxt, next_actions, append=non_final_append_nxt
      )

    # Gets Bellman update.
    y = get_bellman_update(
        self.mode, self.batch_size, self.device, next_q1, next_q2,
        non_final_mask, reward, g_x, l_x, binary_cost, self.GAMMA,
        terminal_type=self.terminal_type
    )

    if self.mode == 'performance':
      y[non_final_mask] -= self.GAMMA * self.alpha * next_log_prob.view(-1)

    # Regresses MSE loss for both Q1 and Q2.
    loss_q1 = mse_loss(input=q1.view(-1), target=y)
    loss_q2 = mse_loss(input=q2.view(-1), target=y)
    loss_q = loss_q1 + loss_q2

    # Backpropagates.
    self.critic_optimizer.zero_grad()
    loss_q.backward()
    self.critic_optimizer.step()

    return loss_q.item()

  def update_actor(self, batch):
    state = batch[2]
    append = batch[8]

    self.critic.eval()
    self.actor.train()

    action_sample, log_prob = self.actor.sample(
        state, append=append
    )
    if self.critic_has_act_ind:
      act_ind_rep = self.act_ind.repeat(action_sample.shape[0], 1)
      action_sample = torch.cat((action_sample, act_ind_rep), dim=-1)

    # print(state.dtype)
    # print(action_sample.dtype)
    # input()

    action_sample = action_sample.float()

    q_pi_1, q_pi_2 = self.critic(
        state, action_sample, append=append
    )

    if self.mode == 'RA' or self.mode == 'safety' or self.mode == 'risk':
      q_pi = torch.max(q_pi_1, q_pi_2)
    elif self.mode == 'performance':
      q_pi = torch.min(q_pi_1, q_pi_2)

    # cost: min_theta E[ Q + alpha * (log pi + H)]
    # loss_pi = Q + alpha * log pi
    # reward: max_theta E[ Q - alpha * (log pi + H)]
    # loss_pi = -Q + alpha * log pi
    loss_entropy = log_prob.view(-1).mean()
    loss_q_eval = q_pi.mean()
    if self.mode == 'RA' or self.mode == 'safety' or self.mode == 'risk':
      loss_pi = loss_q_eval + self.alpha * loss_entropy
    elif self.mode == 'performance':
      loss_pi = -loss_q_eval + self.alpha * loss_entropy
    self.actor_optimizer.zero_grad()
    loss_pi.backward()
    self.actor_optimizer.step()

    # Automatic temperature tuning
    loss_alpha = (self.alpha *
                  (-log_prob - self.target_entropy).detach()).mean()
    if self.LEARN_ALPHA:
      self.log_alpha_optimizer.zero_grad()
      loss_alpha.backward()
      self.log_alpha_optimizer.step()
    return loss_pi.item(), loss_entropy.item(), loss_alpha.item()

  def update(self, batch, timer, update_period=2):
    self.critic.train()
    self.actor.train()

    loss_q = self.update_critic(batch)
    loss_pi, loss_entropy, loss_alpha = 0, 0, 0
    if timer % update_period == 0:
      loss_pi, loss_entropy, loss_alpha = self.update_actor(batch)
      self.update_target_networks()

    self.critic.eval()
    self.actor.eval()

    return loss_q, loss_pi, loss_entropy, loss_alpha

  # endregion

  # region: utils
  def value(self, state, append):
    action = self.actor(state, append=append)
    action = torch.from_numpy(action).to(self.device)
    if self.critic_has_act_ind:
      act_ind_rep = self.act_ind.repeat(action.shape[0], 1)
      action = torch.cat((action, act_ind_rep), dim=-1)

    q_pi_1, q_pi_2 = self.critic(state, action, append=append)
    value = (q_pi_1+q_pi_2) / 2
    return value

  # TODO: check in the spirit and hexapod simulator
  def check(self, env, cnt_step, states, verbose=True, **kwargs):
    if self.mode == 'safety' or self.mode == 'risk':
      end_type = 'fail'
    else:
      end_type = 'succ_or_fail'

    self.actor.eval()
    self.critic.eval()

    # TODO: customize it to fit in Spirit simulator
    results = env.simulate_trajectories(
        self, states=states, end_type=end_type, **kwargs
    )
    if self.mode == 'safety' or self.mode == 'risk':
      failure = np.sum(results == -1) / results.shape[0]
      success = 1 - failure
      train_progress = np.array([success, failure])
    else:
      success = np.sum(results == 1) / results.shape[0]
      failure = np.sum(results == -1) / results.shape[0]
      unfinish = np.sum(results == 0) / results.shape[0]
      train_progress = np.array([success, failure, unfinish])

    if verbose:
      print('\n{} policy after [{}] steps:'.format(self.mode, cnt_step))
      if self.mode == 'safety' or self.mode == 'risk':
        print('  - success/failure ratio:', end=' ')
      else:
        print('  - success/failure/unfinished ratio:', end=' ')
      with np.printoptions(formatter={'float': '{: .2f}'.format}):
        print(train_progress)
    self.actor.train()
    self.critic.train()

    return train_progress

  # endregion
