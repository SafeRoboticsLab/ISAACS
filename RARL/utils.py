# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

import torch
import os
import glob
import numpy as np


def get_bellman_update(
    mode, batch_size, device, next_q1, next_q2, non_final_mask, reward, g_x,
    l_x, binary_cost, gamma, terminal_type=None
):
  # max for reach-avoid Bellman equation, safety Bellman equation and risk
  # (recovery RL).
  if (mode == 'RA' or mode == 'safety' or mode == 'risk'):
    target_q = torch.max(next_q1, next_q2).view(-1)
  elif mode == 'performance':
    target_q = torch.min(next_q1, next_q2).view(-1)
  else:
    raise ValueError("Unsupported RL mode.")

  y = torch.zeros(batch_size).float().to(device)  # placeholder
  final_mask = torch.logical_not(non_final_mask)
  if mode == 'RA':
    # V(s) = max{ g(s), min{ ell(s), V(s') }}
    # Q(s, u) = V( f(s,u) ) = max{ g(s'), min{ ell(s'), min_{u'} Q(s', u')}}
    y[non_final_mask] = (
        (1.0-gamma) * torch.max(l_x[non_final_mask], g_x[non_final_mask])
        + gamma * torch.
        max(g_x[non_final_mask], torch.min(l_x[non_final_mask], target_q))
    )
    if terminal_type == 'g':
      y[final_mask] = g_x[final_mask]
    elif terminal_type == 'max':
      y[final_mask] = torch.max(l_x[final_mask], g_x[final_mask])
    else:
      raise ValueError("invalid terminal type")
  elif mode == 'safety':
    # V(s) = max{ g(s), V(s') }
    # Q(s, u) = V( f(s,u) ) = max{ g(s'), min_{u'} Q(s', u') }
    # normal state
    y[non_final_mask] = ((1.0-gamma) * g_x[non_final_mask]
                         + gamma * torch.max(g_x[non_final_mask], target_q))

    # terminal state
    y[final_mask] = g_x[final_mask]
  elif mode == 'performance':
    y = reward
    y[non_final_mask] += gamma * target_q
  elif mode == 'risk':
    y = binary_cost  # y = 1 if it's a terminal state
    y[non_final_mask] += gamma * target_q
  return y


def soft_update(target, source, tau):
  for target_param, param in zip(target.parameters(), source.parameters()):
    target_param.data.copy_(target_param.data * (1.0-tau) + param.data * tau)


def save_model(model, step, logs_path, types, MAX_MODEL):
  start = len(types) + 1
  os.makedirs(logs_path, exist_ok=True)
  model_list = glob.glob(os.path.join(logs_path, '*.pth'))
  
  if len(model_list) > MAX_MODEL - 1:
    min_step = min([int(li.split('/')[-1][start:-4]) for li in model_list])
    os.remove(os.path.join(logs_path, '{}-{}.pth'.format(types, min_step)))
  logs_path = os.path.join(logs_path, '{}-{}.pth'.format(types, step))
  torch.save(model.state_dict(), logs_path)
  print('  => Save {} after [{}] updates'.format(logs_path, step))


# == shielding ==
def check_shielding(  #TODO check in the spirit/hexapod
    backup, shield_dict, observation, action, append, context_backup=None,
    state=None, simulator=None, use_vec_env=False, policy=None,
    context_policy=None
):
  """
  Checks if shielding is needed. Currently, the latent is equivalent to
  context.

  Args:
      backup (object): a backup agent consisting of actor and critic
      shield_dict (dict): a dictionary consisting of shielding-related
          hyper-parameters.
      observation (np.ndarray or torch.tensor): the observation.
      action (np.ndarray or torch.tensor): action from the policy.
      append (np.ndarray or torch.tensor): the extra information that is
          appending after conv layers.
      context_backup (np.ndarray or torch.tensor, optional): the variable
          inducing policy distribution. It can be latent directly from a
          distribution or after encoder. Defaults to None.
      state (np.ndarray or torch.tensor): the real state. Defaults to None.
      simulator (object, optional): the environment on which we rollout
          trajectories. Defaults to None.

  Returns:
      torch.tensor: flags representing whether the shielding is necessary
  """
  if isinstance(state, np.ndarray):
    state = torch.FloatTensor(state).to(backup.device)
  if isinstance(observation, np.ndarray):
    observation = torch.from_numpy(observation).to(backup.device)
  back_to_numpy = False
  if isinstance(action, np.ndarray):
    action = torch.FloatTensor(action).to(backup.device)
    back_to_numpy = True
  if isinstance(append, np.ndarray):
    append = torch.FloatTensor(append).to(backup.device)

  # make sure the leading dim is the same
  if observation.dim() == 3:
    observation = observation.unsqueeze(0)
  if state is not None:
    if state.dim() == 1:
      state = state.unsqueeze(0)
  if action.dim() == 1:
    action = action.unsqueeze(0)
  if append.dim() == 1:
    append = append.unsqueeze(0)

  leading_equal = ((observation.shape[0] == action.shape[0])
                   and (observation.shape[0] == append.shape[0]))
  if state is not None:
    leading_equal = ((state.shape[0] == action.shape[0])
                     and (state.shape[0] == append.shape[0]))

  if not leading_equal:
    print(observation.shape, append.shape, action.shape)
    raise ValueError("The leading dimension is not the same!")
  shield_type = shield_dict['Type']

  if shield_type == 'value':
    if not backup.critic_has_act_ind:
      action = action[:, :-1]
    safe_value = backup.critic(
        observation, action, append=append, latent=context_backup
    )[0].data.squeeze(1)
    shield_flag = safe_value > shield_dict['Threshold']
    info = {}
  elif shield_type == 'rej':
    safe_thr = shield_dict['Threshold']
    max_resample = shield_dict['max_resample']
    cnt_resample = 0
    resample_flag = True
    while resample_flag:
      if cnt_resample == max_resample:  # resample budget
        break
      if not backup.critic_has_act_ind:
        action = action[:, :-1]
      safe_value = backup.critic(
          observation, action, append=append, latent=context_backup
      )[0].data.squeeze(1)
      shield_flag = (safe_value > safe_thr)
      resample_flag = torch.any(shield_flag)
      if resample_flag:
        if context_policy is not None:
          context_policy_resample = context_policy[shield_flag]
        else:
          context_policy_resample = None
        a_resample, _ = policy.actor.sample(
            observation[shield_flag], append=append[shield_flag],
            latent=context_policy_resample
        )
        if not backup.critic_has_act_ind:
          action[shield_flag] = a_resample.data.clone()
        else:
          action[shield_flag, :-1] = a_resample.data.clone()
        cnt_resample += 1
    if back_to_numpy:
      action_final = action.cpu().numpy()
    else:
      action_final = action.clone()
    info = {'action_final': action_final}
  elif shield_type == 'simulator' or shield_type == 'mixed':
    T_ro = shield_dict['T_rollout']
    state_np = state.cpu().numpy()
    action_np = action.cpu().numpy()
    # single env squeezes state_nxt to be of 1 dimension
    state_nxt = simulator.move_robot(action_np, state_np)
    if use_vec_env:
      _, results, states_final = simulator.simulate_trajectories(
          backup, mode=backup.mode, states=state_nxt, revert_task=False,
          sample_task=False, T=T_ro, end_type='fail'
      )
      results = torch.tensor(results).to(backup.device)
      info = {}
    else:
      traj_fwd, results, _, _ = simulator.simulate_one_trajectory(
          backup, mode=backup.mode, state=state_nxt, T=T_ro, end_type='fail',
          encoder=None
      )
      states_final = np.copy(traj_fwd[-1, :])
      info = {'traj_fwd': traj_fwd}
    shield_flag = (results == -1)

    if shield_type == 'mixed':  #! needs to update contexts with encoder
      safe_thr = shield_dict['Threshold']
      if use_vec_env:
        obs_final = simulator.get_obs(states_final)
      else:
        obs_final = torch.FloatTensor(simulator._get_obs(states_final)
                                     ).to(backup.device).unsqueeze(0)
      append_final = simulator.get_append(states_final)

      actions_final = backup.actor(
          obs_final, append=append_final, latent=context_backup
      ).data
      if backup.critic_has_act_ind:
        act_ind_rep = backup.act_ind.repeat(actions_final.shape[0], 1)
        actions_final = torch.cat((actions_final, act_ind_rep), dim=-1)
      safe_value = backup.critic(
          obs_final, actions_final, append=append_final, latent=context_backup
      )[0].data.squeeze(1)

      if use_vec_env:
        shield_flag = torch.logical_or(shield_flag, (safe_value > safe_thr))
      else:
        shield_flag = shield_flag or (safe_value > safe_thr)

  return shield_flag, info
