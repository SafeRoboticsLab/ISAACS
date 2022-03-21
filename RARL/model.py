# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from .neural_network import MLP


def tie_weights(src, trg):
  assert type(src) == type(trg)
  trg.weight = src.weight
  trg.bias = src.bias


def get_mlp_input(
    state, action=None, append=None, latent=None, device=torch.device("cpu")
):
  """
  Transforms inputs of the Q-network or policy into torch tensor on the desired
  device. Concatenates action, append, latent if provided.

  Args:
      state (np.ndarray or torch.tensor): state of the system.
      action (np.ndarray or torch.tensor, optional): action taken. Defaults to
          None.
      append (np.ndarray or torch.tensor, optional): extra information appended
          to the MLP. Defaults to None.
      latent (np.ndarray or torch.tensor, optional): information about the
          environment. Defaults to None.
      device (torch.device, optional): torch device. Defaults to
          torch.device("cpu").

  Returns:
      torch.tensor: input to the Q-network or policy.
      bool: cast the output to numpy if True.
      int: the number of extra dimension to be squeezed.
  """
  np_input = False
  if isinstance(state, np.ndarray):
    state = torch.from_numpy(state).float().to(device)
    np_input = True
  if isinstance(action, np.ndarray):
    action = torch.from_numpy(action).float().to(device)
  if isinstance(append, np.ndarray):
    append = torch.from_numpy(append).float().to(device)
  if isinstance(latent, np.ndarray):
    latent = torch.from_numpy(latent).float().to(device)
  state = state.to(device)

  num_extra_dim = 0
  if state.dim() == 1:  # running policy deterministically at test time
    state = state.unsqueeze(0)
    if action is not None:
      action = action.unsqueeze(0)
    if append is not None:
      append = append.unsqueeze(0)
    if latent is not None:
      latent = latent.unsqueeze(0)
    num_extra_dim += 1

  if append is not None:
    state = torch.cat((state, append), dim=-1)
  if latent is not None:
    state = torch.cat((state, latent), dim=-1)
  if action is not None:
    state = torch.cat((state, action), dim=-1)

  return state, np_input, num_extra_dim


class TwinnedQNetwork(nn.Module):

  def __init__(
      self, state_dim, mlp_dim, action_dim, append_dim=0, latent_dim=0,
      activation_type='Tanh', device='cpu', verbose=True
  ):
    super(TwinnedQNetwork, self).__init__()
    if verbose:
      print("The neural networks for CRITIC have the architecture as below:")
    dim_list = [state_dim+action_dim+append_dim+latent_dim] + mlp_dim + [1]
    self.Q1 = MLP(dim_list, activation_type, verbose=verbose).to(device)
    self.Q2 = copy.deepcopy(self.Q1)

    if device == torch.device('cuda'):
      self.Q1.cuda()
      self.Q2.cuda()
    self.device = device

  def forward(self, state, action, append=None, latent=None):
    state, np_input, num_extra_dim = get_mlp_input(
        state, action=action, append=append, latent=latent, device=self.device
    )
    q1 = self.Q1(state)
    q2 = self.Q2(state)

    # Restore dimension
    for _ in range(num_extra_dim):
      q1 = q1.squeeze(0)
      q2 = q2.squeeze(0)

    if np_input:
      q1 = q1.detach().cpu().numpy()
      q2 = q2.detach().cpu().numpy()
    return q1, q2


class GaussianPolicy(nn.Module):

  def __init__(
      self, state_dim, mlp_dim, action_dim, action_range, append_dim=0,
      latent_dim=0, activation_type='Tanh', device='cpu', verbose=True
  ):
    super(GaussianPolicy, self).__init__()
    dim_list = [state_dim+append_dim+latent_dim] + mlp_dim + [action_dim]
    self.device = device
    if verbose:
      print("The neural network for MEAN has the architecture as below:")
    self.mean = MLP(
        dim_list, activation_type, out_activation_type="Tanh", verbose=verbose
    ).to(device)
    if verbose:
      print("The neural network for LOG_STD has the architecture as below:")
    self.log_std = MLP(
        dim_list, activation_type, out_activation_type="Identity",
        verbose=verbose
    ).to(device)

    if isinstance(action_range, np.ndarray):
      action_range = torch.from_numpy(action_range).to(self.device)
    if action_range.dim() == 1:
      action_range = action_range.unsqueeze(0)

    self.a_max = action_range[:, 1]
    self.a_min = action_range[:, 0]
    self.scale = (self.a_max - self.a_min) / 2.0
    self.bias = (self.a_max + self.a_min) / 2.0

    self.LOG_STD_MAX = 1
    self.LOG_STD_MIN = -10
    self.eps = 1e-8

  def forward(self, state, append=None, latent=None):
    state, np_input, num_extra_dim = get_mlp_input(
        state, action=None, append=append, latent=latent, device=self.device
    )
    output = self.mean(state)
    output = output * self.scale + self.bias

    # Restore dimension
    for _ in range(num_extra_dim):
      output = output.squeeze(0)

    # Convert back to np
    if np_input:
      output = output.detach().cpu().numpy()

    return output

  def sample(self, state, append=None, latent=None):
    state, _, num_extra_dim = get_mlp_input(
        state, action=None, append=append, latent=latent, device=self.device
    )
    mean = self.mean(state)
    log_std = self.log_std(state)
    log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

    std = torch.exp(log_std)
    normalRV = Normal(mean, std)

    x = normalRV.rsample()  # reparameterization trick (mean + std * N(0,1))
    y = torch.tanh(x)  # constrain the output to be within [-1, 1]

    action = y * self.scale + self.bias
    log_prob = normalRV.log_prob(x)

    # Get the correct probability: x -> a, a = c * y + b, y = tanh x
    # followed by: p(a) = p(x) x |det(da/dx)|^-1
    # log p(a) = log p(x) - log |det(da/dx)|
    # log |det(da/dx)| = sum log (d a_i / d x_i)
    # d a_i / d x_i = c * ( 1 - y_i^2 )
    log_prob -= torch.log(self.scale * (1 - y.pow(2)) + self.eps)
    if log_prob.dim() > 1:
      log_prob = log_prob.sum(1, keepdim=True)
    else:
      log_prob = log_prob.sum()

    # Restore dimension
    for _ in range(num_extra_dim):
      action = action.squeeze(0)
      log_prob = log_prob.squeeze(0)

    return action, log_prob
