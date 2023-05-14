# --------------------------------------------------------
# ISAACS: Iterative Soft Adversarial Actor-Critic for Safety
# https://arxiv.org/abs/2212.03228
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, duyn@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""Classes for basic neural network building blocks.

modified from: https://github.com/SafeRoboticsLab/SimLabReal/blob/main/agent/neural_network.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from torch.nn.utils import spectral_norm


class Sin(nn.Module):
  """
  Sin: Wraps element-wise `sin` activation as a nn.Module.

  Shape:
      - Input: `(N, *)` where `*` means, any number of additional dimensions
      - Output: `(N, *)`, same shape as the input

  Examples:
      >>> m = Sin()
      >>> input = torch.randn(2)
      >>> output = m(input)
  """

  def __init__(self):
    super().__init__()  # init the base class

  def forward(self, input):
    return torch.sin(input)  # simply apply already implemented sin


activation_dict = nn.ModuleDict({
    "ReLU": nn.ReLU(),
    "ELU": nn.ELU(),
    "Tanh": nn.Tanh(),
    "Sin": Sin(),
    "Identity": nn.Identity()
})


class MLP(nn.Module):
  """
  Construct a fully-connected neural network with flexible depth, width and
  activation function choices.
  """

  def __init__(
      self, dim_list, activation_type='Tanh', out_activation_type='Identity',
      use_ln=False, use_spec=False, use_bn=False, verbose=False
  ):
    """Initalizes the multilayer Perceptrons.

    Args:
        dim_list (int List): the dimension of each layer.
        activation_type (str, optional): type of activation layer. Support
            'Sin', 'Tanh' and 'ReLU'. Defaults to 'Tanh'.
        verbose (bool, optional): print info or not. Defaults to False.
    """
    super(MLP, self).__init__()

    # Construct module list: if use `Python List`, the modules are not
    # added to computation graph. Instead, we should use `nn.ModuleList()`.
    self.moduleList = nn.ModuleList()
    numLayer = len(dim_list) - 1
    for idx in range(numLayer):
      i_dim = dim_list[idx]
      o_dim = dim_list[idx + 1]

      linear_layer = nn.Linear(i_dim, o_dim)
      if use_spec:
        linear_layer = spectral_norm(linear_layer)
      if idx == 0:
        if use_ln:
          module = nn.Sequential(
              OrderedDict([
                  ('linear_1', linear_layer),
                  ('norm_1', nn.LayerNorm(o_dim)),
              ])
          )
        elif use_bn:
          module = nn.Sequential(
              OrderedDict([
                  ('linear_1', linear_layer),
                  ('norm_1', nn.BatchNorm1d(o_dim)),
                  ('act_1', activation_dict[activation_type]),
              ])
          )
        else:
          module = nn.Sequential(
              OrderedDict([
                  ('linear_1', linear_layer),
                  ('act_1', activation_dict[activation_type]),
              ])
          )
      elif idx == numLayer - 1:
        module = nn.Sequential(
            OrderedDict([
                ('linear_1', linear_layer),
                ('act_1', activation_dict[out_activation_type]),
            ])
        )
      else:
        if use_bn:
          module = nn.Sequential(
              OrderedDict([
                  ('linear_1', linear_layer),
                  ('norm_1', nn.BatchNorm1d(o_dim)),
                  ('act_1', activation_dict[activation_type]),
              ])
          )
        else:
          module = nn.Sequential(
              OrderedDict([
                  ('linear_1', linear_layer),
                  ('act_1', activation_dict[activation_type]),
              ])
          )

      self.moduleList.append(module)
    if verbose:
      print(self.moduleList)

  def forward(self, x):
    for m in self.moduleList:
      x = m(x)
    return x


class SpatialSoftmax(torch.nn.Module):

  def __init__(self, height, width, channel, data_format='NCHW'):
    super(SpatialSoftmax, self).__init__()
    self.data_format = data_format
    self.height = height
    self.width = width
    self.channel = channel

    pos_x, pos_y = np.meshgrid(
        np.linspace(-1., 1., self.height), np.linspace(-1., 1., self.width)
    )
    pos_x = torch.FloatTensor(pos_x.reshape(self.height * self.width))
    pos_y = torch.FloatTensor(pos_y.reshape(self.height * self.width))
    self.register_buffer('pos_x', pos_x)
    self.register_buffer('pos_y', pos_y)

  def forward(self, feature):
    # Output:
    #   (N, C*2) x_0 y_0 ...

    N = feature.shape[0]

    if self.data_format == 'NHWC':
      feature = feature.transpose(1, 3).tranpose(2, 3).view(
          -1, self.height * self.width
      )
    else:
      feature = feature.view(N, self.channel, self.height * self.width)

    softmax_attention = F.softmax(feature, dim=-1)

    # Sum over all pixels
    expected_x = torch.sum(
        self.pos_x * softmax_attention, dim=2, keepdim=False
    )
    expected_y = torch.sum(
        self.pos_y * softmax_attention, dim=2, keepdim=False
    )
    expected_xy = torch.cat([expected_x, expected_y], 1)

    return expected_xy


class ResidualBlock(nn.Module):

  def __init__(self, in_channels, out_channels, stride=1):
    super(ResidualBlock, self).__init__()
    self.conv1 = conv3x3(in_channels, out_channels, stride)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(out_channels, out_channels)

  def forward(self, x):
    residual = x
    out = self.conv1(x)
    out = self.relu(out)
    out = self.conv2(out)
    out += residual
    out = self.relu(out)
    return out


def conv3x3(in_channels, out_channels, stride=1):
  return nn.Conv2d(
      in_channels, out_channels, kernel_size=3, stride=stride, padding=1,
      bias=True
  )  # assume not using batchnorm so use bias


def conv2d_size_out(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
  from math import floor
  if type(kernel_size) is not tuple:
    kernel_size = (kernel_size, kernel_size)
  h = floor(((h_w[0] + (2*pad) - (dilation * (kernel_size[0] - 1)) - 1)
             / stride) + 1)
  w = floor(((h_w[1] + (2*pad) - (dilation * (kernel_size[1] - 1)) - 1)
             / stride) + 1)
  return h, w


class ConvNet(nn.Module):

  def __init__(
      self,
      input_n_channel=1,
      append_dim=0,
      cnn_kernel_size=[5, 3],
      cnn_stride=[2, 1],
      output_n_channel=[16, 32],
      img_size=128,
      use_sm=True,
      use_bn=True,
      use_spec=False,
      use_residual=False,
      verbose=True,
  ):

    super(ConvNet, self).__init__()

    self.append_dim = append_dim
    assert len(cnn_kernel_size) == len(output_n_channel), (
        "The length of the kernel_size list does not match with the "
        + "#channel list!"
    )
    self.n_conv_layers = len(cnn_kernel_size)

    if np.isscalar(img_size):
      height = img_size
      width = img_size
    else:
      height, width = img_size

    # Use ModuleList to store [] conv layers, 1 spatial softmax and [] MLP
    # layers.
    self.moduleList = nn.ModuleList()

    # = CNN: W' = (W - kernel_size + 2*padding) / stride + 1
    # Nx1xHxW -> Nx16xHxW -> Nx32xHxW
    for i, (kernel_size, stride, out_channels) in enumerate(
        zip(cnn_kernel_size, cnn_stride, output_n_channel)
    ):

      # Add conv
      padding = 0
      if i == 0:
        in_channels = input_n_channel
      else:
        in_channels = output_n_channel[i - 1]
      module = nn.Sequential()
      conv_layer = nn.Conv2d(
          in_channels=in_channels, out_channels=out_channels,
          kernel_size=kernel_size, stride=stride, padding=padding
      )
      if use_spec:
        conv_layer = spectral_norm(conv_layer)
      module.add_module("conv_1", conv_layer)

      # Add batchnorm
      if use_bn:
        module.add_module('bn_1', nn.BatchNorm2d(num_features=out_channels))

      # Always ReLU
      module.add_module('act_1', nn.ReLU())

      # Add module
      self.moduleList.append(module)

      # Add residual block, does not change shape
      if use_residual:
        self.moduleList.append(ResidualBlock(out_channels, out_channels))

      # Update height and width of images after modules
      height, width = conv2d_size_out([height, width], kernel_size, stride,
                                      padding)

    # = Spatial softmax, output 64 (32 features x 2d pos) or Flatten
    self.use_sm = use_sm
    if use_sm:
      module = nn.Sequential(
          OrderedDict([(
              'softmax',
              SpatialSoftmax(
                  height=height, width=width, channel=output_n_channel[-1]
              )
          )])
      )
      cnn_output_dim = int(output_n_channel[-1] * 2)
    else:
      module = nn.Sequential(OrderedDict([('flatten', nn.Flatten())]))
      cnn_output_dim = int(output_n_channel[-1] * height * width)
    self.moduleList.append(module)
    self.cnn_output_dim = cnn_output_dim

    if verbose:
      print(self.moduleList)

  def get_output_dim(self):
    return self.cnn_output_dim

  def forward(self, x):

    if x.dim() == 3:
      x = x.unsqueeze(1)  # Nx1xHxW
    for module in self.moduleList:
      x = module(x)
    return x
