from typing import Optional, Dict, Tuple, Union, List, Any
import os
import copy
import time
import pickle
from collections import OrderedDict
import numpy as np
import torch as th
from torch.utils.data import DataLoader, Dataset
from torch.nn import MSELoss
import pytorch_lightning as pl

from agent.neural_network import MLP
from agent.model import get_mlp_input
from simulators.policy.base_policy import BasePolicy


# region: data
class ODPDataset(Dataset):

  def __init__(self, datapath: str, obs_type: str):
    super().__init__()
    with open(datapath, "rb") as f:
      dataset = pickle.load(f)
    if obs_type == "perfect":
      self.observations = np.asarray(dataset['states'], dtype=float)
    else:
      self.observations = np.asarray(dataset['observations'], dtype=float)
    self.controls = np.asarray(dataset['controls'], dtype=float)
    self.disturbances = np.asarray(dataset['disturbances'], dtype=float)
    self.values = np.asarray(dataset['values'], dtype=float)

  def __len__(self) -> int:
    return len(self.observations)

  def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        self.observations[idx], self.controls[idx], self.disturbances[idx],
        self.values[idx]
    )

  def collate_fn(
      self, batch_elems: List[Tuple[np.ndarray, np.ndarray, np.ndarray,
                                    np.ndarray]]
  ) -> Dict[str, th.Tensor]:
    observations = []
    controls = []
    disturbances = []
    values = []
    for obs, ctrl, dstb, val in batch_elems:
      observations.append(obs)
      controls.append(ctrl)
      disturbances.append(dstb)
      values.append(val)

    batch = {
        'observations': th.FloatTensor(np.asarray(observations)),
        'controls': th.FloatTensor(np.asarray(controls)),
        'disturbances': th.FloatTensor(np.asarray(disturbances)),
        'values': th.FloatTensor(np.asarray(values)).reshape(-1, 1)
    }
    return batch


class ODPDataModule(pl.LightningDataModule):

  def __init__(self, cfg):
    super().__init__()
    self.data_dir = cfg.DATASET_DIR
    self.cfg = cfg

  def setup(self, stage: Optional[str] = None):
    self.train_dataset = ODPDataset(
        os.path.join(self.data_dir, 'train.pkl'), self.cfg.obs_type
    )
    self.valid_dataset = ODPDataset(
        os.path.join(self.data_dir, 'valid.pkl'), self.cfg.obs_type
    )

  def train_dataloader(self):
    return DataLoader(
        dataset=self.train_dataset, shuffle=True,
        batch_size=self.cfg.train_batch_size,
        num_workers=self.cfg.train_num_data_workers, drop_last=False,
        collate_fn=self.train_dataset.collate_fn, persistent_workers=True
    )

  def val_dataloader(self):
    return DataLoader(
        dataset=self.valid_dataset, shuffle=False,
        batch_size=self.cfg.valid_batch_size,
        num_workers=self.cfg.valid_num_data_workers, drop_last=False,
        collate_fn=self.valid_dataset.collate_fn, persistent_workers=True
    )

  def test_dataloader(self):
    pass

  def predict_dataloader(self):
    pass


# endregion


# region: models
class ODPSupervisedModel(th.nn.Module):

  def __init__(self, cfg_algo) -> None:
    super().__init__()
    self.ctrl_range = th.FloatTensor(np.asarray(cfg_algo.action_range.actor_0))
    self.dstb_range = th.FloatTensor(np.asarray(cfg_algo.action_range.actor_1))
    self.ctrl_scale = (self.ctrl_range[:, 1] - self.ctrl_range[:, 0]) / 2.0
    self.ctrl_bias = (self.ctrl_range[:, 1] + self.ctrl_range[:, 0]) / 2.0
    self.dstb_scale = (self.dstb_range[:, 1] - self.dstb_range[:, 0]) / 2.0
    self.dstb_bias = (self.dstb_range[:, 1] + self.dstb_range[:, 0]) / 2.0

    self.obs_dim: int = cfg_algo.obs_dim
    self.ctrl_dim: int = self.ctrl_range.shape[0]
    self.dstb_dim: int = self.dstb_range.shape[0]
    self.dstb_obs_only: bool = cfg_algo.dstb_obs_only
    self.critic_obs_only: bool = cfg_algo.critic_obs_only
    if cfg_algo.dstb_obs_only:
      self.dstb_obs_dim = self.obs_dim
    else:
      self.dstb_obs_dim = self.obs_dim + self.ctrl_dim
    if cfg_algo.critic_obs_only:
      self.critic_obs_dim = self.obs_dim
    else:
      self.critic_obs_dim = self.obs_dim + self.ctrl_dim + self.dstb_dim

    ctrl_dim_list = ([self.obs_dim] + cfg_algo.mlp_dim.actor_0
                     + [self.ctrl_dim])
    dstb_dim_list = ([self.dstb_obs_dim] + cfg_algo.mlp_dim.actor_1
                     + [self.dstb_dim])
    critic_dim_list = ([self.critic_obs_dim] + cfg_algo.mlp_dim.critic + [1])
    if cfg_algo.tanh_output:
      out_activation_type = "Tanh"
    else:
      out_activation_type = "Identity"
    self.ctrl = MLP(
        ctrl_dim_list, cfg_algo.activation.actor,
        out_activation_type=out_activation_type,
        verbose=getattr(cfg_algo, "VERBOSE", True)
    )
    self.dstb = MLP(
        dstb_dim_list, cfg_algo.activation.actor,
        out_activation_type=out_activation_type,
        verbose=getattr(cfg_algo, "VERBOSE", True)
    )
    self.critic = MLP(
        critic_dim_list, cfg_algo.activation.critic,
        out_activation_type='Identity',
        verbose=getattr(cfg_algo, "VERBOSE", True)
    )

  def forward(self, batch) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    observations = batch['observations']
    ctrl = self.ctrl(observations)
    ctrl = (
        ctrl * self.ctrl_scale.to(ctrl.device)
        + self.ctrl_bias.to(ctrl.device)
    )

    if self.dstb_obs_only:
      dstb = self.dstb(observations)
    else:
      obs_ctrl = th.cat((observations, batch['controls']), dim=-1)
      dstb = self.dstb(obs_ctrl)
    dstb = (
        dstb * self.dstb_scale.to(dstb.device)
        + self.dstb_bias.to(dstb.device)
    )

    if self.critic_obs_only:
      values = self.critic(observations)
    else:
      obs_ctrl_dstb = th.cat(
          (observations, batch['controls'], batch['disturbances']), dim=-1
      )
      values = self.critic(obs_ctrl_dstb)
    return ctrl, dstb, values

  def compute_loss(self, batch) -> Dict:
    pred_ctrl, pred_dstb, pred_values = self.forward(batch)
    gt_ctrl = batch['controls']
    gt_dstb = batch['disturbances']
    gt_values = batch['values']

    loss_fn = MSELoss(reduction='none')
    loss_ctrl = loss_fn(input=pred_ctrl, target=gt_ctrl)
    loss_dstb = loss_fn(input=pred_dstb, target=gt_dstb)
    loss_critic = loss_fn(input=pred_values, target=gt_values)

    losses = OrderedDict(
        loss_ctrl=loss_ctrl, loss_dstb=loss_dstb, loss_critic=loss_critic
    )
    return losses


class ODPSupervisedModule(pl.LightningModule):

  def __init__(self, cfg_algo) -> None:
    super().__init__()
    self.cfg_algo = cfg_algo
    self.net = ODPSupervisedModel(cfg_algo)
    self.normalize_loss = cfg_algo.normalize_loss
    self.weights = {
        "loss_ctrl": th.FloatTensor([1.]),
        "loss_dstb": th.FloatTensor([1.]),
        "loss_critic": th.FloatTensor([1.]),
    }
    if self.normalize_loss:
      self.weights["loss_ctrl"] = 1 / (self.net.ctrl_scale**2)
      self.weights["loss_dstb"] = 1 / (self.net.dstb_scale**2)

  def forward(self, x):
    return self.net(x)

  @property
  def checkpoint_monitor_keys(self):
    return {"val_loss": "val_loss"}

  def training_step(self, batch, batch_idx):
    losses = self.net.compute_loss(batch)
    total_loss = 0.0
    for loss_key, loss in losses.items():
      weighted_loss_raw = loss * self.weights[loss_key].to(loss.device)
      weighted_loss = th.mean(weighted_loss_raw)
      self.log("train/" + loss_key, weighted_loss)
      total_loss += weighted_loss
    self.log("train_loss", total_loss)
    return total_loss

  def validation_step(self, batch, batch_idx):
    with th.no_grad():
      losses = self.net.compute_loss(batch)
      for loss_key, loss in losses.items():
        losses[loss_key] = th.mean(loss)
    return losses

  def validation_epoch_end(self, outputs) -> None:
    total_loss = 0.0
    for loss_key in outputs[0]:
      loss_avg = th.stack([o[loss_key] for o in outputs]).mean()
      self.log("val/" + loss_key, loss_avg)
      total_loss += loss_avg
    self.log("val_loss", total_loss)

  def configure_optimizers(self):
    return th.optim.AdamW(
        params=self.parameters(), lr=self.cfg_algo.lr_a,
        weight_decay=getattr(self.cfg_algo, "weight_decay", 0.01)
    )


# # This is for NNCS.
class ODPSupervisedPolicy(BasePolicy):

  def __init__(self, id: str, model: ODPSupervisedModule, config: Any):
    super().__init__(id, config)
    self.policy_type = "NNCS"

    # Constructs NNs.
    self.net = copy.deepcopy(model.net)
    self.net.to(self.device)

  def get_action(
      self, obs: np.ndarray,
      agents_action: Optional[Dict[str, np.ndarray]] = None, **kwargs
  ) -> Tuple[np.ndarray, dict]:
    """Gets the action to execute.

    Args:
        state (np.ndarray): current state.

    Returns:
        np.ndarray: the action to be executed.
        dict: info for the solver, e.g., processing time, status, etc.
    """
    if self.obs_other_list is not None:
      flat_action = np.concatenate([
          agents_action[k].copy() for k in self.obs_other_list
      ], axis=0)
      obs = np.concatenate((obs, flat_action), axis=0)
    append = kwargs.get("append", None)
    latent = kwargs.get("latent", None)

    time0 = time.time()
    obs, np_input, num_extra_dim = get_mlp_input(
        obs, action=None, append=append, latent=latent, device=self.device
    )
    action = self.net.ctrl(obs)
    action = (
        action * self.net.ctrl_scale.to(self.device)
        + self.net.ctrl_bias.to(self.device)
    )
    # Restore dimension
    for _ in range(num_extra_dim):
      action = action.squeeze(0)

    # Convert back to np
    if np_input:
      action = action.detach().cpu().numpy()
    t_process = time.time() - time0
    status = 1
    return action, dict(t_process=t_process, status=status)

  def to(self, device):
    super().to(device)
    self.net.to(self.device)


# endregion
