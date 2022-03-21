# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )
#          Allen Z. Ren (allen.ren@princeton.edu)

from abc import ABC, abstractmethod
from collections import namedtuple
from queue import PriorityQueue
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from .replay_memory import ReplayMemory

Transition = namedtuple('Transition', ['s', 'a', 'r', 's_', 'done', 'info'])
TransitionLatent = namedtuple(
    'TransitionLatent', ['z', 's', 'a', 'r', 's_', 'done', 'info']
)


class BaseTraining(ABC):

  def __init__(self, CONFIG, CONFIG_ENV, CONFIG_UPDATE):
    super(BaseTraining, self).__init__()

    self.device = CONFIG.DEVICE
    self.n_envs = CONFIG.NUM_CPUS
    self.action_dim = CONFIG_ENV.ACTION_DIM
    self.CONFIG = CONFIG
    self.MAX_TRAIN_STEPS = CONFIG_ENV.MAX_TRAIN_STEPS
    self.MAX_EVAL_STEPS = CONFIG_ENV.MAX_EVAL_STEPS
    # self.NUM_VISUALIZE_TASK = CONFIG.NUM_VISUALIZE_TASK

    #! We assume backup and performance use the same parameters.
    self.batch_size = CONFIG_UPDATE.BATCH_SIZE
    self.UPDATE_PERIOD = CONFIG_UPDATE.UPDATE_PERIOD
    self.MAX_MODEL = CONFIG_UPDATE.MAX_MODEL

    # memory
    self.build_memory(CONFIG.MEMORY_CAPACITY, CONFIG.SEED)
    self.rng = np.random.default_rng(seed=CONFIG.SEED)

    # saving models
    self.save_top_k = self.CONFIG.SAVE_TOP_K
    self.pq_top_k = PriorityQueue()

    self.use_wandb = CONFIG.USE_WANDB

  @property
  @abstractmethod
  def has_backup(self):
    raise NotImplementedError

  def build_memory(self, capacity, seed):
    self.memory = ReplayMemory(capacity, seed)

  def sample_batch(self, batch_size=None, recent_size=0):
    if batch_size is None:
      batch_size = self.batch_size
    if recent_size > 0:  # use recent
      transitions = self.memory.sample_recent(batch_size, recent_size)
    else:
      transitions = self.memory.sample(batch_size)
    batch = Transition(*zip(*transitions))
    return batch

  def store_transition(self, *args):
    self.memory.update(Transition(*args))

  # def store_transition_latent(self, *args):
  #   self.memory.update(TransitionLatent(*args))

  def unpack_batch(
      self, batch, get_append=False, get_latent=False, get_perf_action=False
  ):
    non_final_mask = torch.tensor(
        tuple(map(lambda s: not s, batch.done)), dtype=torch.bool
    ).view(-1).to(self.device)
    non_final_state_nxt = torch.cat([
        s for done, s in zip(batch.done, batch.s_) if not done
    ]).to(self.device)
    state = torch.cat(batch.s).to(self.device)

    reward = torch.FloatTensor(batch.r).to(self.device)

    g_x = torch.FloatTensor([info['g_x'] for info in batch.info]
                           ).to(self.device).view(-1)
    l_x = torch.FloatTensor([info['l_x'] for info in batch.info]
                           ).to(self.device).view(-1)

    if get_perf_action:  # recovery RL separates a_shield and a_perf.
      if batch.info[0]['a_perf'].dim() == 1:
        action = torch.FloatTensor([info['a_perf'] for info in batch.info])
      else:
        action = torch.cat([info['a_perf'] for info in batch.info])
      action = action.to(self.device)
    else:
      action = torch.cat(batch.a).to(self.device)
    
    action = action.float()
    state = state.float()
    non_final_state_nxt = non_final_state_nxt.float()

    latent = None
    if get_latent:
      latent = torch.cat(batch.z).to(self.device)

    append = None
    non_final_append_nxt = None
    if get_append:
      append = torch.cat([info['append'] for info in batch.info]
                        ).to(self.device)
      non_final_append_nxt = torch.cat([
          info['append_nxt'] for info in batch.info
      ]).to(self.device)[non_final_mask]

    binary_cost = torch.FloatTensor([
        info['binary_cost'] for info in batch.info
    ])
    binary_cost = binary_cost.to(self.device).view(-1)

    return (
        non_final_mask, non_final_state_nxt, state, action, reward, g_x, l_x,
        latent, append, non_final_append_nxt, binary_cost
    )

  def save(self, metric=None, force_save=False):
    assert metric is not None or force_save, \
        "should provide metric of force save"
    save_current = False
    if force_save:
      save_current = True
    elif self.pq_top_k.qsize() < self.save_top_k:
      self.pq_top_k.put((metric, self.cnt_step))
      save_current = True
    elif metric > self.pq_top_k.queue[0][0]:  # overwrite
      # Remove old one
      _, step_remove = self.pq_top_k.get()
      for module, module_folder in zip(
          self.module_all, self.module_folder_all
      ):
        module.remove(int(step_remove), module_folder)
      self.pq_top_k.put((metric, self.cnt_step))
      save_current = True

    if save_current:
      print()
      print('Saving current model...')
      for module, module_folder in zip(
          self.module_all, self.module_folder_all
      ):
        module.save(self.cnt_step, module_folder, self.MAX_MODEL)
      print(self.pq_top_k.queue)

  def restore(self, step, logs_path, agent_type, actor_path=None):
    """Restore the weights of the neural network.

    Args:
        step (int): #updates trained.
        logs_path (str): the path of the directory, under this folder there
            should be critic/ and agent/ folders.
        agent_type (str): performance or backup.
    """
    model_folder = path_c = os.path.join(logs_path, agent_type)
    path_c = os.path.join(model_folder, 'critic', 'critic-{}.pth'.format(step))
    if actor_path is not None:
      path_a = actor_path
    else:
      path_a = os.path.join(model_folder, 'actor', 'actor-{}.pth'.format(step))
    if agent_type == 'backup':
      self.backup.critic.load_state_dict(
          torch.load(path_c, map_location=self.device)
      )
      self.backup.critic.to(self.device)
      self.backup.critic_target.load_state_dict(
          torch.load(path_c, map_location=self.device)
      )
      self.backup.critic_target.to(self.device)
      self.backup.actor.load_state_dict(
          torch.load(path_a, map_location=self.device)
      )
      self.backup.actor.to(self.device)
    elif agent_type == 'performance':
      self.performance.critic.load_state_dict(
          torch.load(path_c, map_location=self.device)
      )
      self.performance.critic.to(self.device)
      self.performance.critic_target.load_state_dict(
          torch.load(path_c, map_location=self.device)
      )
      self.performance.critic_target.to(self.device)
      self.performance.actor.load_state_dict(
          torch.load(path_a, map_location=self.device)
      )
      self.performance.actor.to(self.device)
    print(
        '  <= Restore {} with {} updates from {}.'.format(
            agent_type, step, model_folder
        )
    )

  # TODO: check in the spirit and hexapod simulator
  # def get_check_states(self, env, num_rnd_traj):

  # TODO: check in the spirit and hexapod simulator
  def get_figures(
      self, venv, env, plot_v, vmin, vmax, save_figure, plot_figure,
      figure_folder_perf, plot_backup=False, figure_folder_backup=None,
      plot_shield=False, plot_shield_value=False, figure_folder_shield=None,
      shield_dict=None, latent_dist=None, **kwargs
  ):
    if latent_dist is not None:
      perf_latent_dist = latent_dist
      backup_latent_dist = latent_dist
    else:
      perf_latent_dist = self.performance.latent_dist
      if plot_backup:
        backup_latent_dist = self.backup.latent_dist

    for task_ind in range(self.NUM_VISUALIZE_TASK):
      if task_ind == 0:
        plot_v_task = True
      else:
        plot_v_task = plot_v
      task, task_id = venv.sample_task(return_id=True)
      perf_mode = self.performance.mode
      if perf_mode == 'safety' or perf_mode == 'risk':
        end_type = 'fail'
        plot_contour = True
      elif perf_mode == 'RA':
        end_type = 'safety_ra'
        plot_contour = True
      else:
        end_type = 'TF'
        plot_contour = False

      fig_perf = env.visualize(
          self.performance.value, self.performance, mode=self.performance.mode,
          end_type=end_type, latent_dist=perf_latent_dist, plot_v=plot_v_task,
          vmin=vmin, vmax=vmax, cmap='seismic', normalize_v=True,
          plot_contour=plot_contour, task=task, revert_task=False, **kwargs
      )
      if plot_backup:
        fig_backup = env.visualize(
            self.backup.value, self.backup, mode=self.backup.mode,
            end_type='fail', latent_dist=backup_latent_dist,
            plot_v=plot_v_task, vmin=vmin, vmax=vmax, cmap='seismic',
            normalize_v=False, plot_contour=True, task=task, revert_task=False,
            **kwargs
        )
      if plot_shield:
        assert self.has_backup, (
            "This figure requires policy with shielding scheme."
        )
        fig_shield = env.visualize(
            self.value, self.performance, mode=self.performance.mode,
            end_type='TF', latent_dist=perf_latent_dist,
            plot_v=plot_shield_value, vmin=vmin, vmax=vmax, cmap='seismic',
            normalize_v=False, plot_contour=False, task=task,
            revert_task=False, shield=True, backup=self.backup,
            shield_dict=shield_dict, **kwargs
        )
      if save_figure:
        fig_perf.savefig(
            os.path.join(
                figure_folder_perf, '{:d}_{:d}_{:d}.png'.format(
                    self.cnt_step, task_id, int(plot_v_task)
                )
            )
        )
        if plot_backup:
          fig_backup.savefig(
              os.path.join(
                  figure_folder_backup, '{:d}_{:d}_{:d}.png'.format(
                      self.cnt_step, task_id, int(plot_v_task)
                  )
              )
          )
        if plot_shield:
          fig_shield.savefig(
              os.path.join(
                  figure_folder_shield, '{:d}_{:d}_{:d}.png'.format(
                      self.cnt_step, task_id, int(plot_v_task)
                  )
              )
          )
        plt.close('all')
      if plot_figure:
        fig_perf.show()
        if plot_backup:
          fig_backup.show()
        if plot_shield:
          fig_shield.show()
        plt.pause(0.01)
        plt.close()

  @abstractmethod
  def learn(self):
    raise NotImplementedError
