# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )
#          Duy
#          Allen Z. Ren (allen.ren@princeton.edu)

import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import time

import pybullet as p
import wandb

from .sac_mini import SAC_mini
from .base_training import BaseTraining


class NaiveRL(BaseTraining):

  def __init__(
      self, CONFIG, CONFIG_UPDATE, CONFIG_ARCH, CONFIG_ENV, verbose=True
  ):
    super().__init__(CONFIG, CONFIG_ENV, CONFIG_UPDATE)

    print("= Constructing policy agent")
    self.policy = SAC_mini(CONFIG_UPDATE, CONFIG_ARCH, CONFIG_ENV)
    self.policy.build_network(verbose=verbose)

    # alias
    self.module_all = [self.policy]
    self.performance = self.policy

  @property
  def has_backup(self):
    return False

  def initBuffer(self, env, ratio=1.):
    cnt = 0
    s = env.reset()
    while len(self.memory) < self.memory.capacity * ratio:
      cnt += 1
      print('\rWarmup Buffer [{:d}]'.format(cnt), end='')
      # a = env.action_space.sample()

      a = np.array([
          np.random.uniform(-math.pi * 0.5, math.pi * 0.5),
          np.random.uniform(0, math.pi),
          np.random.uniform(0, math.pi),
          np.random.uniform(-math.pi * 0.5, math.pi * 0.5),
          np.random.uniform(0, math.pi),
          np.random.uniform(0, math.pi),
          np.random.uniform(-math.pi * 0.5, math.pi * 0.5),
          np.random.uniform(0, math.pi),
          np.random.uniform(0, math.pi),
          np.random.uniform(-math.pi * 0.5, math.pi * 0.5),
          np.random.uniform(0, math.pi),
          np.random.uniform(0, math.pi)
      ])

      # a = np.array([
      #     0.0 + np.random.uniform(-math.pi * 0.5, math.pi * 0.5),
      #     0.75 + np.random.uniform(-math.pi * 0.5, math.pi * 0.5),
      #     1.45 + np.random.uniform(-math.pi * 0.5, math.pi * 0.5),
      #     0.0 + np.random.uniform(-math.pi * 0.5, math.pi * 0.5),
      #     0.75 + np.random.uniform(-math.pi * 0.5, math.pi * 0.5),
      #     1.45 + np.random.uniform(-math.pi * 0.5, math.pi * 0.5),
      #     0.0 + np.random.uniform(-math.pi * 0.5, math.pi * 0.5),
      #     0.75 + np.random.uniform(-math.pi * 0.5, math.pi * 0.5),
      #     1.45 + np.random.uniform(-math.pi * 0.5, math.pi * 0.5),
      #     0.0 + np.random.uniform(-math.pi * 0.5, math.pi * 0.5),
      #     0.75 + np.random.uniform(-math.pi * 0.5, math.pi * 0.5),
      #     1.45 + np.random.uniform(-math.pi * 0.5, math.pi * 0.5)
      # ])

      # a = self.genRandomActions(1)[0]

      s_, r, done, info = env.step(a, safety_action=True)
      s_ = None if done else s_
      self.store_transition(s.unsqueeze(0), a, r, s_.unsqueeze(0), info)
      if done:
        s = env.reset()
      else:
        s = s_
    print(" --- Warmup Buffer Ends")

  def learn(self, env, current_step=None):
    # TODO: visualization
    # TODO: cast to torch.tensor in env.step()
    # TODO: implement env.simulate_trajectories
    # TODO: keep a counter for how many steps it run -> raise done flag if
    # timeout or failed
    # TODO: vectorized env (KC)

    # hyper-parameters
    max_steps = self.CONFIG.MAX_STEPS
    opt_freq = self.CONFIG.OPTIMIZE_FREQ
    num_update_per_opt = self.CONFIG.UPDATE_PER_OPT
    check_opt_freq = self.CONFIG.CHECK_OPT_FREQ
    min_step_b4_opt = self.CONFIG.MIN_STEPS_B4_OPT
    out_folder = self.CONFIG.OUT_FOLDER

    # main training
    start_learning = time.time()
    train_records = []
    train_progress = []
    violation_record = []
    episode_record = []
    cnt_opt = 0
    cnt_opt_period = 0
    cnt_safety_violation = 0
    cnt_num_episode = 0

    # saving model
    model_folder = os.path.join(out_folder, 'model')
    os.makedirs(model_folder, exist_ok=True)
    self.module_folder_all = [model_folder]
    save_metric = self.CONFIG.SAVE_METRIC

    if current_step is None:
      self.cnt_step = 0
    else:
      self.cnt_step = current_step
      print("starting from {:d} steps".format(self.cnt_step))

    s = env.reset()

    while self.cnt_step <= max_steps:
      #! no for-loop, to progress to vectorized environments
      # Select action, assume no append for now
      with torch.no_grad():
        if self.cnt_step < min_step_b4_opt:
          a = torch.Tensor(np.array([
              np.random.uniform(-math.pi * 0.5, math.pi * 0.5),
              np.random.uniform(0, math.pi),
              np.random.uniform(0, math.pi),
              np.random.uniform(-math.pi * 0.5, math.pi * 0.5),
              np.random.uniform(0, math.pi),
              np.random.uniform(0, math.pi),
              np.random.uniform(-math.pi * 0.5, math.pi * 0.5),
              np.random.uniform(0, math.pi),
              np.random.uniform(0, math.pi),
              np.random.uniform(-math.pi * 0.5, math.pi * 0.5),
              np.random.uniform(0, math.pi),
              np.random.uniform(0, math.pi)
          ])).squeeze(0)
        else:
          a, _ = self.policy.actor.sample(
            s.float().to(self.device), append=None,
            latent=None
          )

      # Interact with env
      s_, r, done, info = env.step(a, safety_action=True)
      self.store_transition(
          s.unsqueeze(0), a.unsqueeze(0), r, s_.unsqueeze(0), done, info
      )
      if done:
        s_ = env.reset()
        g_x = info['g_x']
        if g_x > 0:
          cnt_safety_violation += 1
        cnt_num_episode += 1
      violation_record.append(cnt_safety_violation)
      episode_record.append(cnt_num_episode)

      s = s_

      # Optimize
      if (self.cnt_step > min_step_b4_opt and cnt_opt_period >= opt_freq):
        cnt_opt_period = 0

        # Update critic/actor
        loss = np.zeros(4)
        for timer in range(num_update_per_opt):
          batch = self.unpack_batch(self.sample_batch())

          loss_tp = self.policy.update(
              batch, timer, update_period=self.UPDATE_PERIOD
          )
          for i, l in enumerate(loss_tp):
            loss[i] += l
        loss /= num_update_per_opt

        # Record: loss_q, loss_pi, loss_entropy, loss_alpha
        train_records.append(loss)
        if self.CONFIG.USE_WANDB:
          wandb.log({
              "loss_q": loss[0],
              "loss_pi": loss[1],
              "loss_entropy": loss[2],
              "loss_alpha": loss[3],
          }, step=self.cnt_step, commit=False)

        # Count number of optimization
        cnt_opt += 1
        
        if self.CONFIG.USE_WANDB:
          lr_c = self.policy.critic_optimizer.state_dict()['param_groups'][0]['lr']
          lr_a = self.policy.actor_optimizer.state_dict()['param_groups'][0]['lr']
          wandb.log({
              "cnt_safety_violation": cnt_safety_violation,
              "alpha": self.policy.alpha,
              "gamma": self.policy.GAMMA,
              "lr_critic": lr_c,
              "lr_actor": lr_a
          }, step=self.cnt_step, commit=False)

        # Check after fixed number of gradient updates
        if cnt_opt % check_opt_freq == 0:
          print()
          print(
              '  - Safety violations so far: {:d}'.
              format(cnt_safety_violation)
          )

          progress = self.policy.check(env, self.cnt_step, states=None)  #TODO
          success_rate = progress[0]
          safe_rate = progress[0] + progress[2]

          if self.CONFIG.USE_WANDB:
            wandb.log({
                "success_rate": success_rate,
                "safe_rate": safe_rate,
            }, step=self.cnt_step, commit=True)

          # Saving model
          if save_metric == 'success':
            self.save(metric=success_rate)
          elif save_metric == 'safety':
            self.save(metric=safe_rate)
          else:
            raise NotImplementedError

          # Save training details
          torch.save({
              'train_records': train_records,
              'train_progress': train_progress,
              'violation_record': violation_record,
          }, os.path.join(out_folder, 'train_details'))

          # TODO: add visualization below, such as get_figures()

      # Count
      self.cnt_step += 1
      cnt_opt_period += 1

      # Update gamma, lr etc.
      for _ in range(self.n_envs):
        self.policy.update_hyper_param()

    self.save(force_save=True)
    end_learning = time.time()
    time_learning = end_learning - start_learning
    print('\nLearning: {:.1f}'.format(time_learning))

    train_records = np.array(train_records)
    train_progress = np.array(train_progress)
    violation_record = np.array(violation_record)
    episode_record = np.array(episode_record)
    return train_records, train_progress, violation_record, episode_record

  #! cleanup
  def evaluate(self, env, epsilon=0.25):
    s = env.reset()
    # for i in range(1000):
    while True:
      # a, _ = self.actor.sample(torch.from_numpy(s).float().to(self.device))
      a = self.policy.actor(s.float().to(self.device))
      # a = torch.from_numpy(np.array([0] * 18)).float().to(self.device)

      # run for a_{t+1} and s_{t}
      critic_q = max(
          self.policy.critic(s.float().to(self.device), a.float().to(self.device))
      )

      # print(env.robot.get_observation()[2])

      # if larger than threshold, then not save nor live
      print("\r{:.3f}".format(float(critic_q.detach().numpy())), end="")

      # if critic_q > epsilon:
      #     print("\rNOT GOOD      ", end = "")
      # else:
      #     print("\rGOOD          ", end = "")

      a = a.detach().numpy()
      # print("State: {}, action: {}".format(s.detach().numpy()[:5], a))
      s_, r, done, info = env.step(a, safety_action=True)
      s = s_
      # time.sleep(0.02)
      if done:
        if p.getKeyboardEvents().get(49):
          continue
        else:
          env.reset()

  def shielding_with_IK(self, env, controller, epsilon=0.25, override=True):
    dt = 1. / 240.
    # state from SACRA
    state = env.reset()
    for i in range(1000):
      action = controller.get_action()

      # check if action and state are safe
      critic_q = max(
          self.policy.critic(
              state.float().to(self.device),
              torch.from_numpy(action).float().to(self.device)
          )
      )

      if critic_q > epsilon:
        # NOT GOOD, USE SHIELDING
        if override:
          # action, _ = self.actor.sample(torch.from_numpy(state_sacra).float().to(self.device))
          action = self.policy.actor(state.float().to(self.device))
          action = action.detach().numpy()
        print(
            "\rStep: {}\tQ: {:.3f}\tSHIELDED!           ".format(str(i).zfill(3), float(critic_q.detach().numpy())
            ), end=""
        )
        state, r, done, info = env.step(action, safety_action=True)
      else:
        # GOOD, CONTINUE WITH THE ACTION CHOICE FROM PERFORMANCE
        action = action
        print(
            "\rStep: {}\tQ: {:.3f}\t                    ".format(str(i).zfill(3), float(critic_q.detach().numpy())
            ), end=""
        )
        state, r, done, info = env.step(action, safety_action=False)
      
      print(action[:4])
      # Check if roll pitch are not too high
      error = False
      euler = p.getEulerFromQuaternion(env.robot.linc_get_pos()[1])
      if ((abs(euler[1]) >= math.pi / 2) or (abs(euler[0]) >= math.pi / 2)):
        error = True

      # if done or sum(env.robot.linc_get_ground_contacts()) == 0:
      # if done or error:
      if error:  # only care if the robot flips, do not care about the safety margin of safety policy
        return 0
    return 1

  #! cleanup
  def shielding(
      self, env, controller, ctrl_parameters, epsilon=0.25, override=True
  ):
    dt = 1. / 240.
    controller.set_parameters(ctrl_parameters)
    # state from SACRA
    state_sacra = env.reset()
    for i in range(1000):
      state_linc = env.robot.linc_get_state(i * dt)

      # SACRA and LINC use similar action
      action = controller.get_action(state_linc)

      # check if action and state are safe
      critic_q = max(
          self.policy.critic(
              state_sacra.float().to(self.device),
              torch.from_numpy(action).float().to(self.device)
          )
      )

      if critic_q > epsilon:
        # NOT GOOD, USE SHIELDING
        if override:
          # action, _ = self.actor.sample(torch.from_numpy(state_sacra).float().to(self.device))
          action = self.policy.actor(state_sacra.float().to(self.device))
          action = action.detach().numpy()
        print(
            "\rStep: {}\tSHIELDED!           ".format(
                str(i).zfill(3),
            ), end=""
        )
        state_sacra, r, done, info = env.step(action, safety_action=True)
      else:
        # GOOD, CONTINUE WITH THE ACTION CHOICE FROM PERFORMANCE
        action = action
        print("\r                    ", end="")
        state_sacra, r, done, info = env.step(action, safety_action=False)

      # Check if roll pitch are not too high
      error = False
      euler = p.getEulerFromQuaternion(env.robot.linc_get_pos()[1])
      if ((abs(euler[1]) >= math.pi / 2) or (abs(euler[0]) >= math.pi / 2)):
        error = True

      # if done or sum(env.robot.linc_get_ground_contacts()) == 0:
      # if done or error:
      if error:  # only care if the robot flips, do not care about the safety margin of safety policy
        return 0
    return 1

  #! cleanup
  def shielding_rollout(
      self, env, controller, ctrl_parameters, override=True, rollout_steps=50
  ):
    dt = 1. / 240.
    controller.set_parameters(ctrl_parameters)
    # state from SACRA
    state_sacra = env.reset()
    for i in range(1000):
      state_linc = env.robot.linc_get_state(i * dt)

      # SACRA and LINC use similar action
      action = controller.get_action(state_linc)

      # start rollout for rollout_steps times
      rollout_failure = True
      env.rollout_reset()
      # take the current action, receive the next state
      rollout_state, r, done, info = env.rollout_step(action)

      for step in range(rollout_steps):
        rollout_action = self.actor(
            torch.from_numpy(rollout_state).float().to(self.device)
        )
        rollout_action = rollout_action.detach().numpy()
        rollout_state, r, done, info = env.rollout_step(rollout_action)
        # print(step)
        if info["g_x"] > 0:
          # print("FAILED")
          rollout_failure = True
          break
        elif info['l_x'] <= 0 and info['g_x'] <= 0:
          # print("SUCCESS AND SAFE")
          rollout_failure = False
      # p.disconnect(env.rollout_client)

      if rollout_failure:
        # NOT GOOD, USE SHIELDING
        if override:
          # action, _ = self.actor.sample(torch.from_numpy(state_sacra).float().to(self.device))
          action = self.actor(
              torch.from_numpy(state_sacra).float().to(self.device)
          )
          action = action.detach().numpy()
        print(
            "\rStep: {}\tRollout: {}\tSHIELDED!           ".format(
                str(i).zfill(3),
                str(step).zfill(3)
            ), end=""
        )
        state_sacra, r, done, info = env.step(action, safety_action=True)
      else:
        # GOOD, CONTINUE WITH THE ACTION CHOICE FROM PERFORMANCE
        action = action
        print(
            "\rStep: {}\tRollout: {}\t                    ".format(
                str(i).zfill(3),
                str(step).zfill(3)
            ), end=""
        )
        state_sacra, r, done, info = env.step(action, safety_action=False)

      # Check if roll pitch are not too high
      error = False
      euler = p.getEulerFromQuaternion(env.robot.linc_get_pos()[1])
      if ((abs(euler[1]) >= math.pi / 2) or (abs(euler[0]) >= math.pi / 2)):
        error = True

      # if done or sum(env.robot.linc_get_ground_contacts()) == 0:
      # if done or error:
      if error:  # only care if the robot flips, do not care about the safety margin of safety policy
        return 0
    return 1

  #! cleanup
  def shielding_duo(self, env, controller, ctrl_parameters, epsilon=0.25):
    dt = 1. / 240.
    controller.set_parameters(ctrl_parameters)
    # state from SACRA
    state_sacra = env.reset()
    for i in range(1000):
      state_linc = {
          "safety": env.robots["safety"].linc_get_state(i * dt),
          "performance": env.robots["performance"].linc_get_state(i * dt)
      }

      # SACRA and LINC use similar action
      action = {
          "safety": controller.get_action(state_linc["safety"]),
          "performance": controller.get_action(state_linc["performance"])
      }

      # check if action and state are safe
      critic_q = max(
          self.critic(
              torch.from_numpy(state_sacra["safety"]).float().to(self.device),
              torch.from_numpy(action["safety"]).float().to(self.device)
          )
      )

      if critic_q > epsilon:
        # NOT GOOD, USE SHIELDING
        # action["safety"], _ = self.actor.sample(torch.from_numpy(state_sacra["safety"]).float().to(self.device))
        action["safety"] = self.actor(
            torch.from_numpy(state_sacra["safety"]).float().to(self.device)
        )
        action["safety"] = action["safety"].detach().numpy()

      state_sacra, r, done, info = env.step(action)

  # region: utils
  def restore(self, step, logs_path):
    """

    Args:
        step (int): #updates trained.
        logs_path (str): the path of the directory, under this folder there
            should be critic/ and agent/ folders.
    """
    model_folder = path_c = os.path.join(logs_path)
    path_c = os.path.join(model_folder, 'critic', 'critic-{}.pth'.format(step))
    path_a = os.path.join(model_folder, 'actor', 'actor-{}.pth'.format(step))

    self.policy.critic.load_state_dict(
        torch.load(path_c, map_location=self.device)
    )
    self.policy.critic.to(self.device)
    self.policy.critic_target.load_state_dict(
        torch.load(path_c, map_location=self.device)
    )
    self.policy.critic_target.to(self.device)
    self.policy.actor.load_state_dict(
        torch.load(path_a, map_location=self.device)
    )
    self.policy.actor.to(self.device)
    print(
        '  <= Restore policy with {} updates from {}.'.format(
            step, model_folder
        )
    )

  # endregion
