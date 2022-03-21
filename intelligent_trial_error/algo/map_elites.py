
"""
Main MAP-Elites algorithm class implementation.

References:
- https://github.com/resibots/pymap_elites

"""


import pdb

import os
import pickle
import logging
import functools
import numpy as np
import multiprocessing as mpi
from collections import namedtuple

import linc.algo.mutation_operators as mo
import linc.algo.fitness_functions as ff
import linc.algo.behaviour_descriptors as bd

import torch
import pybullet as p

import time

logger = logging.getLogger(__name__)

MAX_PROCESSES = 32
Solution = namedtuple('Solution', 'bd_index ctrl_parameters fitness')


class MAPElites(object):
    """Base MAP-Elites algorithm implementation."""

    def __init__(
        self,
        collection,
        controller,
        n_iterations,
        n_samples,
        fn_mutation,
        fn_fitness,
        fn_behaviour,
        bd_dimensions,
        experiment_directory,
        **kwargs
    ):
        self.run_distributed = False # SAC is not designed to do distributed system

        for key in kwargs.keys():
            if key == "run_distributed":
                self.run_distributed = kwargs[key]
        
        self.experiment_directory = experiment_directory
        self.collection = collection
        self.controller = controller
        # Initialise MAP-Elites components
        self.n_iterations = n_iterations
        self.n_init_samples = n_samples * 10
        self.n_samples = n_samples
        self.fn_mutation = getattr(mo, 'mutate_{}'.format(fn_mutation))
        _fn_fitness = getattr(ff, 'hexapod_fit_{}'.format(fn_fitness))
        _fn_behaviour = functools.partial(
            getattr(bd, 'hexapod_grid_{}'.format(fn_behaviour)),
            bd_dimensions=bd_dimensions)
        # Set random seed
        np.random.seed(100)
        # Initialise progress tracker
        self.track_training = []
        # Set up simulator for evaluation
        if self.run_distributed:
            def proc(in_queue, out_queue):
                """Initialise the process."""
                sim_instance = Simulator(
                    controller=controller,
                    fn_behaviour=_fn_behaviour,
                    fn_fitness=_fn_fitness,
                    **kwargs)
                while True:
                    i, data = in_queue.get()
                    result = sim_instance.run_episode(data)
                    out_queue.put((i, result))
            # Initialise queues
            self.in_queue = mpi.Queue()
            self.out_queue = mpi.Queue()
            queues = (self.in_queue, self.out_queue)
            # Start processes
            for _ in range(min(MAX_PROCESSES, mpi.cpu_count() - 2)):
                process = mpi.Process(target=proc, args=queues)
                process.daemon = True
                process.start()
        else:
            # Initialise one instance
            self.sim_instance = Simulator(
                controller=controller,
                fn_behaviour=_fn_behaviour,
                fn_fitness=_fn_fitness,
                **kwargs)

    def log_progress(self, n_iter, s_added, n_sampled):
        # TODO: Track new added behaviours
        self.track_training.append(dict(
            n_iter=n_iter,
            n_sampled=n_sampled,
            collection_size=self.collection.size,
            max_fitness=self.collection.max_fitness,
            avg_fitness=self.collection.avg_fitness,
            qd_score=self.collection.qd_score))
        if n_iter % 10 == 0:
            logging.info(
                "ITER {} > Discovered behaviours {} / {} (total: {})".format(
                    n_iter, len(s_added), n_sampled, self.collection.size))
            self.save_progress()

    def save_progress(self):
        """Save progress."""
        self.collection.save_collection(self.experiment_directory)
        filename = os.path.join(self.experiment_directory, 'training_data.pkl')
        with open(filename, "wb") as f:
            pickle.dump(self.track_training, f)

    def evaluate_population(self, sample_population):
        """
        Take a list of controller parameters and evaluate them in
        the environment.
        """
        if self.run_distributed:
            # Enqueue params to execute
            for i, p in enumerate(sample_population):
                self.in_queue.put((i, p))
            # Collect the results
            results = [None] * len(sample_population)
            for _ in range(len(sample_population)):
                i, r = self.out_queue.get()
                results[i] = r
        else:
            # Execute sequentially and collect the results
            results = [self.sim_instance.run_episode(p)
                       for p in sample_population]
        return results

    def train_loop(self):
        """Main MAP-Elites training loop."""
        # Initialise first batch of parameter solutions randomly
        logging.info("Generating inital {} random sample solutions.".format(
            self.n_init_samples))
        solutions_init = self.controller.generate_parameters(
            self.n_init_samples)
        list_evaluations_init = self.evaluate_population(solutions_init)
        # Evaluate if new solutions should be added to the collection
        track_added = []
        for s_ in list_evaluations_init:
            if self.collection.add_solution(s_):
                track_added.append((s_.bd_index, s_.fitness))
        self.log_progress(
            n_iter=0, s_added=track_added, n_sampled=self.n_init_samples)

        # Run MAP-Elites iterations
        logging.info("Starting main training loop ({} iterations).".format(
            self.n_iterations))
        for i_ in range(1, self.n_iterations + 1):
            # Sample solutions, modify and evaluate them
            solutions_sampled = self.collection.sample_solutions(self.n_samples)
            solutions_mutated = self.fn_mutation(solutions_sampled)
            list_evaluations = self.evaluate_population(solutions_mutated)
            # Evaluate if new solutions should be added to the collection
            track_added = []
            for s_ in list_evaluations:
                if self.collection.add_solution(s_):
                    track_added.append((s_.bd_index, s_.fitness))
            self.log_progress(
                n_iter=i_, s_added=track_added, n_sampled=self.n_samples)

        # Show training statistics
        logging.info("Training DONE.")
        logging.info("Training stats:\
            \n\tUpdated behaviours: {} / {}\
            \n\tMaximum collection fitness: {:.2}\
            \n\tAverage collection fitness: {:.2}\
            \n\tCollection QD score: {:.2}".format(
            self.collection.size,
            np.prod(self.collection.grid_dimensions),
            self.collection.max_fitness,
            self.collection.avg_fitness,
            self.collection.qd_score))


class Simulator(object):
    """Helper class that runs an agent controller in the environment."""

    def __init__(self, environment, controller, fn_behaviour, fn_fitness, **kwargs):
        self.environment = environment
        self.controller = controller
        self.fn_behaviour = fn_behaviour
        self.fn_fitness = fn_fitness
        self.shielding = False
        self.safety_agent = None
        self.epsilon = 0.25
        self.done_signal = False
        self.state = None
        self.continuous_run = False
        self.baseline = False
        self.adversarial = False

        for key in kwargs.keys():
            if key == "shielding":
                self.shielding = kwargs[key]
            if key == "safety_agent":
                self.safety_agent = kwargs[key]
            if key == "epsilon":
                self.epsilon = kwargs[key]
            if key == "done_signal":
                self.done_signal = kwargs[key]
            if key == "continuous_run":
                self.continuous_run = kwargs[key]
            if key == "baseline":
                self.baseline = kwargs[key]
            if key == "adversarial":
                self.adversarial = kwargs[key]

    def run_episode(self, ctrl_parameters):
        """
        Run main episode loop to estimate controller performance,
        both in terms of fitness and diversity.
        """
        self.controller.set_parameters(ctrl_parameters)

        if self.continuous_run:
            if self.state is None:
                self.state = self.environment.reset()
            else:
                self.state = self.environment.soft_reset()
        else:
            self.state = self.environment.reset()

        trajectory = [self.state]
        sacra_ob_old = None
        done = False
        steps_before_done = 0

        # Run episode with given controller parameters
        for t_ in range(self.environment.n_timesteps):
            action = self.controller.get_action(self.state)

            if self.shielding:
                # if shielding is on, check if the state action pair will result in unsafe activity
                ## get state with format of SAC safety policy training
                joint_state = p.getJointStates(self.environment.botId, jointIndices = self.environment.joint_list)
                joint_position_state = np.array([state[0] for state in joint_state], dtype = np.float32)
                pos, ang = p.getBasePositionAndOrientation(self.environment.botId)
                ang = p.getEulerFromQuaternion(ang)
                vel = p.getBaseVelocity(self.environment.botId)[0][:]
                observation_state = (pos + ang + vel) # 3 + 3 + 3

                if sacra_ob_old != None:
                    state_sacra = np.concatenate((observation_state, sacra_ob_old, joint_position_state, action), axis=0)
                else:
                    state_sacra = np.concatenate((observation_state, observation_state, joint_position_state, action), axis=0)
                sacra_ob_old = observation_state

                if not self.adversarial:
                    critic_q = max(self.safety_agent.critic(torch.from_numpy(state_sacra).float().to("cpu"), torch.from_numpy(action).float().to("cpu")))
                else:
                    adversarial_action = self.safety_agent.adversarial(torch.from_numpy(np.concatenate((state_sacra, action), axis=0)).float().to("cpu"))
                    critic_q = max(self.safety_agent.critic(
                        torch.from_numpy(state_sacra).float().to("cpu"), 
                        torch.from_numpy(
                            np.concatenate((adversarial_action.detach().numpy(), action), axis = 0)).float().to("cpu")))

                if critic_q > self.epsilon:
                    # NOT GOOD, USE SHIELDING
                    # action, _ = self.safety_agent.actor.sample(torch.from_numpy(state_sacra).float().to("cpu"))
                    if not self.baseline:
                        action = self.safety_agent.actor(torch.from_numpy(state_sacra).float().to("cpu"))
                        action = action.detach().numpy()
                    else:
                        # action = np.array([
                        #     0, 0.45, 0.8,
                        #     0, 0.45, 0.8,
                        #     0, 0.45, 0.8,
                        #     0, 0.45, 0.8
                        # ])
                        # baseline action is to maintain the current stance
                        action = joint_position_state
                    
                    # override the angles value of environment, because there is a difference in action apply cycle between the performance and safety policy
                    self.environment.angles = action
                    print("\rSHIELDED        , {:.2f}".format(critic_q.detach().cpu().numpy().reshape(-1)[0]), end = "")
                else:
                    # GOOD, CONTINUE WITH THE ACTION CHOICE FROM PERFORMANCE
                    action = action
                    print("\rNOT SHIELDED    , {:.2f}".format(critic_q.detach().cpu().numpy().reshape(-1)[0]), end = "")
                
                # time.sleep(0.02)
            done, self.state = self.environment.step(action)
            if not done:
                steps_before_done += 1
            trajectory.append(self.state)
        # Calculate the behaviour descriptor and fitness based on the trajectory
        bd_index = self.fn_behaviour(trajectory=trajectory)
        fitness = self.fn_fitness(trajectory=trajectory)
        
        if self.done_signal:
            return done, steps_before_done, Solution(bd_index, ctrl_parameters, fitness)
        else:
            return Solution(bd_index, ctrl_parameters, fitness)
