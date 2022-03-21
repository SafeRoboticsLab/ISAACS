
"""Bayesian Optimisation algorithm implementation."""


import pdb

import numpy as np

from scipy.stats import norm
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor

from linc.algo.map_elites import Solution


class BayesianOptimisation(object):
    """
    Bayesian Optimisation algorithm implementation with three acquisition
    functions (PI, EI, UCB). Uses discrete domain over which to perform the
    optimisation. The fitness values of the source policy collection are used
    as the mean prior.

    Acquisition functions reference:
        https://modal-python.readthedocs.io/en/latest/content/query_strategies/Acquisition-functions.html
    """

    def __init__(self, grid_collection, hyperparams):
        self.grid_collection = grid_collection
        self.fn_acquisition_name = hyperparams['fn_acquisition']
        self.beta = hyperparams['beta']
        # Set up GPR for BO
        self.train_inputs = []
        self.train_fitness = []
        self.domain = np.array(list(grid_collection.keys()))
        self.domain_prior = np.array(
            [grid_collection[tuple(gk)].fitness for gk in self.domain])
        kernel = RBF(length_scale_bounds="fixed", **hyperparams['kernel'])
        self.gp_model = GaussianProcessRegressor(
            kernel=kernel, **hyperparams['gpr'])
        # Set up acquisition function for BO
        self.test_fitness = self.domain_prior
        self.test_std = np.ones(len(self.domain))
        self.fn_acquisition = self.test_fitness  # only follow prior.

    def update_model(self, bd_index, target_fitness):
        """Add new solution to update the model."""
        self.train_inputs.append(list(bd_index))
        self.train_fitness.append(target_fitness)
        # Update GPR model
        # We substract the prior because the GP models the difference between
        # the prior and the reality.
        self.gp_model.fit(
            self.train_inputs,
            self.train_fitness - np.array([
                self.grid_collection[tuple(k)].fitness
                for k in self.train_inputs]))

        # Calculate GPR posterior over the domain, including mean prior info
        self.test_fitness, self.test_std = self.gp_model.predict(
            self.domain, return_std=True)
        self.test_fitness += self.domain_prior

        # Calculate GPR posterior for evaluated training data
        train_fitness = self.gp_model.predict(self.train_inputs) + np.array(
            [self.grid_collection[tuple(k)].fitness for k in self.train_inputs])

        # Update acquisition function
        input_best = max(train_fitness)
        if self.fn_acquisition_name == 'PI':
            # Probability of improvement
            self.fn_acquisition = norm.cdf(
                (self.test_fitness - input_best) / (self.test_std + 1e-8))
        elif self.fn_acquisition_name == 'EI':
            # Expected Improvement
            scaled_pred = (self.test_fitness - input_best) / (
                self.test_std + 1e-8)
            self.fn_acquisition = sum([
                (self.test_fitness - input_best) * norm.cdf(scaled_pred),
                self.test_std * norm.pdf(scaled_pred)])
        elif self.fn_acquisition_name == 'UCB':
            # Upper Confidence Bound
            self.fn_acquisition = self.test_fitness + self.beta * self.test_std
        else:
            raise AttributeError("Acquisition function not defined!")

    def query_model(self):
        """Query the model for the next policy to evaluate."""
        bd_index = self.domain[np.argmax(self.fn_acquisition)]
        predicted_fit = self.test_fitness[np.argmax(self.fn_acquisition)]
        return (tuple(bd_index), predicted_fit)

    @property
    def avg_uncertainty(self):
        """Calculate the mean of the model's std over the domain."""
        return np.mean(self.test_std)

    @property
    def best_solution(self):
        """Return the current best solution."""
        best_idx = np.argmax(self.train_fitness)
        return tuple(self.train_inputs[best_idx]), self.train_fitness[best_idx]

    @property
    def trained_collection(self):
        """Return the collection with predicted transfer fitness values."""
        trained_collection = {
            tuple(k): Solution(
                tuple(k), self.grid_collection[tuple(k)].ctrl_parameters, v)
            for k, v in zip(self.domain, self.test_fitness)}
        return trained_collection
