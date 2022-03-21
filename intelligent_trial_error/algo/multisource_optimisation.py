
"""Multi Source Optimisation algorithm implementation."""


import pdb

import numpy as np

from scipy.stats import norm
from scipy.optimize import minimize
from scipy.optimize import Bounds
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor

from linc.algo.map_elites import Solution


class MultiSourceOptimisation(object):
    """
    Multi source optimisation algorithm implementation that uses a list of
    source policy collections when finding the optimal solution for the
    transfer experiment. The learned linear combinations of the source
    collection fitness values are used as the mean prior.

    Reference:
        https://spiral.imperial.ac.uk/bitstream/10044/1/70239/2/cully_TKDE.pdf
    """

    def __init__(self, list_collections, hyperparams):
        """Initialise MultiSourceOptimisation."""
        self.list_collections = list_collections
        self.fn_acquisition_name = hyperparams['fn_acquisition']
        self.beta = hyperparams['beta']
        self.noise_var = hyperparams['noise_var']
        # Set up GPR for BO
        self.train_inputs = []
        self.train_fitness = []
        self.fn_kernel = RBF(
            length_scale_bounds="fixed",
            **hyperparams['kernel'])
        self.gp_model = GaussianProcessRegressor(
            kernel=self.fn_kernel,
            **hyperparams['gpr'])
        # Check if source collection behaviours match
        tmp_keys = [set(cc.keys()) for cc in list_collections]
        assert all([set(cc.keys()) == set.intersection(*tmp_keys)
                    for cc in list_collections])
        # Define domain and helper variables for linear combination
        self.n_sources = len(list_collections)
        self.domain = np.array(list(list_collections[0].keys()))
        self.source_fitness = {
            tuple(k): np.array([
                cc[tuple(k)].fitness for cc in list_collections])
            for k in self.domain}
        self.W_coeff = np.ones(self.n_sources) / self.n_sources
        self.domain_prior = self._update_combined_prior(domain=self.domain)
        # Set up acquisition function for BO
        self.test_fitness = self.domain_prior
        self.test_std = np.ones(len(self.domain))
        self.fn_acquisition = self.test_fitness  # only follow prior.


    def _expand_weights(self, w_coeff):
        return np.append(w_coeff, np.sqrt(1 - np.sum(np.square(w_coeff))))

    def _likelihood(self, w_coeff):
        w_coeff = self._expand_weights(w_coeff)
        n_train = len(self.train_inputs)
        A = self.train_fitness - np.array([
            np.dot(self.source_fitness[tuple(k)], w_coeff)
            for k in self.train_inputs])
        K_noise = self.fn_kernel(self.train_inputs)
        K_noise += self.noise_var * np.eye(n_train)
        likelihood = - 0.5 * np.dot(np.dot(A.T, np.linalg.inv(K_noise)), A)
        likelihood += - 0.5 * np.log(np.linalg.det(K_noise))
        likelihood += - 0.5 * n_train * np.log(2 * np.pi)
        return -likelihood

    
    def _update_linear_coefficients(self):
        if len(self.W_coeff) == 1:
            return np.ones(1)

        opt_model = minimize(self._likelihood, self.W_coeff[:-1], method='L-BFGS-B', bounds = Bounds(0,1)) # we optimise all the coeffs except the last one as its value is defined by the constraint norm(W_coeff)=1.
        return self._expand_weights(opt_model['x'])

    def _update_combined_prior(self, domain):
        prior = np.array([
            np.dot(self.source_fitness[tuple(gk)], self.W_coeff)
            for gk in domain])
        return prior

    def update_model(self, bd_index, target_fitness):
        """Add new solution to update the model."""
        self.train_inputs.append(list(bd_index))
        self.train_fitness.append(target_fitness)

        # Update linear combination coefficients
        self.W_coeff = self._update_linear_coefficients()

        # Update GPR model
        # We substract the prior because the GP models the difference between
        # the prior and the reality.
        self.gp_model.fit(
            self.train_inputs,
            self.train_fitness - self._update_combined_prior(
                domain=self.train_inputs))

        # Calculate GPR posterior over the domain, including mean prior info
        self.test_fitness, self.test_std = self.gp_model.predict(
            self.domain, return_std=True)
        self.domain_prior = self._update_combined_prior(domain=self.domain)
        self.test_fitness += self.domain_prior

        # Calculate GPR posterior for evaluated training data
        train_fitness = self.gp_model.predict(self.train_inputs) \
            + self._update_combined_prior(domain=self.train_inputs)

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
        best_source = self.list_collections[np.argmax(self.W_coeff)]

        trained_collection = {
            tuple(k): Solution(
                tuple(k), best_source[tuple(k)].ctrl_parameters, v)
            for k, v in zip(self.domain, self.test_fitness)}
        return trained_collection
