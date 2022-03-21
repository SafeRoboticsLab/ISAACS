
"""MAP-Elites collection definition"""

import pdb

import os
import pickle
import logging
import numpy as np
from operator import itemgetter


logger = logging.getLogger(__name__)


class Collection(object):
    """Collection for storing controller parameters"""

    def __init__(self, bd_dimensions, **kwargs):
        #assert len(grid_dimensions) == 2
        self.grid_dimensions = bd_dimensions
        self.grid = {}

    def add_solution(self, solution):
        """
        Add a new solution to the collection.

        Args:
            solution (namedtuple): 'bd_index', 'ctrl_parameters', 'fitness'

        """

        # pdb.set_trace()

        if solution.bd_index not in self.grid.keys():
            self.grid[solution.bd_index] = solution
            return True
        elif solution.fitness > self.grid[solution.bd_index].fitness:
            self.grid[solution.bd_index] = solution
            return True
        else:
            return False

    def sample_solutions(self, n_samples, replace=False):
        """
        Sample n solutions from the collection.

        Args:
            n_samples (int): number of solutions to sample
            replace (bool): sample with or without replacement

        """
        if self.size < n_samples and not replace:
            return list(self.grid.values())
        else:
            idx_sample = np.random.choice(self.size, n_samples, replace=replace)
            return list(itemgetter(*idx_sample)(list(self.grid.values())))

    def save_collection(self, filepath):
        """Save current collection to file"""
        filename = os.path.join(filepath, 'trained_collection.pkl')
        with open(filename, "wb") as f:
            pickle.dump((self.grid_dimensions, self.grid), f)
        logger.info("Collection saved: {}".format(filename))

    def load_collection(self, filepath):
        """Load collection from file"""
        with open(filepath, "rb") as f:
            _, self.grid = pickle.load(f)

    @property
    def size(self):
        """Return collection size"""
        return len(self.grid)

    @property
    def max_fitness(self):
        """Return fitness of best performing solution"""
        max_fit_item = max([(v.fitness, k) for k, v in self.grid.items()])
        return max_fit_item[0]

    @property
    def avg_fitness(self):
        """Return average solution fitness within the collection"""
        return np.mean([v.fitness for v in self.grid.values()])

    @property
    def qd_score(self):
        """Return average solution fitness within the collection"""
        all_fit = [v.fitness for v in self.grid.values()]
        return np.sum(all_fit) - min(all_fit) * self.size
