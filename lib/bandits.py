__doc__ = r""" Implements bandit Gaussian and Bernoulli distibutions """
import abc
import numpy as np
from .utils import DISTRIB, compute_psi_gap, is_non_dominated


class Bandit(object):
    r"""Base class for bandit sampler"""

    def __init__(self, arms_means: np.ndarray):
        assert arms_means.ndim == 2, "arms_means should be 2d np array"
        self.arms_means = arms_means
        self.K = arms_means.shape[0]
        self.D = arms_means.shape[1]
        self.arms_space = np.arange(self.K)
        self.arms = self.arms_space
        self.action_space = self.arms_space
        self.distrib = DISTRIB.NONE
        self.sigma = None
        self.Deltas = np.array(compute_psi_gap(arms_means))
        self.H = np.square(1/self.Deltas).sum()
        # compute the optimal set
        self.S_star = set(is_non_dominated(arms_means).nonzero()[0])


    @abc.abstractmethod
    def sample(self, arms):
        r"""Get batch samples form arms"""
        raise NotImplementedError

    def initialize(self):
        r""" Re-initialize the bandit environment"""


class Gaussian(Bandit):
    r"""Implement a Gaussian bandit"""

    def __init__(self, arms_means: np.ndarray, stddev: np.ndarray) -> None:
        r"""
        @constructor
        Parameters
        ----------
        arms_means: np.ndarray of shape [K, d]
           Mean reward of each arm
        stddev: float or np.ndarray
        """
        super(Gaussian, self).__init__(arms_means)
        self.stddev = stddev
        self.distrib = DISTRIB.GAUSSIAN
        self.sigma = stddev

    def sample(self, arms):
        r"""
        Sample from a Gaussiant bandit
        Parameters
        -----------
        arms : set  of arms to sample
        Returns
        ------
        Samples from arms
        Test
        ----
        >>> gaussian= Gaussian(K=10)
        >>> gaussian.sample([1,2,4])
        """
        arms = [arms] if isinstance(arms, int) else np.asarray(arms, dtype=int)
        if self.D > 1:
            return np.concatenate(
                [np.random.normal(loc=self.arms_means[:, i][arms], scale=self.arms_scale[i]).reshape(-1,
                                                                                                                 1)
                 for i in range(self.D)], -1)

            # return batch_multivariate_normal(self._arms_means[arms], self._arms_scale) / np.sqrt(np.diag(
            # self._arms_scale))
        elif self.D == 1:
            return np.random.normal(loc=self.arms_means[arms], scale=self.arms_scale).reshape(-1, 1)
        raise ValueError(f"Value of D should be larger than or equal to 1 but given {self.D}")

    @property
    def arms_scale(self):
        r"""Arms scale"""
        return self.stddev


class Bernoulli(Bandit):
    r"""Implement a Bernoulli bandit"""

    def __init__(self, arms_means: np.ndarray) -> None:
        r"""
        @constructor
        Parameters
        ----------
        arms_means: np.ndarray of shape [K, d]
           Mean reward of each arm
                """
        super(Bernoulli, self).__init__(arms_means)
        self.distrib = DISTRIB.BERNOULLI
        self.sigma = 0.5

    def sample(self, arms):
        r"""
         Sample from a Bernoulli bandit
         Parameters
         -----------
         arms : set  of arms to sample
         Returns
         ------
         Samples from arms
         Test
         ----
         >>> bernoulli = Bernoulli(K=10)
         >>> bernoulli.sample([1,2,4])
         """
        arms = [arms] if isinstance(arms, int) else arms
        return np.random.binomial(1, self.arms_means[arms]).reshape(-1, self.D)
