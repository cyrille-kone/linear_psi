__doc__ = r''' File for classes and utility functions'''

from enum import Enum


class DISTRIB(Enum):
    GAUSSIAN = 1
    BERNOULLI = 2
    NONE = -1


import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint

inf = (1 << 31) * 1.


def is_non_dominated(Y: np.ndarray, eps=0.) -> np.ndarray:
    r"""Computes the non-dominated front.

    Note: this assumes maximization.

    For small `n`, this method uses a highly parallel methodology
    that compares all pairs of points in Y. However, this is memory
    intensive and slow for large `n`. For large `n` (or if Y is larger
    than 5MB), this method will dispatch to a loop-based approach
    that is faster and has a lower memory footprint.

    Args:
        Y: A `(batch_shape) x n x m`-dim tensor of outcomes.
        deduplicate: A boolean indicating whether to only return
            unique points on the pareto frontier.

    Returns:
        A `(batch_shape) x n`-dim boolean tensor indicating whether
        each point is non-dominated.
    """
    # n = Y.shape[-2]
    Y1 = np.expand_dims(Y, -3)
    Y2 = np.expand_dims(Y, -2)
    # eps from context
    dominates = (Y1 >= Y2 + eps).all(axis=-1) & (Y1 > Y2 + eps).any(axis=-1)
    nd_mask = ~(dominates.any(axis=-1))
    return nd_mask


# @title  Set up
def batch_multivariate_normal(batch_mean, batch_cov) -> np.ndarray:
    r"""Batch samples from a multivariate normal
    Parameters
    ----------
    batch_mean: np.ndarray of shape [N, d]
                Batch of multivariate normal means
    batch_cov: np.ndarray of shape [N, d, d]
                Batch of multivariate normal covariances
    Returns
    -------
    Samples from N(batch_mean, batch_cov)"""
    batch_size = np.shape(batch_mean)[0]
    samples = np.arange(batch_size).astype(np.float32).reshape(-1, 1)
    return np.apply_along_axis(
        lambda i: np.random.multivariate_normal(mean=batch_mean[int(i[0])], cov=batch_cov),
        axis=1,
        arr=samples)


def M(xi, xj):
    return np.max(xi - xj, -1)


def m(xi, xj):
    return np.min(xj - xi, -1)


def delta_i_plus(i, S_star, means):
    return min([min(M(means[i], means[j]), M(means[j], means[i])) + inf * (j == i) for j in S_star])


def delta_i_minus(i, S_star_comp, means):
    if len(S_star_comp) == 0: return inf
    return min([max(M(means[j], means[i]), 0) + max(Delta_i_star(j, means), 0) for j in S_star_comp])


def delta_i_minus_prime(i, S_star_comp, means):
    if len(S_star_comp) == 0: return inf
    return min(
        [m(means[j], means[i]) for j in S_star_comp if (m(means[j], means[i]) == Delta_i_star(j, means))] + [inf])


def Delta_i_star(i, means):
    return np.max(m(means[i], means))


def Delta_i(i, S_star, S_star_comp, means):
    if i in S_star: return min(delta_i_plus(i, S_star, means), delta_i_minus(i, S_star_comp, means))
    return Delta_i_star(i, means)


def Delta_i_prime(i, S_star, S_star_comp, means):
    if i in S_star: return min(delta_i_plus(i, S_star, means), delta_i_minus_prime(i, S_star_comp, means))
    return Delta_i_star(i, means)


def compute_psi_gap(arms_means: np.ndarray) -> np.ndarray[float]:
    r"""
    Compute the PSI gaps for the input means vector  
    @reference to PSI gaps in Auer et al. 2016 Pareto front identification with stochastic feedback"""
    K = arms_means.shape[0]
    S_star_mask = is_non_dominated(arms_means)
    S_star = set(S_star_mask.nonzero()[0])
    S_star_comp = set((~S_star_mask).nonzero()[0])
    return np.array([Delta_i(i, S_star, S_star_comp, arms_means) for i in range(K)])


def beta_ij(T_i, T_j, delta):
    r""" Confidence bonuses on pairs"""
    # return beta(T_i, delta) + beta(T_j, delta)
    return 0.5 * np.sqrt(
        2 * (
                2 * (np.log(1 / delta) / 2 + np.log(np.log(1 / delta) / 2)) + 2 * np.log(4 + np.log(T_i)) + 2 * np.log(
            4 + np.log(T_j))
        ) * (
                (1 / T_i) + (1 / T_j)
        )
    )


def beta(T_i, delta):
    return 0.5 * np.sqrt((2 * np.log(1 / delta) + 1 * np.log(np.log(1 / delta)) + 1 * np.log(np.log(np.e * T_i))) / T_i)


def g_opt_design(designs, tol=1e-9):
    r"""
    Compute a G-optimal design by using the equivalence theorem of Kieffer and Wolfovitz
    """
    n = designs.shape[0]
    lc = LinearConstraint(np.ones(n), lb=1, ub=1)

    def func(x):
        return -np.linalg.det(designs.T @ (designs * x[:, None]))

    ret = minimize(func, x0=np.ones(n) / n, method="SLSQP", bounds=[(0, 1)] * n, constraints=(lc), tol=tol)
    pi_star = ret.x
    inv_V_pi_star = np.linalg.pinv(designs.T @ (designs * pi_star[:, None]))
    vals = [designs[i].T @ inv_V_pi_star @ designs[i] for i in range(n)]
    return pi_star, max(vals)


# @title Utils for the algorithms to run in Parallel
def run_batch_seeds(seeds, T, callback):
    r"""
    Runs callback with a budget T with the seeds in [seeds]
    :param T: Budget
    :param seeds: list of seeds
    :param callback: algorithm to run [ege_sr | ege_sh| round_robin]
    :return: The average return over thr seeds in [seeds]
    """
    return Parallel(n_jobs=-1, verbose=0)(delayed(callback)(seed, T) for seed in seeds)


def compute_Theta_hat(X, Y, zero_val=1e-9, B=None):
    r""" Compute the estimator Theta_hat
    return
    ------
    The estimated regression parameter and the rank of X.T
    """
    if B is None:
        U, s, _ = np.linalg.svd(X.T)
        # computer tolerance
        # cf https://numpy.org /doc/stable/reference/generated/numpy.linalg.matrix_rank.html

        zero_val = max((s.max() * max(np.shape(U)) * np.finfo(U.dtype).eps) * 100, 1e-9)
        h = sum(s > zero_val)  # rank of the matrix
        # compute lstq on the projected matrix
        B = U[:, :h]
        # print(zero_val)
    h = B.shape[-1]
    # both are equivalent return np.linalg.lstsq(X.T @ X, X.T @ Y, rcond =-1)[0]
    return B @ np.linalg.inv(B.T @ (X.T @ X) @ B) @ B.T @ X.T @ Y, h


def compute_B_h(X, zero_val=1e-9):
    r""" Compute the matrix B cf section 2.2 """
    U, s, _ = np.linalg.svd(X.T)
    # computer tolerance
    # cf https://numpy.org /doc/stable/reference/generated/numpy.linalg.matrix_rank.html
    zero_val = zero_val = max((s.max() * max(np.shape(U)) * np.finfo(U.dtype).eps) * 100, 1e-9)
    h = sum(s > zero_val)  # rank of the matrix
    return U[:, :h], h
