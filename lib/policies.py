__doc__ = r''' This file contains the Python-implemented policies'''
import math
import numpy as np
from operator import add
from numpy import ndarray
from functools import reduce
from typing import Callable, Any
from lib.utils import (compute_B_h, compute_Theta_hat,
                       compute_psi_gap, is_non_dominated,
                       M, m, g_opt_design)

inf = (1 << 31) * 1.


class Policy(object):
    def __init__(self, bandit, features=None):
        self.bandit = bandit
        self.K = self.bandit.K
        self.D = self.bandit.D
        self.arms = self.bandit.arms
        self.sigma = bandit.sigma
        # features of size Kxh
        self.features = np.asarray(features) if features is not None else np.eye(self.K)
        self.df = self.features.shape[-1]
        self.h = self.df
        # compute the optimal set
        self.S_star = set(is_non_dominated(self.bandit.arms_means).nonzero()[0])
        # compute the gaps
        self.Deltas = np.asarray(sorted(compute_psi_gap(self.bandit.arms_means)))
        # tmp for H1 and H2
        tmp1 = 1 / self.Deltas ** 2
        tmp2 = (1 + np.arange(self.K)) * tmp1
        # compute complexities
        self.H1_lin = sum(tmp1[:self.h])
        self.H1 = sum(tmp1)
        self.H2_lin = max(tmp2[:self.h])
        self.H2 = max(tmp2)


class EGE_SH(Policy):
    def __init__(self, bandit):
        super(EGE_SH, self).__init__(bandit)

    def loop(self, seed: int, T):
        r""" Run the EGE_SH algorithm to find the optimal set"""
        np.random.seed(seed)
        K = self.K
        D = self.D
        arms = self.arms
        total = np.zeros((K, D))
        Nc = np.zeros(K, dtype=int)
        active = np.ones(K, bool)
        means = np.empty((K, D), float)
        ceil_log2_K = math.ceil(np.log2(K))
        accepts = []
        rejects = []
        for r in range(ceil_log2_K):
            num_pulls = math.floor(T / (sum(active) * ceil_log2_K))
            # forgetting
            total = np.zeros((K, D))
            Nc = np.zeros(K, dtype=int)
            if num_pulls > 0:
                for a in arms[active]:
                    total[a] += self.bandit.sample([a] * num_pulls).sum(0)
                    Nc[a] += num_pulls
                means[active] = total[active] / Nc[active, None]
            active_idx = arms[active]
            Sk_star_mask = is_non_dominated(means[active_idx])
            # Sk_star = active_idx[Sk_star_mask]
            # Sk_star_comp = active_idx[~Sk_star_mask]
            Ik = np.eye(active.sum())
            index_of = {v: k for k, v in enumerate(active_idx)}
            g_i: Callable[[Any], Any] = lambda i: max(m(means[i], means[active]) - inf * Ik[index_of[i]])
            f_i: Callable[[Any], Any] = lambda i: min(
                min(M(means[i], means[active]) + inf * Ik[index_of[i]]),
                min([max(M(means[j], means[i]), 0) + max(g_i(j), 0) for j in active_idx] + inf * Ik[
                    index_of[i]]))
            # arm to remove from the active set
            delta_i = lambda i: f_i(i) if (Sk_star_mask[index_of[i]]) else g_i(i) * (
                        1 + 1e-6)  # implement tie-breaking rule
            num_arms_to_keep = math.ceil(len(active_idx) / 2)
            sorted_arms = np.argsort([-delta_i(i) for i in active_idx])
            arms_to_dismiss = active_idx[sorted_arms[:-num_arms_to_keep]]
            for a in arms_to_dismiss:
                active[a] = False
                if Sk_star_mask[index_of[a]]:
                    accepts += [a]
                else:
                    rejects += [a]
        assert sum(active) == 1, "There should not be more than one active arm remaining"
        accepts += [*arms[active]]
        return set(accepts) == set(self.S_star)


class EGE_SR(Policy):
    def __init__(self, bandit):
        super(EGE_SR, self).__init__(bandit)

    def loop(self, seed: int, T):
        r""" Run the EGE_SR algorithm to find the optimal set"""
        np.random.seed(seed)
        K = self.K
        D = self.D
        arms = self.arms
        np.random.seed(seed)
        log_K = 1 / 2 + np.sum(1 / np.arange(2, K + 1))
        n_ks = np.ceil([0, *(1 / log_K) * (T - K) / (K + 1 - np.arange(1, K))]).astype(int)
        total = np.zeros((K, D))
        active = np.ones(K, bool)
        means = np.empty((K, D), float)
        Nc = np.zeros(K, dtype=int)
        accepts = []
        rejects = []
        for k in range(1, K):
            num_pulls = n_ks[k] - n_ks[k - 1]
            if num_pulls > 0:
                for a in arms[active]:
                    # test with
                    total[a] += self.bandit.sample([a] * num_pulls).sum(0)
                    Nc[a] += num_pulls
                means[active] = total[active] / Nc[active, None]
            active_idx = arms[active]
            # Sk_star_mask = is_non_dominated(means[active_idx])
            # Sk_star = active_idx[Sk_star_mask]
            # Sk_star_comp = active_idx[~Sk_star_mask]
            Ik = np.eye(active.sum())
            index_of = {v: k for k, v in enumerate(active_idx)}
            g_i = lambda i: max(m(means[i], means[active]) - inf * Ik[index_of[i]])
            f_i = lambda i: min(min(M(means[i], means[active]) + inf * Ik[index_of[i]]),
                                min([max(M(means[j], means[i]), 0) + max(g_i(j), 0) for j in active_idx] + inf * Ik[
                                    index_of[i]]))
            rk, dk, ak = [None] * 3
            indices = [-np.inf, -np.inf]
            dk = active_idx[np.argmax([g_i(i) for i in active_idx])]
            indices[0] = g_i(dk)
            ak = active_idx[np.argmax([f_i(i) for i in active_idx])]
            indices[1] = f_i(ak)
            if indices[0] >= indices[1]:
                # remove an arm
                rejects += [dk]
                rk = dk
            else:
                # accept an arm
                accepts += [ak]
                rk = ak
            active[rk] = False
        accepts += [*arms[active]]
        return set(accepts).issubset(set(self.S_star))


class GEGE(Policy):
    r""" Implement the GEGE algorithm"""

    def __init__(self, bandit, features):
        super(GEGE, self).__init__(bandit, features)

    def loop_fb(self, seed: int, T):
        r""" Run the two-stage algorithm to find the optimal set"""
        np.random.seed(seed)
        arms = self.arms
        K = self.K
        dv = self.D
        df = self.df
        X = self.features
        active = np.ones(K, bool)
        means = np.empty((K, dv), float)
        accepts = []
        rejects = []
        ceil_log2_df = math.ceil(np.log2(df))
        for r in range(1, ceil_log2_df + 1):
            active_idx = arms[active]
            index_of = {v: k for k, v in enumerate(active_idx)}
            pi_star, _ = g_opt_design(X[active_idx])
            # define t_r by crude ceiling or use the sophisticated rounding of Allen Zhu et al
            num_pulls = np.ceil(T * pi_star / ceil_log2_df).astype("int")
            idxs = reduce(add, [[a] * num_pulls[index_of[a]] for a in active_idx])
            X_r = X[idxs]
            Y_r = self.bandit.sample(idxs)
            Theta_r, h_r = compute_Theta_hat(X_r, Y_r, 1e-6)
            means[active_idx] = X[active_idx] @ Theta_r
            Sr_star_mask = is_non_dominated(means[active_idx])
            Sr_star = active_idx[Sr_star_mask]
            Sr_star_comp = active_idx[~Sr_star_mask]
            Ir = np.eye(active.sum())
            g_i = lambda i: max(m(means[i], means[active]) - inf * Ir[index_of[i]])
            f_i = lambda i: min(min(M(means[i], means[active]) + inf * Ir[index_of[i]]),
                                min([max(M(means[j], means[i]), 0) + max(g_i(j), 0) for j in active_idx] + inf * Ir[
                                    index_of[i]]))
            # arm to remove from the active set
            delta_i = lambda i: f_i(i) if (Sr_star_mask[index_of[i]]) else g_i(i) + 1e-5  # implement tie-breaking rule
            num_arms_to_keep = math.ceil(df / 2 ** r)
            sorted_arms = np.argsort([-delta_i(i) for i in active_idx])
            arms_to_dismiss = active_idx[sorted_arms[:-num_arms_to_keep]]
            for a in arms_to_dismiss:
                active[a] = False
                if Sr_star_mask[index_of[a]]:
                    accepts += [a]
                else:
                    rejects += [a]
            # print(accepts, rejects)
        assert sum(active) == 1, "There should not be more than one active arm remaining"
        accepts += [*arms[active]]
        return set(accepts) == set(self.S_star)  # , means#, Nc#, atot#, Nc.sum()

    def loop_fc(self, seed, delta):
        r""" Run the two-stage algorithm to find the optimal set in the
        fixed-confidence setting"""
        np.random.seed(seed)
        arms = self.arms
        K = self.K
        dv = self.D
        df = self.df
        X = self.features
        active = np.ones(K, bool)
        means = np.empty((K, dv), float)
        active = np.ones(K, bool)
        accepts = []
        r = 1
        total_pulls = 0
        while sum(active) > 1:
            # update storage lists for restarting
            means: ndarray = np.empty((K, dv), float)
            active_idx = arms[active]
            index_of = {v: k for k, v in enumerate(active_idx)}
            # update internal state
            eps_r = 1 / (2 * (2 ** r))
            n_r = sum(active)  # number of active elements
            delta_r = (6 / np.pi ** 2) * delta / (r ** 2)
            # pull active arms using design count

            # compute B and h_r
            cX_r = X[active_idx]
            B_r, h_r = compute_B_h(cX_r, 1e-9)

            # compute the G-design on dimensionality reduced-features
            pi_star, val = g_opt_design(X[active_idx] @ B_r)

            # define t_r / overly pessimistic union bound terms are discarded
            T_r = h_r * np.log(1 / delta_r) / (eps_r ** 2)  # 2sigma^2
            # define t_r by crude ceiling or use the sophisticated rounding of Allen Zhu et al
            num_pulls = np.ceil(T_r * pi_star).astype("int")  # for implementation efficiency

            idxs = reduce(add, [[a] * num_pulls[index_of[a]] for a in active_idx])
            X_r = X[idxs]
            Y_r = self.bandit.sample(idxs)

            # compute estimator of Theta
            Theta_r, _ = compute_Theta_hat(X_r, Y_r, 1e-9,
                                           B_r)
            # update means of active arms
            means[active_idx] = X[active_idx] @ Theta_r
            # elimination stage
            Sk_star_mask = is_non_dominated(means[active])
            Sk_star = active_idx[Sk_star_mask]
            Sk_star_comp = active_idx[~Sk_star_mask]
            # estimate gaps
            Ik = np.eye(active.sum())
            g_i = lambda i: max(m(means[i], means[active]) - inf * Ik[index_of[i]])
            f_i = lambda i: min(min(M(means[i], means[active]) + inf * Ik[index_of[i]]),
                                min([max(M(means[j], means[i]), 0) + max(g_i(j), 0) for j in active_idx] + inf * Ik[
                                    index_of[i]]))
            Deltas_r = np.array([max(g_i(i), f_i(i)) for i in active_idx])
            # discard arms
            mask_discard = Deltas_r >= eps_r  # discard empirically optimal arms
            mask_discard[(Deltas_r >= eps_r / 2) * (~Sk_star_mask)] = True  # discard empirically sub-optimal arms
            accepts.extend(active_idx[Sk_star_mask * mask_discard])
            active[active_idx[mask_discard]] = False
            total_pulls += np.ceil(T_r)
            r += 1
        accepts += [*arms[active]]
        return set(self.S_star) == set(accepts), total_pulls


class PAL(Policy):
    r""" Implement the PAl algorithm [Zuluaga et al 2016]"""

    def __init__(self, bandit, features):
        super(PAL, self).__init__(bandit, features)

    def loop(self, seed, delta):
        r""" Run the PAL algorithm [Zuluaga et al 2016] to find the optimal set in the
        fixed-confidence setting.
        The implementation is simplified by using insights from Auer et al 2016"""
        np.random.seed(seed)
        arms = self.arms
        K = self.K
        dv = self.D
        df = self.df
        X = self.features
        np.random.seed(seed)
        A_1 = np.arange(K)
        means = np.zeros((K, dv), float)
        t = 0
        optimal_arms = []
        X_t_Y = 0
        V_t = 0
        total_pulls = 0
        t = 1
        b_t = inf
        ci = np.ones(K)
        while True:
            I = np.eye(len(A_1))
            x_t = A_1  # [A_1[np.argmax(ci)]] #A_1 #
            samples = self.bandit.sample(x_t)
            # print(samples)
            total_pulls += len(x_t)
            V_t = V_t + X[x_t].T @ X[x_t]
            inv_V_t = np.linalg.inv(V_t + 1e-6 * np.eye(df))
            X_t_Y = X_t_Y + X[x_t].T @ samples
            Theta_t = np.linalg.lstsq(V_t, X_t_Y, rcond=-1)[0]  # np.linalg.pinv(V_r) @ X_r.T @ Y_r
            means[A_1] = X[A_1] @ Theta_t
            b_t = np.log(K * dv * t ** 2 * np.pi ** 2 / (6 * delta)) / 5
            A_1 = A_1[np.max(
                np.min(means[A_1] - means[A_1, None], -1) - self.beta_pal(A_1, inv_V_t, b_t) - self.beta_pal(A_1, inv_V_t, b_t)[:,
                                                                                          None] - inf * I, -1) < 0]
            I = np.eye(len(A_1))
            if len(A_1) == 0: break
            P1_mask = np.min(
                np.max(-means[A_1] + means[A_1, None], -1) - self.beta_pal(A_1, inv_V_t, b_t) - self.beta_pal(A_1, inv_V_t,
                                                                                                          b_t)[:,
                                                                                                 None] + inf * I,
                -1) > 0
            P_1 = A_1[P1_mask]
            A_1_notP_1 = A_1[~P1_mask]
            P_2 = [j for j in P_1 if np.min(
                M(means[A_1], means[j]) - self.beta_pal(A_1, inv_V_t, b_t) - self.beta_pal(j, inv_V_t,
                                                                                       b_t) + inf * P1_mask) > 0]
            A_1_notP_2 = [i for i in A_1 if not i in P_2]
            # A_1_notP_2 = A_1[~P2_mask]
            t += 1
            optimal_arms.extend(P_2)
            A_1 = np.array(A_1_notP_2[:])
            if len(A_1) == 0: break
            ci = self.beta_pal(A_1, inv_V_t, b_t)
        return set(self.S_star) == set(optimal_arms), total_pulls

    def beta_pal(self, idx, inv_V, b_t):
        r''' Confidence bonus for the PAL algorithm'''
        if len(np.shape(idx)) == 0: return np.sqrt(b_t * self.features[idx].T @ inv_V @ self.features[idx])/1.
        return np.sqrt(b_t * np.array([self.features[i].T @ inv_V @ self.features[i] for i in idx]))/1.


class RoundRobin(Policy):
    r""" Round Robin algorithm that equally divides the budget"""

    def __init__(self, bandit):
        super(RoundRobin, self).__init__(bandit)

    def loop(self, seed: int, T):
        r""" Run the two-stage algorithm to find the optimal set"""
        np.random.seed(seed)
        K = self.K
        D = self.D
        arms = self.arms
        np.random.seed(seed)
        num_pulls = int(np.floor(T / K))
        total = np.zeros((K, D))
        Nc = np.zeros(K, dtype=int)
        for a in arms:
            total[a] += self.bandit.sample([a] * num_pulls).sum(0)
            Nc[a] += num_pulls
        remains = T - num_pulls * K
        for _ in range(remains):
            a = np.random.randint(K)
            total[a] += self.bandit.sample(a).reshape(-1)
            Nc[a] += 1
        St_star_mask = is_non_dominated(total / Nc[:, None])
        St_star = arms[St_star_mask]
        return set(list(St_star)) == set(self.S_star)
