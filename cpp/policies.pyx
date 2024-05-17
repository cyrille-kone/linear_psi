# distutils: language = c++
# coding=utf-8
cimport cython
import numpy as np
from lib.bandits import Bandit
from .bandits cimport bernoulli
from .bandits cimport gaussian
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref
from lib.utils import DISTRIB

cdef class Policy(object):
    def __init__(self, py_bandit: Bandit):
        self.K = py_bandit.K
        self.D = py_bandit.D
        #self.sigma = py_bandit.sigma
        self.dim = py_bandit.D
        self.action_space = py_bandit.action_space
        #self.py_bandit = py_bandit

cdef class p_auer(Policy):
    def __init__(self, py_bandit: Bandit ):
        super().__init__(py_bandit)
    def __cinit__(self, py_bandit: Bandit ):
        #self.bandit_ref = py_bandit.bandit_ref
        if py_bandit.distrib == DISTRIB.GAUSSIAN:
            self.bandit_ref = new gaussian(py_bandit.arms_means, py_bandit.sigma)
        elif py_bandit.distrib == DISTRIB.BERNOULLI:
            self.bandit_ref = new bernoulli(py_bandit.arms_means)
        else:
            raise ValueError
        self.policy_ref = new psi_auer(deref(self.bandit_ref))
    def loop(self, size_t seed=42, double delta=0.1):
        return self.policy_ref.loop(seed, delta)

cdef class p_ape(Policy):
    def __init__(self, py_bandit:Bandit):
        super().__init__(py_bandit)
    def __cinit__(self, py_bandit:Bandit ):
        if py_bandit.distrib == DISTRIB.GAUSSIAN:
            self.bandit_ref = new gaussian(py_bandit.arms_means, py_bandit.stddev)
        elif py_bandit.distrib == DISTRIB.BERNOULLI:
            self.bandit_ref = new bernoulli(py_bandit.arms_means)
        else:
            raise ValueError
        #self.bandit_ref = py_bandit.bandit_ref
        self.policy_ref = new psi_ape(deref(self.bandit_ref))
    def loop(self, size_t seed=42, double delta=0.1):
        return self.policy_ref.loop(seed, delta)


cdef class p_unif(Policy):
    def __init__(self,  py_bandit:Bandit):
        super().__init__(py_bandit)
    def __cinit__(self, py_bandit:Bandit ):
        #self.bandit_ref = py_bandit.bandit_ref
        if py_bandit.distrib == DISTRIB.GAUSSIAN:
            self.bandit_ref = new gaussian(py_bandit.arms_means, py_bandit.stddev)
        elif py_bandit.distrib == DISTRIB.BERNOULLI:
            self.bandit_ref = new bernoulli(py_bandit.arms_means)
        else : raise ValueError
        self.policy_ref = new psi_uniform(deref(self.bandit_ref))
    def loop(self, size_t seed=42, double delta=0.1):
        return self.policy_ref.loop(seed, delta)

cdef class py_ege_sr(Policy):
    def __init__(self, py_bandit):
        super().__init__(py_bandit)
    def __cinit__(self, py_bandit):
        if py_bandit.distrib == DISTRIB.GAUSSIAN:
            self.bandit_ref = new gaussian(py_bandit.arms_means, py_bandit.stddev)
        elif py_bandit.distrib == DISTRIB.BERNOULLI:
            self.bandit_ref = new bernoulli(py_bandit.arms_means)
        else:
            raise ValueError
        #self.bandit_ref = py_bandit.bandit_ref
        self.policy_ref = new ege_sr(deref(self.bandit_ref))
    def loop(self, size_t seed, size_t T):
        return self.policy_ref.loop(seed, T)

cdef class py_ege_sh(Policy):
    def __init__(self, py_bandit):
        super().__init__(py_bandit)
    def __cinit__(self, py_bandit):
        if py_bandit.distrib == DISTRIB.GAUSSIAN:
            self.bandit_ref = new gaussian(py_bandit.arms_means, py_bandit.stddev)
        elif py_bandit.distrib == DISTRIB.BERNOULLI:
            self.bandit_ref = new bernoulli(py_bandit.arms_means)
        else:
            raise ValueError
        #self.bandit_ref = py_bandit.bandit_ref
        self.policy_ref = new ege_sh(deref(self.bandit_ref))
    def loop(self, size_t seed, size_t T):
        return self.policy_ref.loop(seed, T)

cpdef py_batch_sr(py_bandit:Bandit, vector[size_t]& Ts, vector[size_t]&seeds):
    cdef bandit* bandit_ref
    if py_bandit.distrib == DISTRIB.GAUSSIAN:
        bandit_ref = new gaussian(py_bandit.arms_means, py_bandit.stddev)
    elif py_bandit.distrib == DISTRIB.BERNOULLI:
        bandit_ref = new bernoulli(py_bandit.arms_means)
    #self.bandit_ref = py_bandit.bandit_ref
    return batch_sr(deref(bandit_ref), Ts, seeds)

cpdef py_batch_sh(py_bandit:Bandit, vector[size_t]& Ts, vector[size_t]&seeds):
    cdef bandit* bandit_ref
    if py_bandit.distrib == DISTRIB.GAUSSIAN:
        bandit_ref = new gaussian(py_bandit.arms_means, py_bandit.stddev)
    elif py_bandit.distrib == DISTRIB.BERNOULLI:
        bandit_ref = new bernoulli(py_bandit.arms_means)
    #self.bandit_ref = py_bandit.bandit_ref
    return batch_sh(deref(bandit_ref), Ts, seeds)


cpdef py_batch_ape(py_bandit:Bandit, double delta, vector[size_t]&seeds):
    cdef bandit* bandit_ref
    if py_bandit.distrib == DISTRIB.GAUSSIAN:
        bandit_ref = new gaussian(py_bandit.arms_means, py_bandit.stddev)
    elif py_bandit.distrib == DISTRIB.BERNOULLI:
        bandit_ref = new bernoulli(py_bandit.arms_means)
    #self.bandit_ref = py_bandit.bandit_ref
    return batch_ape(deref(bandit_ref), delta, seeds)