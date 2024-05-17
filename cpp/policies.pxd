# distutils: language = c++
# coding=utf-8
import numpy as np
cimport numpy as np
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from lib.bandits import Bandit
from .bandits cimport bandit
from .bandits import Bandit as cBandit
from libcpp cimport bool
# Create a Cython extension type which holds a C++ instance
# as an attribute and create a bunch of forwarding methods
# Python extension type.
cdef extern from "src/policies.cxx":
    pass
# Declare the class with cdef
cdef extern from "src/policies.hpp":
    cdef cppclass policy:
        policy() except+;
        policy(bandit& bandit_ref) except+;
        size_t K;
        size_t dim;
        size_t D;
        double sigma;
        vector[size_t] action_space;
        bandit* bandit_ref;
        pair[pair[size_t, np.npy_bool], vector[size_t]] loop() nogil;

# Declare the class with cdef
cdef extern from "src/policies.hpp":
    cdef cppclass psi_auer(policy):
        size_t K;
        size_t dim;
        size_t D;
        double sigma;
        vector[size_t] action_space;
        bandit* bandit_ref;
        double delta;
        double eps;
        psi_auer() except+;
        psi_auer(bandit &) except+;
        pair[np.npy_bool, size_t] loop(const size_t&, const double&) nogil;

#Declare the class with cdef
cdef extern from "src/policies.hpp":
    cdef cppclass psi_ape(policy):
        size_t K, D;
        size_t dim;
        size_t m;
        double sigma;
        vector[size_t] action_space;
        bandit* bandit_ref;
        double delta;
        double eps_1, eps_2;
        psi_ape() except+;
        psi_ape(bandit &) except+;
        pair[np.npy_bool, size_t] loop(const size_t&, const double&) nogil;

#Declare the class with cdef
cdef extern from "src/policies.hpp":
    cdef cppclass psi_uniform(policy):
        size_t K, D, dim, m;
        vector[size_t] action_space;
        bandit* bandit_ref;
        double delta, eps_1, eps_2, sigma;
        psi_uniform() except+;
        psi_uniform(bandit &) except+;
        pair[np.npy_bool, size_t] loop(const size_t&, const double&) nogil;
# Define Python interfaces
cdef class Policy:
    cdef readonly size_t K;
    cdef readonly size_t D;
    cdef readonly size_t dim;
   # cdef readonly double sigma;
    #Bandit py_bandit;
    cdef bandit* bandit_ref
    #cdef policy* policy_ref;
    cdef readonly vector[size_t] action_space;

cdef class p_auer(Policy):
    cdef psi_auer* policy_ref
cdef class p_ape(Policy):
    cdef psi_ape* policy_ref
cdef class p_unif(Policy):
    cdef psi_uniform* policy_ref


cdef extern from "src/policies.hpp":
    cdef cppclass policy_fb:
        policy_fb() except+;
        policy_fb(bandit& bandit_ref) except+;
        size_t K;
        size_t dim;
        size_t D;
        double sigma;
        vector[size_t] action_space;
        bandit* bandit_ref;
        np.npy_bool loop() nogil;


#Declare the class with cdef
cdef extern from "src/policies.hpp":
    cdef cppclass ege_sr(policy_fb):
        size_t K, D, dim,
        vector[size_t] action_space;
        bandit * bandit_ref;
        ege_sr() except+;
        ege_sr(bandit &) except+;
        np.npy_bool loop(size_t, size_t) nogil;

#Declare the class with cdef
cdef extern from "src/policies.hpp":
    cdef cppclass ege_sh(policy_fb):
        size_t K, D, dim,
        vector[size_t] action_space;
        bandit * bandit_ref;
        ege_sh() except+;
        ege_sh(bandit &) except+;
        np.npy_bool loop(size_t, size_t) nogil;

cdef class py_ege_sr(Policy):
    cdef ege_sr* policy_ref


cdef class py_ege_sh(Policy):
    cdef ege_sh* policy_ref

cdef extern from "src/policies.hpp":
    cdef vector[vector[bool]] batch_sr(bandit& bandit_ref, vector[size_t]& Ts, vector[size_t]& seeds) nogil ;
    cdef vector[vector[bool]] batch_sh(bandit& bandit_ref, vector[size_t]& Ts, vector[size_t]& seeds) nogil;
    vector[pair[bool, size_t]] batch_ape(bandit& bandit_ref, double delta, vector[size_t]&seeds ) nogil;