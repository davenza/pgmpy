#!/usr/bin/env python
#cython: boundscheck=False, wraparound=False, cdivision=True

from math import log

import numpy as np
cimport numpy as np
import scipy
from pgmpy.estimators import StructureScore

from scipy.special import loggamma

from libc.math cimport log, log1p, M_PI, exp, lgamma
from libc.string cimport memcpy
from libc.time cimport clock_t, clock

cdef extern from "time.h":
    enum: CLOCKS_PER_SEC


from ..cython_backend cimport linear_algebra
IF SIMD == True:
    from ..cython_backend cimport covariance_simd as covariance
ELSE:
    from ..cython_backend cimport covariance

class BGeScore(StructureScore):
    def __init__(self, data, iss_mu=1, iss_w=None, nu=None, **kwargs):
        """
        Class for Bayesian structure scoring for LinearGaussianBayesianNetwork with Bayesian Gaussian Equivalent score.

        Parameters
        ----------
        data: pandas DataFrame object
            datafame object where each column represents one variable.
            (If some values in the data are missing the data cells should be set to `numpy.NaN`.
            Note that pandas converts each column containing `numpy.NaN`s to dtype `float`.)

        state_names: dict (optional)
            A dict indicating, for each variable, the discrete set of states (or values)
            that the variable can take. If unspecified, the observed values in the data set
            are taken to be the only possible states.

        complete_samples_only: bool (optional, default `True`)
            Specifies how to deal with missing data, if present. If set to `True` all rows
            that contain `np.Nan` somewhere are ignored. If `False` then, for each variable,
            every row where neither the variable nor its parents are `np.NaN` is used.
            This sets the behavior of the `state_count`-method.

        References
        ---------
        [1] Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009
        Section 18.3.4-18.3.6 (esp. page 802)
        [2] AM Carvalho, Scoring functions for learning Bayesian networks,
        http://www.lx.it.pt/~asmc/pub/talks/09-TA/ta_pres.pdf
        """

        self.iss_mu = iss_mu
        if iss_w is None:
            self.iss_w = data.shape[1] + 2
        else:
            self.iss_w = iss_w

        self.nu = nu

        super(BGeScore, self).__init__(data, **kwargs)

    def local_score(self, variable, parents):
        """
        Computes a score that measures how much a given variable is "influenced" by a given list of potential parents.

        See page 23 of Bottcher's PhD thesis.
        """
        parents = list(parents)
        node_data = self.data[[variable] + parents].dropna()

        # # TODO: Implement non-uniform graph priors.
        if not parents:
            if self.nu is None:
                nu = covariance.mean(node_data[variable].values)
            else:
                nu = self.nu[variable]
            return _local_score_noparents(node_data[variable].values, self.data.shape[1], self.iss_mu, nu, self.iss_w)
        else:
            N = self.data.shape[0]
            parents = list(parents)

            if self.nu is None:
                nu = _build_nu(node_data[variable].values, node_data[parents].values)
            else:
                nu = self.nu[[variable] + parents].iloc[0].values

            return _local_score_with_parents(node_data[parents].values,
                                             node_data[variable].values,
                                             self.data.shape[1],
                                             self.iss_mu,
                                             nu,
                                             self.iss_w)

    def benchmark(self, variable, parents):
        parents = list(parents)
        node_data = self.data[[variable] + parents].dropna()



        # # TODO: Implement non-uniform graph priors.
        if not parents:
            if self.nu is None:
                nu = covariance.mean(node_data[variable].values)
            else:
                nu = self.nu[variable]
            _benchmark_noparents(node_data[variable].values, self.data.shape[1], self.iss_mu, nu, self.iss_w)
        else:
            N = self.data.shape[0]
            parents = list(parents)

            if self.nu is None:
                nu = _build_nu(node_data[variable].values, node_data[parents].values)
            else:
                nu = self.nu[[variable] + parents].iloc[0].values

            _benchmark_with_parents(node_data[parents].values,
                                             node_data[variable].values,
                                             self.data.shape[1],
                                             self.iss_mu,
                                             nu,
                                             self.iss_w)

cdef double[:] _build_nu(double[:] variable_data, double[:,:] parents_data):
    cdef Py_ssize_t N = variable_data.shape[0], m, j
    cdef Py_ssize_t k = parents_data.shape[1]

    cdef double[:] nu = np.zeros((k+1,))

    for m in range(N):
        nu[0] += variable_data[m]
        for j in range(k):
            nu[j+1] += parents_data[m,k]

    for j in range(k):
        nu[j] /= N

    return nu

cdef double _local_score_noparents(double[:] variable_data, int n_variables, double alpha_mu, double nu, double alpha_w):
    cdef Py_ssize_t N = variable_data.shape[0]
    cdef double logprob

    logprob = 0.5*(log(alpha_mu) - log(N + alpha_mu))
    logprob += lgamma(0.5*(N+alpha_w - n_variables + 1)) - lgamma(0.5*(alpha_w - n_variables + 1))
    logprob -= 0.5*N*log(M_PI)

    cdef double t = alpha_mu*(alpha_w - n_variables - 1) / (alpha_mu + 1)
    logprob += 0.5*(alpha_w - n_variables + 1)*log(t)

    cdef double mean = covariance.mean(variable_data)
    print("mean = " + str(mean))
    cdef double sse = covariance.sse(variable_data, mean)
    print("sse = " + str(sse))


    cdef double nu_diff = mean  - nu
    cdef double r = t + sse + ((N * alpha_mu) / (N + alpha_mu) * nu_diff * nu_diff)

    logprob -= 0.5 * (N + alpha_w - n_variables + 1)*log(r)
    return logprob


cdef void _benchmark_noparents(double[:] variable_data, int n_variables, double alpha_mu, double nu, double alpha_w):
    # cdef Py_ssize_t i, REP = 10000
    # cdef clock_t start_time, end_time
    #
    # start_time = clock()
    # for i in range(REP):
    #     _local_score_noparents(variable_data, n_variables, alpha_mu, nu, alpha_w)
    # end_time = clock()
    # cdef double cpu_time_used = (<double> (end_time - start_time)) / (CLOCKS_PER_SEC/1000)
    # print("Time per cycle general = " + str(cpu_time_used/REP))
    # print("Total time general = " + str(cpu_time_used))
    _local_score_noparents(variable_data, n_variables, alpha_mu, nu, alpha_w)

cdef void _benchmark_with_parents(double[:,:] parents_data, double[:] variable_data, int n_variables, double alpha_mu,
                                      double[:] nu, double alpha_w):

    cdef int i, REP = 10000
    cdef clock_t start_time, end_time

    start_time = clock()
    for i in range(REP):
        _local_score_with_parents(parents_data, variable_data, n_variables, alpha_mu, nu, alpha_w)
    end_time = clock()
    cdef double cpu_time_used = (<double> (end_time - start_time)) / (CLOCKS_PER_SEC/1000)
    print("Time per cycle general = " + str(cpu_time_used/REP))
    print("Total time general = " + str(cpu_time_used))
    #
    # _local_score_with_parents(parents_data, variable_data, n_variables, alpha_mu, nu, alpha_w)

cdef double _local_score_with_parents(double[:,:] parents_data, double[:] variable_data, int n_variables, double alpha_mu,
                                      double[:] nu, double alpha_w):
    """
    Computes a score that measures how much a given variable is "influenced" by a given list of potential parents.

    See page 23 of Bottcher's PhD thesis or "Learning Conditional Gaussian Networks", Bottcher, 2004 .

    Based on bnlearn's code (wishart.posterior.c).
    """
    cdef Py_ssize_t N = parents_data.shape[0], p = parents_data.shape[1], i, j
    cdef double logprob

    logprob = 0.5 * (log(alpha_mu) - log(N + alpha_mu))

    logprob += lgamma(0.5 * (N + alpha_w - n_variables + p + 1)) - lgamma(0.5*(alpha_w - n_variables + p + 1))
    logprob -= 0.5 * N * log(M_PI)

    cdef double t = alpha_mu * (alpha_w - n_variables - 1) / (alpha_mu + 1)

    logprob += 0.5 * (alpha_w - n_variables + p + 1) * (p + 1) * log(t) - \
               0.5 * (alpha_w - n_variables + p) * p * log(t)

    cdef double[:] parents_mean = covariance.mean_vec(parents_data)
    cdef double variable_mean = covariance.mean(variable_data)
    cdef double[:,:] r_full = covariance.sse_mat_with_vec(parents_data, variable_data, parents_mean, variable_mean)

    for i in range(p+1):
        r_full[i,i] += t

    cdef double cte_r = (N * alpha_mu) / (N + alpha_mu)
    for i in range(1,p+1):
        for j in range(i,p+1):
            r_full[i,j] += cte_r * (parents_mean[i-1] - nu[i]) * (parents_mean[j-1] - nu[j])
            r_full[j,i] = r_full[i,j]

        r_full[0,i] += cte_r * (parents_mean[i-1] - nu[i]) * (variable_mean - nu[0])
        r_full[i,0] = r_full[0,i]

    r_full[0,0] += cte_r * (variable_mean - nu[0]) * (variable_mean - nu[0])

    logprob -= 0.5 * (N + alpha_w - n_variables + p + 1)*log(linear_algebra.det(r_full))

    cdef double[:,:] r_parents = covariance.drop_variable(r_full, 0)
    logprob += 0.5 * (N + alpha_w - n_variables + p)*log(linear_algebra.det(r_parents))

    return logprob