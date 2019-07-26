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

class BGeScore(StructureScore):
    def __init__(self, data, iss=10, rho=None, phi="heckerman", **kwargs):
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

        self.iss = iss
        self.rho = rho

        N = data.shape[0]
        if phi == "bottcher":
            self.phi_coef = (N-1) / N * (iss-1)
        self.phi_coef = (N-1) / N * iss / (iss + 1) * (iss-2)

        super(BGeScore, self).__init__(data, **kwargs)

    def _build_mu(self, node, parents):
        mu = np.zeros((len(parents) + 1))
        mu[0] = self.data[node].mean()
        return mu

    def _build_tau(self, parents):
        """
        This implementation is copied from blnearn's wisharh.posterior.c build_tau() function.
        :param parents:
        :param phi:
        :return:
        """
        tau = np.empty((len(parents) + 1, len(parents) + 1))

        data_parents = self.data[parents]

        inv_cov = np.linalg.inv(data_parents.cov() * self.phi_coef)
        means = data_parents.mean()

        tau[1:, 1:] = inv_cov
        tau[0, 1:] = tau[1:, 0] = -np.dot(means, inv_cov)
        tau[0, 0] = 1 / self.iss - np.inner(means, tau[0, 1:])

        return np.linalg.inv(tau), tau

    def local_score(self, variable, parents):
        """
        Computes a score that measures how much a given variable is "influenced" by a given list of potential parents.

        See page 23 of Bottcher's PhD thesis.
        """

        node_data = self.data[[variable] + parents].dropna()
        if not parents:
            return _local_score_noparents(node_data[variable].values, self.iss, self.phi_coef)
        else:
            N = self.data.shape[0]
            parents = list(parents)
            return _local_score_with_parents(node_data[parents].values,
                                             node_data[variable].values,
                                             self.iss, self.phi_coef)


cdef double _local_score_noparents(double[:] variable_data, double iss, double phi_coef):
    cdef Py_ssize_t N = variable_data.shape[0]
    cdef Py_ssize_t i

    cdef double mu = 0

    for i in range(N):
        mu += variable_data[i]
    mu = mu / N

    cdef double phi = 0
    cdef double tmp
    for i in range(N):
        tmp = variable_data[i] - mu
        phi += tmp*tmp
    phi = phi / (N-1) *phi_coef

    cdef double tau = iss
    cdef double rho = iss

    cdef double logscale, logk, res=0
    cdef double oldtau, oldmu

    for i in range(N):
        logscale = log(phi) + log1p(1.0 / tau)
        logk = lgamma(0.5* (1.0 + rho)) - lgamma(rho*0.5)
        logk -= 0.5 * (logscale + log(M_PI))
        res += logk - 0.5 * (1.0 + rho) * log1p( (variable_data[i] - mu) * (variable_data[i] - mu) / exp(logscale) )

        oldtau = tau
        oldmu = mu

        tau += 1
        rho += 1

        mu = (oldtau * mu + variable_data[i]) / tau

        phi += (variable_data[i] - mu)*variable_data[i] + (oldmu - mu)*oldtau*oldmu

    return res

cdef double[:] get_mean_data(double[:,:] linregress_data):
    cdef Py_ssize_t N = linregress_data.shape[0]
    cdef Py_ssize_t k = linregress_data.shape[1]

    cdef Py_ssize_t i, j

    cdef double[:] means = np.zeros((k))
    for i in range(N):
        for j in range(k):
            means[j] += linregress_data[i,j]

    for j in range(k):
        means[j] /= N
    return means


cdef double[:,:] get_covariance_data(double[:,:] linregress_data, double[:] means):
    cdef Py_ssize_t N = linregress_data.shape[0]
    cdef Py_ssize_t k = linregress_data.shape[1]
    cdef Py_ssize_t i, j, m

    cdef double tmp
    cdef double[:,:] cov = np.empty((k,k))
    for i in range(k):
        for j in range(k):
            for m in range(N):
                cov[i,j] += (linregress_data[m,i] - means[i]) * (linregress_data[m, j] - means[j])

            cov[i,j] /= N-1
    return cov


cdef void _build_tau(double[:,:] linregress_data, double phi_coef, double iss, double[:,:] tau, double[:,:] inv_tau):
    """
    This implementation is copied from blnearn's wisharh.posterior.c build_tau() function.
    :param parents:
    :param phi:
    :return:
    """
    cdef Py_ssize_t ncol = linregress_data.shape[1], tau_ncol = ncol + 1
    cdef double[:] means = get_mean_data(linregress_data)
    cdef Py_ssize_t i, j

    cdef double[:,:] inv_cov = get_covariance_data(linregress_data, means)
    for i in range(ncol):
        for j in range(ncol):
            inv_cov[i,j] *= phi_coef

    linear_algebra.inverse_symmetric_psd(inv_cov, inv_cov)

    inv_tau[1:,1:] = inv_cov

    cdef double tmp
    for i in range(1, tau_ncol):
        tmp = 0
        for j in range(ncol):
            tmp += means[j] * inv_cov[j, i-1]

        inv_tau[i,0] = inv_tau[0,i] = -tmp

    inv_tau[0,0] = 0
    for i in range(1, tau_ncol):
        inv_tau[0,0] += -means[i-1] * inv_tau[i,0]
    inv_tau[0,0] += 1.0 / iss

    linear_algebra.inverse_symmetric_psd(inv_tau, tau)

# TODO: Should we use dgemm as bnlearn?
cdef double mahalanobis(double[:] vec1, double[:,:] sigma, double[:] vec2):
    cdef Py_ssize_t K = vec1.shape[0]
    cdef double res = 0
    cdef Py_ssize_t i, j
    for i in range(K):
        for j in range(K):
            res += vec1[i] * sigma[i,j] * vec2[j]
    return res

cdef void _benchmark(double[:,:] linregress_data, double[:] variable_data, int iss, double phi_coef):

    cdef int i, REP = 10000
    cdef clock_t start_inv, end_time_inv
    cdef double score
    score = _local_score_with_parents(linregress_data, variable_data, iss, phi_coef)
    print("Final score = " + str(score))
    #
    start_inv = clock()

    for i in range(REP):
        score = _local_score_with_parents(linregress_data, variable_data, iss, phi_coef)

    end_time_inv = clock()
    cpu_time_used = (<double> (end_time_inv - start_inv)) / (CLOCKS_PER_SEC/1000)
    print("Time per cycle general = " + str(cpu_time_used/REP))
    print("Total time general = " + str(cpu_time_used))


cdef double _local_score_with_parents(double[:,:] linregress_data, double[:] variable_data, int iss, double phi_coef):
    """
    Computes a score that measures how much a given variable is "influenced" by a given list of potential parents.

    See page 23 of Bottcher's PhD thesis.
    """
    cdef Py_ssize_t N = linregress_data.shape[0], ncol = linregress_data.shape[1], tau_ncol = ncol + 1

    cdef double[:] mu = np.zeros((tau_ncol))
    cdef double[:] old_mu = np.empty((tau_ncol))
    cdef double[:] delta_mu = np.empty((tau_ncol))
    cdef Py_ssize_t i, j, k

    for i in range(N):
        mu[0] += variable_data[i]
    mu[0] /= N

    cdef double var_x = 0, tmp
    for i in range(N):
        tmp = variable_data[i] - mu[0]
        var_x += tmp*tmp
    var_x /= N-1

    cdef double phi = var_x * phi_coef

    cdef double[:,:] tau = np.empty((tau_ncol, tau_ncol))
    cdef double[:,:] old_tau = np.empty((tau_ncol, tau_ncol))
    cdef double[:,:] inv_tau = np.empty((tau_ncol, tau_ncol))
    _build_tau(linregress_data, phi_coef, iss, tau, inv_tau)

    cdef double[:] zi = np.empty((tau_ncol))
    zi[0] = 1

    cdef double xprod, logscale, logk, zi_mu, score=0
    cdef double[:] tau_mu = np.empty((tau_ncol))

    cdef int rho = iss + ncol


    cdef double[:] dgemv_workspace = np.empty((tau_ncol))
    dgemv_workspace[0] = 1

    cdef double delta_phi = 0
    cdef Py_ssize_t instances_to_debug = N
    for i in range(instances_to_debug):

        for j in range(1, tau_ncol):
            zi[j] = linregress_data[i,j-1]


        xprod = mahalanobis(zi, inv_tau, zi)

        logscale = log(phi) + log1p(xprod)
        logk = lgamma(0.5*(1.0 + rho)) - lgamma(0.5*rho) - 0.5*(logscale + log(M_PI))

        zi_mu = 0
        for j in range(tau_ncol):
            zi_mu += zi[j] * mu[j]

        tmp = variable_data[i] - zi_mu
        score += logk - 0.5*(1.0+rho)*log1p(tmp*tmp / exp(logscale))

        # Update tau
        memcpy(&old_tau[0,0], &tau[0,0], tau_ncol*tau_ncol*sizeof(double))
        for j in range(tau_ncol):
            for k in range(j+1, tau_ncol):
                tau[k,j] = tau[j,k] = tau[j,k] + zi[j] * zi[k]
        for j in range(tau_ncol):
            tau[j,j] += zi[j] * zi[j]

        linear_algebra.inverse_symmetric_psd(tau, inv_tau)

        # Update mu
        memcpy(&old_mu[0], &mu[0], tau_ncol * sizeof(double))

        # oldtau*mu + variable_data[i]*zi
        linear_algebra.dgemv(1, old_tau, mu, 0, tau_mu)
        for j in range(tau_ncol):
            dgemv_workspace[j] = tau_mu[j] + zi[j]*variable_data[i]

        linear_algebra.dgemv(1, inv_tau, dgemv_workspace, 0, mu)

        rho += 1

        for j in range(tau_ncol):
            delta_mu[j] = old_mu[j] - mu[j]

        zi_mu = 0
        for j in range(tau_ncol):
            zi_mu += zi[j] * mu[j]

        # Update phi
        phi += (variable_data[i] - zi_mu)*variable_data[i] + mahalanobis(delta_mu, old_tau, old_mu)

    return score


