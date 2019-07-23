#!/usr/bin/env python

from math import log

import numpy as np
import scipy
from pgmpy.estimators import StructureScore

from scipy.special import loggamma

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
        parents = list(parents)
        node_data = self.data[[variable] + parents].dropna()
        N = node_data.shape[0]
        linregress_data = np.column_stack((np.ones(N), node_data[parents]))

        current_rho = self.iss + len(parents) if self.rho is None else self.rho
        current_phi = self.phi_coef * self.data[variable].var()

        current_tau, current_inv_tau = self._build_tau(parents)

        current_mu = self._build_mu(variable, parents)

        score = 0
        for (index, row) in enumerate(linregress_data):

            logscale = np.log(current_phi) + np.log1p(row.T.dot(current_inv_tau).dot(row))

            y = node_data.iloc[index, 0]
            diff_regress = y - np.inner(row, current_mu)

            score += loggamma(0.5 * (current_rho + 1)) - loggamma(0.5 * current_rho) - 0.5 * (logscale + np.log(np.pi))\
                     - 0.5 * (current_rho + 1) * np.log1p(np.inner(diff_regress, diff_regress) / np.exp(logscale))

            tau_mu = current_tau.dot(current_mu)

            current_tau += np.outer(row, row)
            current_inv_tau = np.linalg.inv(current_tau)

            new_mu = np.dot(current_inv_tau, tau_mu + row*y)
            delta_mu = current_mu - new_mu
            current_mu = new_mu

            current_rho += 1
            current_phi += (y - np.inner(row, current_mu))*y + np.dot(delta_mu, tau_mu)


        return score


        # ###############################
        # BATCH IMPLEMENTATION:
        # ###############################
        # TODO FIXME: This might be too slow as there are some possibly matrices involved:
        #
        # scale is NxN
        # tau is KxK
        #
        # new_tau = self.tau + linregress_data.T.dot(linregress_data)
        # tau_inv = np.linalg.inv(new_tau)
        # new_mu = np.dot(tau_inv, self.tau.dot(self.mu) - linregress_data.T.dot(node_data))
        # new_rho = self.rho + N
        # new_phi = self.phi + np.dot(node_data - linregress_data.dot(new_mu), node_data) + \
        #           np.dot(self.mu - new_mu, self.tau.dot(self.mu))
        #
        #
        # scale = new_phi*(np.eye(N) + np.dot(linregress_data, tau_inv.dot(linregress_data.T)))
        # scale_inv = np.linalg.inv(scale)
        # logdet = np.linalg.slogdet(scale)[1]
        #
        # diff_regress = node_data - linregress_data.dot(new_mu)
        #
        # return loggamma(0.5*(new_rho + N)) - loggamma(0.5*new_rho) - 0.5*np.log(logdet) - 0.5*np.log(np.power(np.pi, N)) + \
        #     -0.5*(new_rho + N)*np.log1p(np.dot(diff_regress.T.dot(scale_inv), diff_regress))
