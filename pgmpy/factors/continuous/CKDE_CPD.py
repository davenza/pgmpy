# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
import pandas as pd
from scipy.stats import norm

from pgmpy.factors.base import BaseFactor
from pgmpy.utils.gaussian_kde_ocl import shared_gaussian_kde_ocl as gaussian_kde_ocl
from pgmpy.models import HybridContinuousModel

from scipy.special import logsumexp

class CKDE_CPD(BaseFactor):

    def __init__(self, variable, gaussian_cpds, kde_instances, evidence=[], evidence_type={}):
        """
        Parameters
        ----------

        """
        self.variable = variable
        self.evidence = evidence
        self.gaussian_evidence = []
        self.kde_evidence = []

        self.kde_indices = []

        for idx, e in enumerate(evidence):
            if evidence_type[e] == HybridContinuousModel.NodeType.GAUSSIAN:
                self.gaussian_evidence.append(e)
            elif evidence_type[e] == HybridContinuousModel.NodeType.CKDE:
                self.kde_evidence.append(e)
                self.kde_indices.append(idx+1)

        self.variables = [variable] + evidence

        self.gaussian_cpds = gaussian_cpds
        self.n_gaussian = len(gaussian_cpds)

        self.gaussian_indices = []

        for gaussian_cpd in self.gaussian_cpds:
            tmp = []
            tmp.append(self.evidence.index(gaussian_cpd.variable) + 1)
            for e in gaussian_cpd.evidence:
                if e != variable:
                    tmp.append(self.evidence.index(e) + 1)
                else:
                    tmp.append(0)

            self.gaussian_indices.append(tmp)

            # self.gaussian_indices.append([self.evidence.index(e) + 1 if e != variable else 0 for e in gaussian_cpd.evidence])

        self.kde_instances = kde_instances
        self.joint_kde = gaussian_kde_ocl(self.kde_instances.values)
        self.n_kde = self.kde_instances.shape[1] - 1

        super(CKDE_CPD, self).__init__()

    @property
    def pdf(self):
        def _pdf(*args):
            prob = 1
            for i, gaussian_cpd in enumerate(self.gaussian_cpds):
                indices = self.gaussian_indices[i]
                subset_data = [args[idx] for idx in indices]
                prob *= gaussian_cpd.pdf(subset_data)

            prob *= self.joint_kde(args)

            prob /= self._denominator(args)

            return prob

        return _pdf

    def _denominator(self, instance):
        pass

    @property
    def logpdf(self):
        def _logpdf(*args):
            prob = 0
            for i, gaussian_cpd in enumerate(self.gaussian_cpds):
                indices = self.gaussian_indices[i]
                subset_data = [args[idx] for idx in indices]
                prob += gaussian_cpd.logpdf(subset_data)

            prob += self.joint_kde.logpdf(args)

            prob -= self._logdenominator(args)

            return prob

        return _logpdf

    def _logdenominator(self, instance):
        pass

    def logpdf_dataset(self, dataset):
        prob = 0
        for i, gaussian_cpd in enumerate(self.gaussian_cpds):
            indices = self.gaussian_indices[i]
            subset_data = dataset.iloc[:, indices]
            prob += gaussian_cpd.logpdf_dataset(subset_data)

        prob += self.joint_kde.logpdf(dataset.loc[:,[self.variable] + self.kde_evidence]).sum()

        prob -= self._logdenominator_dataset(dataset)

        return prob

    def _logdenominator_dataset(self, dataset):
        covariance = self.joint_kde.covariance
        precision = self.joint_kde.inv_cov

        logvar_mult = 0
        a = precision[0,0]
        for gaussian_cpd in self.gaussian_cpds:
            k_index = gaussian_cpd.evidence.index(self.variable)
            a += (gaussian_cpd.beta[k_index+1] * gaussian_cpd.beta[k_index+1]) / gaussian_cpd.variance
            logvar_mult += np.log(np.sqrt(gaussian_cpd.variance))

        a *= 0.5

        cte = dataset.shape[0]*(0.5*np.log(np.pi) -
              np.log(self.kde_instances.shape[0]) -
              0.5*np.log(a) -
              0.5*(self.n_gaussian + self.n_kde + 1)*np.log(2*np.pi) -
              0.5*np.log(np.linalg.det(covariance)) -
              logvar_mult)

        prob = 0

        for i in range(dataset.shape[0]):
            Ti = (dataset.iloc[i, [0] + self.kde_indices] - self.kde_instances).values

            bi = (self.kde_instances.iloc[:,0]*precision[0,0] - np.dot(Ti[:,1:], precision[0, 1:])).values

            ci = (np.sum(np.dot(Ti[:,1:], precision[1:, 1:])*Ti[:,1:], axis=1) -
                 2*self.kde_instances.iloc[:,0]*np.dot(Ti[:,1:], precision[0, 1:]) +
                 (self.kde_instances.iloc[:,0]*self.kde_instances.iloc[:,0])*precision[0,0]).values


            for j, gaussian_cpd in enumerate(self.gaussian_cpds):
                k_index = gaussian_cpd.evidence.index(self.variable)

                Bjk = gaussian_cpd.beta[k_index+1]
                Gj = dataset.iloc[i, self.evidence.index(gaussian_cpd.variable)+1]

                indices = self.gaussian_indices[j]
                subset_data = dataset.iloc[i, indices].values
                Cj = gaussian_cpd.beta[0] + np.dot(gaussian_cpd.beta[1:], subset_data[1:]) - Bjk*dataset.iloc[i,0]

                diff = Cj - Gj

                bi -= (Bjk * diff) / gaussian_cpd.variance

                ci += (diff * diff) / gaussian_cpd.variance


            ci *= -0.5
            prob += logsumexp(((bi*bi) / (4*a)) + ci)

        return cte + prob

    def copy(self):
        """
        Returns a copy of the distribution.

        Returns
        -------
        LinearGaussianCPD: copy of the distribution

        Examples
        --------
        >>> from pgmpy.factors.continuous import LinearGaussianCPD
        >>> cpd = LinearGaussianCPD('Y',  [0.2, -2, 3, 7], 9.6, ['X1', 'X2', 'X3'])
        >>> copy_cpd = cpd.copy()
        >>> copy_cpd.variable
        'Y'
        >>> copy_cpd.evidence
        ['X1', 'X2', 'X3']
        """
        # TODO: Is this a real copy?
        copy_cpd = CKDE_CPD(self.variable, self.gaussian_cpds, self.kde_instances, list(self.evidence))
        return copy_cpd

    def __str__(self):
        pass