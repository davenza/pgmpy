# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
import pandas as pd
from scipy.stats import norm

from pgmpy.factors.base import BaseFactor
from pgmpy.utils.gaussian_kde_ocl import shared_gaussian_kde_ocl as gaussian_kde_ocl

class CKDE_CPD(BaseFactor):


    def __init__(self, variable, gaussian_cpds, kde_instances, evidence=[]):
        """
        Parameters
        ----------

        """
        self.variable = variable
        self.evidence = evidence
        self.variables = [variable] + evidence

        self.gaussian_cpds = gaussian_cpds
        self.n_gaussian = len(gaussian_cpds)

        self.gaussian_indices = []

        for gaussian_cpd in self.gaussian_cpds:
            self.gaussian_indices.append([self.evidence.index(e) for e in gaussian_cpd.evidence])

        self.kde_instances = kde_instances
        self.joint_kde = gaussian_kde_ocl(self.kde_instances)
        self.n_kde = self.kde_instances.shape[1]

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
        covariance = self.joint_kde.covariance
        precision = self.joint_kde.inv_cov

        var_mult = 1
        a = precision[0,0]
        for gaussian_cpd in self.gaussian_cpds:
            k_index = gaussian_cpd.evidence.index(self.variable)
            a += (gaussian_cpd.beta[k_index] * gaussian_cpd.beta[k_index]) / gaussian_cpd.variance
            var_mult *= gaussian_cpd.variance

        a *= 0.5



        cte = np.sqrt(np.pi) / (self.kde_instances.shape[0]
                                *np.sqrt(a)
                                *np.power(2*np.pi, (self.n_gaussian + self.n_kde + 1)*0.5)
                                *np.sqrt(np.linalg.det(covariance))
                                *var_mult
                                )

        for i in range(self.kde_instances.shape[0]):
            bi = instance[0]*precision[0,0] - np.dot(precision[0, 1:], instance[1:])


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
        copy_cpd = CKDE_CPD(self.variable, self.betas, self.variances, self.kde_instances, list(self.evidence))

        return copy_cpd

    def __str__(self):
        if self.evidence and self.beta.size > 1:
            # P(Y| X1, X2, X3) = N(-2*X1_mu + 3*X2_mu + 7*X3_mu; 0.2)
            rep_str = "P({node} | {parents}) = N({mu} {b_0}; {variance:0.3f})".format(
                node=str(self.variable),
                parents=", ".join([str(var) for var in self.evidence]),
                mu=" + ".join(
                    [
                        "{coeff:0.3f}*{parent}".format(coeff=coeff, parent=parent)
                        for coeff, parent in zip(self.beta[1:], self.evidence)
                    ]
                ),
                b_0="+ {b_0:0.3f}".format(b_0=self.beta[0]) if self.beta[0] >= 0 else "- {b_0}".format(b_0=abs(self.beta[0])),
                variance=self.variance,
            )
        else:
            # P(X) = N(1, 4)
            rep_str = "P({X}) = N({beta_0:0.3f}; {variance})".format(
                X=str(self.variable),
                beta_0=self.beta[0],
                variance=self.variance,
            )
        return rep_str
