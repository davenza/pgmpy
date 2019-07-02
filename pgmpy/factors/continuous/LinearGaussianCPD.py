# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
import pandas as pd
from scipy.stats import norm

from pgmpy.factors.base import BaseFactor


class LinearGaussianCPD(BaseFactor):
    """
    For, X -> Y the Linear Gaussian model assumes that the mean
    of Y is a linear function of mean of X and the variance of Y does
    not depend on X.

    For example,
    $ p(Y|X) = N(-2x + 0.9 ; 1) $
    Here, $ x $ is the mean of the variable $ X $.

    Let $ Y $ be a continuous variable with continuous parents
    $ X1, X2, ..., Xk $. We say that $ Y $ has a linear Gaussian CPD
    if there are parameters $ \beta_0, \beta_1, ..., \beta_k $
    and $ \sigma_2 $ such that,

    $ p(Y |x1, x2, ..., xk) = \mathcal{N}(\beta_0 + x1*\beta_1 + ......... + xk*\beta_k ; \sigma_2) $

    In vector notation,

    $ p(Y |x) = \mathcal{N}(\beta_0 + \boldmath{β}.T * \boldmath{x} ; \sigma_2) $


    Reference: https://cedar.buffalo.edu/~srihari/CSE574/Chap8/Ch8-PGM-GaussianBNs/8.5%20GaussianBNs.pdf
    """

    def __init__(self, variable, beta, sigma, evidence=[]):
        """
        Parameters
        ----------

        variable: any hashable python object
            The variable whose CPD is defined.

        evidence_mean: Mean vector (numpy array) of the joint distribution, X

        evidence_variance: int, float
            The variance of the multivariate gaussian, X = ['x1', 'x2', ..., 'xn']

        evidence: iterable of any hashabale python objects
            An iterable of the parents of the variable. None if there are no parents.

        beta (optional): iterable of int or float
            An iterable representing the coefficient vector of the linear equation.
            The first term represents the constant term in the linear equation.


        Examples
        --------

        # For P(Y| X1, X2, X3) = N(-2x1 + 3x2 + 7x3 + 0.2; 9.6)

        >>> cpd = LinearGaussianCPD('Y',  [0.2, -2, 3, 7], 9.6, ['X1', 'X2', 'X3'])
        >>> cpd.variable
        'Y'
        >>> cpd.evidence
        ['x1', 'x2', 'x3']
        >>> cpd.beta_vector
        [0.2, -2, 3, 7]

        """
        self.variable = variable
        self.evidence = evidence

        self.beta = np.asarray(beta)
        self.sigma = sigma

        super(LinearGaussianCPD, self).__init__(
            [variable] + evidence, mean=self.mean, covariance=self.variance
        )

    @property
    def pdf(self):
        def _pdf(*args):
            # The first element of args is the value of the variable on which CPD is defined
            # and the rest of the elements give the mean values of the parent
            # variables.
            mean = self.beta[0] + np.dot(self.beta[1:], np.asarray(args))

            return norm.pdf(args[0], np.array(mean), self.sigma)

        return _pdf

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
        copy_cpd = LinearGaussianCPD(self.variable, self.beta, self.sigma, list(self.evidence))

        return copy_cpd

    def __str__(self):
        if self.evidence and self.beta.size > 1:
            # P(Y| X1, X2, X3) = N(-2*X1_mu + 3*X2_mu + 7*X3_mu; 0.2)
            rep_str = "P({node} | {parents}) = N({mu} + {b_0}; {sigma})".format(
                node=str(self.variable),
                parents=", ".join([str(var) for var in self.evidence]),
                mu=" + ".join(
                    [
                        "{coeff}*{parent}".format(coeff=coeff, parent=parent)
                        for coeff, parent in zip(self.beta[1:], self.evidence)
                    ]
                ),
                b_0=str(self.beta[0]),
                sigma=str(self.sigma),
            )
        else:
            # P(X) = N(1, 4)
            rep_str = "P({X}) = N({beta_0}; {variance})".format(
                X=str(self.variable),
                beta_0=str(self.beta[0]),
                variance=str(self.sigma),
            )
        return rep_str
