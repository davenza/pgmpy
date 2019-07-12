#!/usr/bin/env python

from math import log

import numpy as np
import scipy
from pgmpy.estimators import StructureScore


class GaussianBicScore(StructureScore):
    def __init__(self, data, **kwargs):
        """
        Class for Bayesian structure scoring for BayesianModels with Dirichlet priors.
        The BIC/MDL score ("Bayesian Information Criterion", also "Minimal Descriptive Length") is a
        log-likelihood score with an additional penalty for network complexity, to avoid overfitting.
        The `score`-method measures how well a model is able to describe the given data set.

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
        super(GaussianBicScore, self).__init__(data, **kwargs)

    def local_score(self, variable, parents):
        'Computes a score that measures how much a \
        given variable is "influenced" by a given list of potential parents.'
        parents = list(parents)
        node_data = self.data[[variable] + parents].dropna()
        N = node_data.shape[0]
        linregress_data = np.column_stack((np.ones(N), node_data[parents]))
        (beta, res, _, _) = np.linalg.lstsq(linregress_data, node_data[variable], rcond=None)

        if node_data.shape[0] <= 1:
            variance = 0
        else:
            variance = res[0] / (N - 1)

        loglik = (1-N)/2 - N/2*np.log(2*np.pi) - N*np.log(np.sqrt(variance))
        return loglik - np.log(N) * 0.5 * (len(parents) + 2)

