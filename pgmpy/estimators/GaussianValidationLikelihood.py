#!/usr/bin/env python

from math import log

import numpy as np
import scipy
from pgmpy.estimators import StructureScore, ValidationLikelihood

from sklearn.model_selection import train_test_split, KFold

from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.factors.continuous import NodeType


class GaussianValidationLikelihood(StructureScore):

    def __init__(self, data, validation_ratio=0.2, k=10, seed=0, **kwargs):
        self.seed = seed
        self.validation_ratio = validation_ratio
        self.data, self.validation_data = train_test_split(data, test_size=self.validation_ratio, shuffle=True,
                                                           random_state=seed)
        self.k = k
        self.fold_indices = list(KFold(k, shuffle=True, random_state=seed).split(self.data))
        self.validation_fold_indices = list(KFold(k, shuffle=True, random_state=seed).split(self.validation_data))
        super(GaussianValidationLikelihood, self).__init__(data, **kwargs)

    # def change_seed(self, seed):
    #     self.seed = seed
    #     self.train_data, self.validation_data = \
    #         train_test_split(self.data, self.validation_ratio, shuffle=True, random_state=seed)

    def local_score(self, variable, parents):
        score = 0
        parents = list(parents)
        node_data = self.data[[variable] + parents].dropna()

        for train_indices, test_indices in self.fold_indices:
            train_data = node_data.iloc[train_indices]
            cpd = MaximumLikelihoodEstimator.gaussian_estimate_with_parents(variable, parents, train_data)
            if cpd is None:
                return np.nan
            test_data = node_data.iloc[test_indices]

            score += cpd.logpdf_dataset(test_data).sum()

        return score

    def validation_local_score(self, variable, parents):
        parents = list(parents)
        node_data = self.data[[variable] + parents].dropna()
        validation_data = self.validation_data[[variable] + parents].dropna()

        cpd = MaximumLikelihoodEstimator.gaussian_estimate_with_parents(variable, parents, node_data)
        if cpd is None:
            return np.nan

        return cpd.logpdf_dataset(validation_data).sum()
