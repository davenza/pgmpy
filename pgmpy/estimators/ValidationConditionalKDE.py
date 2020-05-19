#!/usr/bin/env python
import numpy as np
from pgmpy.estimators import StructureScore

from sklearn.model_selection import train_test_split, KFold

from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.factors.continuous import ConditionalKDE


class ValidationConditionalKDE(StructureScore):

    """
    This score is valid for KDE Bayesian networks.
    """

    def __init__(self, data, validation_ratio=0.2, k=10, seed=0, **kwargs):
        self.seed = seed
        self.validation_ratio = validation_ratio
        self.data, self.validation_data = train_test_split(data, test_size=self.validation_ratio, shuffle=True,
                                                           random_state=seed)
        self.k = k
        self.fold_indices = list(KFold(k, shuffle=True, random_state=seed).split(self.data))

        super(ValidationConditionalKDE, self).__init__(data, **kwargs)

    def local_score(self, variable, parents):
        score = 0
        parents = list(parents)
        # FIXME Here we make a dropna(), but it is not applied in __init__()
        node_data = self.data[[variable] + parents].dropna()

        for train_indices, test_indices in self.fold_indices:
            train_data = node_data.iloc[train_indices,:]
            cpd = ConditionalKDE(variable, train_data, evidence=parents)
            test_data = node_data.iloc[test_indices,:]
            score += cpd.logpdf_dataset(test_data).sum()

        return score

    def validation_local_score(self, variable, parents):
        parents = list(parents)
        node_data = self.data[[variable] + parents].dropna()
        validation_data = self.validation_data[[variable] + parents].dropna()

        cpd = ConditionalKDE(variable, node_data, parents)

        return cpd.logpdf_dataset(validation_data).sum()
