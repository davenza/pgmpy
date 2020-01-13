#!/usr/bin/env python

import numpy as np
from pgmpy.estimators import StructureScore

from sklearn.model_selection import KFold

from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.factors.continuous import NodeType


class CVPredictiveLikelihood(StructureScore):

    def __init__(self, data, k=10, seed=0, **kwargs):
        self.seed = seed
        self.k = k
        self.fold_indices = list(KFold(k, shuffle=True, random_state=seed).split(data))
        super(CVPredictiveLikelihood, self).__init__(data, **kwargs)

    def change_seed(self, seed):
        self.seed = seed
        self.fold_indices = list(KFold(self.k, shuffle=True, random_state=seed).split(self.data))

    def local_score(self, variable, parents, variable_type, parent_types):
        score = 0
        parents = list(parents)
        node_data = self.data[[variable] + parents].dropna()

        for train_indices, test_indices in self.fold_indices:
            train_data = node_data.iloc[train_indices]
            if variable_type == NodeType.GAUSSIAN:
                cpd = MaximumLikelihoodEstimator.gaussian_estimate_with_parents(variable, parents, train_data)
                if cpd is None:
                    return np.nan
            elif variable_type == NodeType.CKDE:
                try:
                    cpd = MaximumLikelihoodEstimator.ckde_estimate_with_parents(variable, parents, parent_types,
                                                                                train_data)
                except np.linalg.LinAlgError:
                    return np.nan
            else:
                raise ValueError("Wrong node type for HybridContinuousModel.")

            test_data = node_data.iloc[test_indices]

            score += cpd.logpdf_dataset(test_data).sum()

        return score
