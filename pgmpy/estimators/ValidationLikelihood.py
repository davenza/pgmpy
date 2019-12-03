#!/usr/bin/env python

from math import log

import numpy as np
import scipy
from pgmpy.estimators import StructureScore

from sklearn.model_selection import train_test_split, KFold

from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.factors.continuous import NodeType

class ValidationLikelihood(StructureScore):
    def __init__(self, data, validation_ratio=0.2, k=10, seed=0, **kwargs):
        self.seed = seed
        self.validation_ratio = validation_ratio
        self.data, self.validation_data = train_test_split(data, self.validation_ratio, shuffle=True, random_state=seed)
        self.k = k
        self.fold_indices = list(KFold(k, shuffle=True, random_state=seed).split(self.data))
        super(ValidationLikelihood, self).__init__(data, **kwargs)

    # def change_seed(self, seed):
    #     self.seed = seed
    #     self.train_data, self.validation_data = train_test_split(self.data, self.validation_ratio, shuffle=True, random_state=seed)

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
                    cpd = MaximumLikelihoodEstimator.ckde_estimate_with_parents(variable, parents, parent_types, train_data)
                except np.linalg.LinAlgError:
                    return np.nan
            else:
                raise ValueError("Wrong node type for HybridContinuousModel.")

            test_data = node_data.iloc[test_indices]

            score += cpd.logpdf_dataset(test_data).sum()

        return score

    def validation_local_score(self, variable, parents, variable_type, parent_types):
        parents = list(parents)
        node_data = self.data[[variable] + parents].dropna()
        validation_data = self.validation_data[[variable] + parents].dropna()

        if variable_type == NodeType.GAUSSIAN:
            cpd = MaximumLikelihoodEstimator.gaussian_estimate_with_parents(variable, parents, node_data)
            if cpd is None:
                return np.nan
        elif variable_type == NodeType.CKDE:
            try:
                cpd = MaximumLikelihoodEstimator.ckde_estimate_with_parents(variable, parents, parent_types, node_data)
            except np.linalg.LinAlgError:
                return np.nan
        else:
            raise ValueError("Wrong node type for HybridContinuousModel.")

        return cpd.logpdf_dataset(validation_data).sum()