#!/usr/bin/env python
import numpy as np
from pgmpy.estimators import StructureScore

from sklearn.model_selection import train_test_split, KFold

from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.factors.continuous import NodeType


class ValidationLikelihood(StructureScore):

    def __init__(self, data, validation_ratio=0.2, k=10, seed=0, **kwargs):
        self.seed = seed
        self.validation_ratio = validation_ratio
        self.data, self.validation_data = train_test_split(data, test_size=self.validation_ratio, shuffle=True,
                                                           random_state=seed)
        self.k = k
        self.fold_indices = list(KFold(k, shuffle=True, random_state=seed).split(self.data))
        self.validation_fold_indices = list(KFold(k, shuffle=True, random_state=seed).split(self.validation_data))

        super(ValidationLikelihood, self).__init__(data, **kwargs)

    # def change_seed(self, seed):
    #     self.seed = seed
    #     self.train_data, self.validation_data = \
    #         train_test_split(self.data, self.validation_ratio, shuffle=True, random_state=seed)

    def local_score(self, variable, parents, variable_type, parent_types):
        score = 0
        parents = list(parents)
        node_data = self.data[[variable] + parents].dropna()

        for train_indices, test_indices in self.fold_indices:
            # print("=========================")
            # print()
            train_data = node_data.iloc[train_indices,:]
            # print("train_data cv = " + str(len(train_data)))
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

            test_data = node_data.iloc[test_indices,:]

            score += cpd.logpdf_dataset(test_data).sum()

        return score

    def debug_score(self, variable, parents, variable_type, parent_types, expected_values=None):
        parents = list(parents)
        node_data = self.data[[variable] + parents].dropna()

        (train_indices, test_indices) = self.fold_indices[0]
        train_data = node_data.iloc[train_indices,:]

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

        test_data = node_data.iloc[test_indices,:]

        kde_score, gaussian_score, den_score = cpd.debug_logpdf_dataset(test_data)
        if expected_values is not None:
            exp_kde_score, exp_gaussian_score, exp_den_score = expected_values

            if np.any(~np.isclose(kde_score, exp_kde_score)):
                fail_indices = np.where(~np.isclose(kde_score, exp_kde_score))
                print("Fail kde scores")
                print("Indices " + str(fail_indices))
                print("score = " + str(kde_score[fail_indices]))
                score2, _ , _ = cpd.debug_logpdf_dataset(test_data)
                print("score2 = " + str(score2[fail_indices]))
                score3, _, _ = cpd.debug_logpdf_dataset(test_data)
                print("score3 = " + str(score3[fail_indices]))

            if np.any(~np.isclose(gaussian_score, exp_gaussian_score)):
                fail_indices = np.where(~np.isclose(gaussian_score, exp_gaussian_score))
                print("Fail gaussian scores")
                print("Indices " + str(fail_indices))
                print("score = " + str(gaussian_score[fail_indices]))
                _, score2, _ = cpd.debug_logpdf_dataset(test_data)
                print("score2 = " + str(score2[fail_indices]))
                _, score3, _ = cpd.debug_logpdf_dataset(test_data)
                print("score3 = " + str(score3[fail_indices]))

            if np.any(~np.isclose(den_score, exp_den_score)):
                fail_indices = np.where(~np.isclose(den_score, exp_den_score))
                print("Fail denominator scores")
                print("Indices " + str(fail_indices))
                print("score = " + str(den_score[fail_indices]))
                _, _, score2 = cpd.debug_logpdf_dataset(test_data)
                print("score2 = " + str(score2[fail_indices]))
                _, _, score3 = cpd.debug_logpdf_dataset(test_data)
                print("score3 = " + str(score3[fail_indices]))

        return kde_score, gaussian_score, den_score


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
