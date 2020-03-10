from __future__ import division

from collections import defaultdict
import numpy as np
import pandas as pd
import networkx as nx
import logging

from pgmpy.models import BayesianModel
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.factors.distributions import GaussianDistribution

import pickle


class LinearGaussianBayesianNetwork(BayesianModel):
    """
    A Linear Gaussain Bayesian Network is a Bayesian Network, all
    of whose variables are continuous, and where all of the CPDs
    are linear Gaussians.

    An important result is that the linear Gaussian Bayesian Networks
    are an alternative representation for the class of multivariate
    Gaussian distributions.

    """

    def __init__(self, ebunch=None):
        super(LinearGaussianBayesianNetwork, self).__init__()
        if ebunch:
            self.add_edges_from(ebunch)
        self.cpds = []

    @classmethod
    def load_model(cls, filename):

        with open(filename, 'rb') as pickle_file:
            o = pickle.load(pickle_file)

        if type(o) is BayesianModel:
            out = LinearGaussianBayesianNetwork(o.edges)
            out.add_nodes_from(o)
        else:
            raise ValueError("Pickle object is not a BayesianModel.")
        return out

    def add_cpds(self, *cpds):
        """
        Add linear Gaussian CPD (Conditional Probability Distribution)
        to the Bayesian Model.

        Parameters
        ----------
        cpds  :  instances of LinearGaussianCPD
            List of LinearGaussianCPDs which will be associated with the model

        Examples
        --------
        >>> from pgmpy.models import LinearGaussianBayesianNetwork
        >>> from pgmpy.factors.continuous import LinearGaussianCPD
        >>> model = LinearGaussianBayesianNetwork([('x1', 'x2'), ('x2', 'x3')])
        >>> cpd1 = LinearGaussianCPD('x1', [1], 4)
        >>> cpd2 = LinearGaussianCPD('x2', [-5, 0.5], 4, ['x1'])
        >>> cpd3 = LinearGaussianCPD('x3', [4, -1], 3, ['x2'])
        >>> model.add_cpds(cpd1, cpd2, cpd3)
        >>> for cpd in model.cpds:
                print(cpd)

        P(x1) = N(1; 4)
        P(x2| x1) = N(0.5*x1_mu); -5)
        P(x3| x2) = N(-1*x2_mu); 4)

        """
        if isinstance(cpds[0], list):
            cpds = cpds[0]

        for cpd in cpds:
            if not isinstance(cpd, LinearGaussianCPD):
                raise ValueError("Only LinearGaussianCPD can be added.")

            if set(cpd.variables) - set(cpd.variables).intersection(set(self.nodes())):
                raise ValueError("CPD defined on variable not in the model", cpd)

            for prev_cpd_index in range(len(self.cpds)):
                if self.cpds[prev_cpd_index].variable == cpd.variable:
                    logging.warning(
                        "Replacing existing CPD for {var}".format(var=cpd.variable)
                    )
                    self.cpds[prev_cpd_index] = cpd
                    break
            else:
                self.cpds.append(cpd)

    def get_cpds(self, node=None):
        """
        Returns the cpd of the node. If node is not specified returns all the CPDs
        that have been added till now to the graph

        Parameter
        ---------
        node: any hashable python object (optional)
            The node whose CPD we want. If node not specified returns all the
            CPDs added to the model.

        Returns
        -------
        A list of linear Gaussian CPDs.

        Examples
        --------
        >>> from pgmpy.models import LinearGaussianBayesianNetwork
        >>> from pgmpy.factors.continuous import LinearGaussianCPD
        >>> model = LinearGaussianBayesianNetwork([('x1', 'x2'), ('x2', 'x3')])
        >>> cpd1 = LinearGaussianCPD('x1', [1], 4)
        >>> cpd2 = LinearGaussianCPD('x2', [-5, 0.5], 4, ['x1'])
        >>> cpd3 = LinearGaussianCPD('x3', [4, -1], 3, ['x2'])
        >>> model.add_cpds(cpd1, cpd2, cpd3)
        >>> model.get_cpds()
        """
        return super(LinearGaussianBayesianNetwork, self).get_cpds(node)

    def remove_cpds(self, *cpds):
        """
        Removes the cpds that are provided in the argument.

        Parameters
        ----------
        *cpds: LinearGaussianCPD object
            A LinearGaussianCPD object on any subset of the variables
            of the model which is to be associated with the model.

        Examples
        --------
        >>> from pgmpy.models import LinearGaussianBayesianNetwork
        >>> from pgmpy.factors.continuous import LinearGaussianCPD
        >>> model = LinearGaussianBayesianNetwork([('x1', 'x2'), ('x2', 'x3')])
        >>> cpd1 = LinearGaussianCPD('x1', [1], 4)
        >>> cpd2 = LinearGaussianCPD('x2', [-5, 0.5], 4, ['x1'])
        >>> cpd3 = LinearGaussianCPD('x3', [4, -1], 3, ['x2'])
        >>> model.add_cpds(cpd1, cpd2, cpd3)
        >>> for cpd in model.get_cpds():
                print(cpd)

        P(x1) = N(1; 4)
        P(x2| x1) = N(0.5*x1_mu); -5)
        P(x3| x2) = N(-1*x2_mu); 4)

        >>> model.remove_cpds(cpd2, cpd3)
        >>> for cpd in model.get_cpds():
                print(cpd)

        P(x1) = N(1; 4)

        """
        return super(LinearGaussianBayesianNetwork, self).remove_cpds(*cpds)

    def to_joint_gaussian(self):
        """
        The linear Gaussian Bayesian Networks are an alternative
        representation for the class of multivariate Gaussian distributions.
        This method returns an equivalent joint Gaussian distribution.

        Returns
        -------
        GaussianDistribution: An equivalent joint Gaussian
                                   distribution for the network.

        Reference
        ---------
        Section 7.2, Example 7.3,
        Probabilistic Graphical Models, Principles and Techniques

        Examples
        --------
        >>> from pgmpy.models import LinearGaussianBayesianNetwork
        >>> from pgmpy.factors.continuous import LinearGaussianCPD
        >>> model = LinearGaussianBayesianNetwork([('x1', 'x2'), ('x2', 'x3')])
        >>> cpd1 = LinearGaussianCPD('x1', [1], 4)
        >>> cpd2 = LinearGaussianCPD('x2', [-5, 0.5], 4, ['x1'])
        >>> cpd3 = LinearGaussianCPD('x3', [4, -1], 3, ['x2'])
        >>> model.add_cpds(cpd1, cpd2, cpd3)
        >>> jgd = model.to_joint_gaussian()
        >>> jgd.variables
        ['x1', 'x2', 'x3']
        >>> jgd.mean
        array([[ 1. ],
               [-4.5],
               [ 8.5]])
        >>> jgd.covariance
        array([[ 4.,  2., -2.],
               [ 2.,  5., -5.],
               [-2., -5.,  8.]])

        """
        variables = list(nx.topological_sort(self))
        mean = np.zeros(len(variables))
        covariance = np.zeros((len(variables), len(variables)))

        for node_idx in range(len(variables)):
            cpd = self.get_cpds(variables[node_idx])
            mean[node_idx] = (
                sum(
                    [
                        coeff * mean[variables.index(parent)]
                        for coeff, parent in zip(cpd.beta[1:], cpd.evidence)
                    ]
                )
                + cpd.beta[0]
            )
            covariance[node_idx, node_idx] = (
                sum(
                    [
                        coeff
                        * coeff
                        * covariance[variables.index(parent), variables.index(parent)]
                        for coeff, parent in zip(cpd.beta[1:], cpd.evidence)
                    ]
                )
                + cpd.variance
            )

        for node_i_idx in range(0,len(variables)):
            for node_j_idx in range(0, len(variables)):
                if covariance[node_j_idx, node_i_idx] != 0:
                    covariance[node_i_idx, node_j_idx] = covariance[
                        node_j_idx, node_i_idx
                    ]
                else:
                    cpd_j = self.get_cpds(variables[node_j_idx])
                    covariance[node_i_idx, node_j_idx] = sum(
                        [
                            coeff * covariance[node_i_idx, variables.index(parent)]
                            for coeff, parent in zip(cpd_j.beta[1:], cpd_j.evidence)
                        ]
                    )

        return GaussianDistribution(variables, mean, covariance)

    def check_model(self):
        """
        Checks the model for various errors. This method checks for the following
        error -

        * Checks if the CPDs associated with nodes are consistent with their parents.

        Returns
        -------
        check: boolean
            True if all the checks pass.

        """
        for node in self.nodes():
            cpd = self.get_cpds(node=node)

            if isinstance(cpd, LinearGaussianCPD):
                if set(cpd.evidence) != set(self.get_parents(node)):
                    raise ValueError(
                        "CPD associated with %s doesn't have "
                        "proper parents associated with it." % node
                    )
        return True

    def get_cardinality(self, node):
        """
        Cardinality is not defined for continuous variables.
        """
        raise ValueError("Cardinality is not defined for continuous variables.")

    def fit(
        self, data, estimator=None, complete_samples_only=True, **kwargs
    ):
        """
        Implemented fit.
        """
        from pgmpy.estimators import MaximumLikelihoodEstimator, BaseEstimator

        if estimator is None:
            estimator = MaximumLikelihoodEstimator
        else:
            if not issubclass(estimator, BaseEstimator):
                raise TypeError("Estimator object should be a valid pgmpy estimator.")

        if any(data.dtypes != 'float64'):
            raise ValueError("All columns should be continuous (float64 dtype).")

        _estimator = estimator(
            self,
            data,
            complete_samples_only=complete_samples_only,
        )
        cpds_list = _estimator.get_parameters(**kwargs)
        self.add_cpds(*cpds_list)

    def predict(self, data):
        """
        Implemented predict.
        """

        if set(data.columns) == set(self.nodes()):
            raise ValueError("No variable missing in data. Nothing to predict")

        elif set(data.columns) - set(self.nodes()):
            raise ValueError("Data has variables which are not in the model")

        joint = self.to_joint_gaussian()

        pred_values = defaultdict(list)

        for _, data_point in data.iterrows():
            reduced = joint.reduce(data_point.to_dict(), inplace=False)

            for k, v in zip(reduced.variables, reduced.mean[0]):
                pred_values[k].append(v)

        return pd.DataFrame(pred_values, index=data.index)

    def to_markov_model(self):
        """
        For now, to_markov_model method has not been implemented for LinearGaussianBayesianNetwork.
        """
        raise NotImplementedError(
            "to_markov_model method has not been implemented for LinearGaussianBayesianNetwork."
        )

    def is_imap(self, JPD):
        """
        For now, is_imap method has not been implemented for LinearGaussianBayesianNetwork.
        """
        raise NotImplementedError(
            "is_imap method has not been implemented for LinearGaussianBayesianNetwork."
        )

    def logpdf_dataset(self, data):
        logpdf = np.zeros((data.shape[0],))

        for n in self.nodes:
            cpd = self.get_cpds(n)
            logpdf += cpd.logpdf_dataset(data)

        return logpdf
