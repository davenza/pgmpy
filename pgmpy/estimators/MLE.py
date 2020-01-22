# coding:utf-8

import numpy as np

from pgmpy.estimators import ParameterEstimator
from pgmpy.factors.discrete import TabularCPD
from pgmpy.factors.continuous import LinearGaussianCPD, NodeType, CKDE_CPD
from pgmpy.models import BayesianModel, LinearGaussianBayesianNetwork, HybridContinuousModel


class MaximumLikelihoodEstimator(ParameterEstimator):
    def __init__(self, model, data, **kwargs):
        """
        Class used to compute parameters for a model using Maximum Likelihood Estimation.

        Parameters
        ----------
        model: A pgmpy.models.BayesianModel instance

        data: pandas DataFrame object
            DataFrame object with column names identical to the variable names of the network.
            (If some values in the data are missing the data cells should be set to `numpy.NaN`.
            Note that pandas converts each column containing `numpy.NaN`s to dtype `float`.)

        state_names: dict (optional)
            A dict indicating, for each variable, the discrete set of states
            that the variable can take. If unspecified, the observed values
            in the data set are taken to be the only possible states.

        complete_samples_only: bool (optional, default `True`)
            Specifies how to deal with missing data, if present. If set to `True` all rows
            that contain `np.NaN` somewhere are ignored. If `False` then, for each variable,
            every row where neither the variable nor its parents are `np.NaN` is used.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.models import BayesianModel
        >>> from pgmpy.estimators import MaximumLikelihoodEstimator
        >>> data = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
        ...                       columns=['A', 'B', 'C', 'D', 'E'])
        >>> model = BayesianModel([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
        >>> estimator = MaximumLikelihoodEstimator(model, data)
        """

        if not isinstance(model, (BayesianModel, LinearGaussianBayesianNetwork, HybridContinuousModel)):
            raise NotImplementedError(
                "Maximum Likelihood Estimate is only implemented for BayesianModel, LinearGaussianBayesianNetwork"
                "and HybridContinuousModel"
            )

        super(MaximumLikelihoodEstimator, self).__init__(model, data, **kwargs)

    def get_parameters(self):
        """
        Method to estimate the model parameters (CPDs) using Maximum Likelihood Estimation.

        Returns
        -------
        parameters: list
            List of TabularCPDs, one for each variable of the model

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.models import BayesianModel
        >>> from pgmpy.estimators import MaximumLikelihoodEstimator
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 4)),
        ...                       columns=['A', 'B', 'C', 'D'])
        >>> model = BayesianModel([('A', 'B'), ('C', 'B'), ('C', 'D'))
        >>> estimator = MaximumLikelihoodEstimator(model, values)
        >>> estimator.get_parameters()
        [<TabularCPD representing P(C:2) at 0x7f7b534251d0>,
        <TabularCPD representing P(B:2 | C:2, A:2) at 0x7f7b4dfd4da0>,
        <TabularCPD representing P(A:2) at 0x7f7b4dfd4fd0>,
        <TabularCPD representing P(D:2 | C:2) at 0x7f7b4df822b0>]
        """
        parameters = []

        for node in sorted(self.model.nodes()):
            cpd = self.estimate_cpd(node)
            parameters.append(cpd)

        return parameters

    def estimate_cpd(self, node):
        """
        Method to estimate the CPD for a given variable.

        Parameters
        ----------
        node: int, string (any hashable python object)
            The name of the variable for which the CPD is to be estimated.

        Returns
        -------
        CPD: TabularCPD

        Examples
        --------
        >>> import pandas as pd
        >>> from pgmpy.models import BayesianModel
        >>> from pgmpy.estimators import MaximumLikelihoodEstimator
        >>> data = pd.DataFrame(data={'A': [0, 0, 1], 'B': [0, 1, 0], 'C': [1, 1, 0]})
        >>> model = BayesianModel([('A', 'C'), ('B', 'C')])
        >>> cpd_A = MaximumLikelihoodEstimator(model, data).estimate_cpd('A')
        >>> print(cpd_A)
        ╒══════╤══════════╕
        │ A(0) │ 0.666667 │
        ├──────┼──────────┤
        │ A(1) │ 0.333333 │
        ╘══════╧══════════╛
        >>> cpd_C = MaximumLikelihoodEstimator(model, data).estimate_cpd('C')
        >>> print(cpd_C)
        ╒══════╤══════╤══════╤══════╤══════╕
        │ A    │ A(0) │ A(0) │ A(1) │ A(1) │
        ├──────┼──────┼──────┼──────┼──────┤
        │ B    │ B(0) │ B(1) │ B(0) │ B(1) │
        ├──────┼──────┼──────┼──────┼──────┤
        │ C(0) │ 0.0  │ 0.0  │ 1.0  │ 0.5  │
        ├──────┼──────┼──────┼──────┼──────┤
        │ C(1) │ 1.0  │ 1.0  │ 0.0  │ 0.5  │
        ╘══════╧══════╧══════╧══════╧══════╛
        """
        if isinstance(self.model, LinearGaussianBayesianNetwork):
            return self.gaussian_estimate(node)
        elif isinstance(self.model, HybridContinuousModel):
            if self.model.node_type[node] == NodeType.GAUSSIAN:
                return self.gaussian_estimate(node)
            elif self.model.node_type[node] == NodeType.CKDE:
                return self.ckde_estimate(node)
        elif isinstance(self.model, BayesianModel):
            return self.discrete_estimate(node)

    def discrete_estimate(self, node):
        state_counts = self.state_counts(node)

        # if a column contains only `0`s (no states observed for some configuration
        # of parents' states) fill that column uniformly instead
        state_counts.ix[:, (state_counts == 0).all()] = 1

        parents = sorted(self.model.get_parents(node))
        parents_cardinalities = [len(self.state_names[parent]) for parent in parents]
        node_cardinality = len(self.state_names[node])

        # Get the state names for the CPD
        # FIXME: state_names is computed in the constructor.
        state_names = {node: list(state_counts.index)}
        if parents:
            state_names.update(
                {
                    state_counts.columns.names[i]: list(state_counts.columns.levels[i])
                    for i in range(len(parents))
                }
            )

        cpd = TabularCPD(
            node,
            node_cardinality,
            np.array(state_counts),
            evidence=parents,
            evidence_card=parents_cardinalities,
            state_names=state_names,
        )
        cpd.normalize()
        return cpd

    def gaussian_estimate(self, node):
        """
        Runs a linear regression with least squares method.
        :param node:
        :return:
        """
        parents = sorted(self.model.get_parents(node))
        node_data = self.data[[node] + parents].dropna()
        return MaximumLikelihoodEstimator.gaussian_estimate_with_parents(node, parents, node_data)

    @classmethod
    def gaussian_estimate_with_parents(cls, node, parents, data):
        linregress_data = np.column_stack((np.ones(data.shape[0]), data[parents]))
        (beta, res, _, _) = np.linalg.lstsq(linregress_data, data[node], rcond=None)

        if data.shape[0] <= 1 or res.size == 0 or res[0] == 0:
            return None
        else:
            variance = res[0] / (data.shape[0] - 1)

        cpd = LinearGaussianCPD(
            node,
            beta,
            variance,
            evidence=parents
        )

        return cpd

    def ckde_estimate(self, node):
        parents = sorted(self.model.get_parents(node))
        node_data = self.data[[node] + parents].dropna()
        return MaximumLikelihoodEstimator.ckde_estimate_with_parents(node, parents, self.model.node_type, node_data)

    @classmethod
    def ckde_estimate_with_parents(cls, node, parents, parent_types, data):
        gaussian_parents = []
        ckde_parents = []

        for parent in parents:
            if parent_types[parent] == NodeType.GAUSSIAN:
                gaussian_parents.append(parent)
            else:
                ckde_parents.append(parent)

        gaussian_cpds = []
        chain_rule_parents = [node] + ckde_parents

        for g in gaussian_parents:
            g_cpd = MaximumLikelihoodEstimator.gaussian_estimate_with_parents(g, chain_rule_parents, data)
            if g_cpd is None:
                raise ValueError("A gaussian CPD could not be estimated.")
            gaussian_cpds.append(g_cpd)
            chain_rule_parents = chain_rule_parents.copy()
            chain_rule_parents.append(g)

        return CKDE_CPD(node, gaussian_cpds, data.loc[:, [node] + ckde_parents], evidence=parents, evidence_type=parent_types)
