from pgmpy.models import BayesianModel

from pgmpy.factors.continuous import LinearGaussianCPD, CKDE_CPD

import networkx as nx

class HybridContinuousModel(BayesianModel):

    class NodeType:
        GAUSSIAN = 0
        CKDE = 1

        @classmethod
        def str(cls, n):
            if n == cls.GAUSSIAN:
                return "GAUSSIAN"
            elif n == cls.CKDE:
                return "CKDE"
            else:
                raise ValueError("Value not valid for NodeType")

    def __init__(self, ebunch=None, node_type=None):
        super(HybridContinuousModel, self).__init__()

        self.node_type = {}
        self.cpds = []

        if ebunch:
            self.add_edges_from(ebunch, node_type=node_type)

            if node_type:
                set_nodes = set(self.nodes)
                keys_nodes = set(node_type.keys())

                difference = keys_nodes - set_nodes
                if difference:
                    raise ValueError("Nodes {} are present in node_type, but not in the model.")

                self.node_type = node_type

                other_difference = set_nodes - keys_nodes
                if other_difference:
                    for n in other_difference:
                        self.node_type[n] = HybridContinuousModel.NodeType.GAUSSIAN
            else:
                self.node_type = {}
                for n in self.nodes:
                    self.node_type[n] = HybridContinuousModel.NodeType.GAUSSIAN



    def add_edge(self, u, v, **kwargs):
        if u == v:
            raise ValueError("Self loops are not allowed.")
        if u in self.nodes() and v in self.nodes() and nx.has_path(self, v, u):
            raise ValueError(
                "Loops are not allowed. Adding the edge from (%s->%s) forms a loop."
                % (u, v)
            )
        else:
            if "node_type" in kwargs:
                if u in self.nodes and self.node_type[u] != kwargs["node_type"][u]:
                    logging.warning("Node {} with a different assigned node type. Previous type {}, New type {}", u,
                                    HybridContinuousModel.NodeType.str(self.node_type[u]),
                                    kwargs["node_type"][u]
                                    )
                if v in self.nodes and self.node_type[v] != kwargs["node_type"][v]:
                    logging.warning("Node {} with a different assigned node type. Previous type {}, New type {}", v,
                                    HybridContinuousModel.NodeType.str(self.node_type[v]),
                                    kwargs["node_type"][v]
                                    )

                self.node_type[u] = kwargs["node_type"][u]
                self.node_type[v] = kwargs["node_type"][v]
            else:
                if not u in self.nodes:
                    self.node_type[u] = HybridContinuousModel.NodeType.GAUSSIAN
                if not v in self.nodes:
                    self.node_type[v] = HybridContinuousModel.NodeType.GAUSSIAN

            super(HybridContinuousModel, self).add_edge(u, v, **kwargs)

    def add_node(self, node, weight=None, **kwargs):
        if "node_type" in kwargs:
            if not isinstance(kwargs["node_type"], HybridContinuousModel.NodeType):
                raise TypeError("Node type should be of type HybridContinuousModel.NodeType")
            self.node_type[node] = kwargs["node_type"]
        else:
            self.node_type[node] = HybridContinuousModel.NodeType.GAUSSIAN

        super(HybridContinuousModel, self).add_node(node, weight=weight)

    def add_nodes_from(self, nodes, weights=None, **kwargs):
        if "node_type" in kwargs:
            if not isinstance(kwargs["node_type"], dict):
                raise TypeError("Node type should be a dictionary of pairs (node, HybridContinuousModel.NodeType).")

            for v in kwargs["node_type"].values():
                if not isinstance(v, HybridContinuousModel.NodeType):
                    raise TypeError("Node type should be of type HybridContinuousModel.NodeType")

            self.node_type.update(kwargs["node_type"])
        else:
            self.node_type.update({n: HybridContinuousModel.NodeType.GAUSSIAN for n in nodes})

        super(HybridContinuousModel, self).add_nodes_from(nodes, weights=weights)

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
        for cpd in cpds:
            if self.node_type[cpd.variable] == HybridContinuousModel.NodeType.GAUSSIAN and not isinstance(cpd, LinearGaussianCPD):
                raise ValueError("Only LinearGaussianCPD can be added for {} node types.",
                                 HybridContinuousModel.NodeType.str(HybridContinuousModel.NodeType.GAUSSIAN))
            elif self.node_type[cpd.variable] == HybridContinuousModel.NodeType.CKDE and not isinstance(cpd, CKDE_CPD):
                raise ValueError("Only CKDE_CPD can be added for {} node types.",
                                 HybridContinuousModel.NodeType.str(HybridContinuousModel.NodeType.CKDE))

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
        return super(HybridContinuousModel, self).get_cpds(node)

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
        return super(HybridContinuousModel, self).remove_cpds(*cpds)

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

            if self.node_type[cpd.variable] == HybridContinuousModel.NodeType.GAUSSIAN and not isinstance(cpd, LinearGaussianCPD):
                raise ValueError("Only LinearGaussianCPD can be added for {} node types.",
                                 HybridContinuousModel.NodeType.str(HybridContinuousModel.NodeType.GAUSSIAN))
            elif self.node_type[cpd.variable] == HybridContinuousModel.NodeType.CKDE and not isinstance(cpd, CKDE_CPD):
                raise ValueError("Only CKDE_CPD can be added for {} node types.",
                                 HybridContinuousModel.NodeType.str(HybridContinuousModel.NodeType.CKDE))


            if set(cpd.evidence) != set(self.get_parents(node)):
                raise ValueError(
                    "CPD associated with %s doesn't have "
                    "proper parents associated with it." % node
                )
        return True

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
        pass
        # if set(data.columns) == set(self.nodes()):
        #     raise ValueError("No variable missing in data. Nothing to predict")
        #
        # elif set(data.columns) - set(self.nodes()):
        #     raise ValueError("Data has variables which are not in the model")
        #
        # joint = self.to_joint_gaussian()
        #
        # pred_values = defaultdict(list)
        #
        # for _, data_point in data.iterrows():
        #     reduced = joint.reduce(data_point.to_dict(), inplace=False)
        #
        #     for k, v in zip(reduced.variables, reduced.mean[0]):
        #         pred_values[k].append(v)
        #
        # return pd.DataFrame(pred_values, index=data.index)