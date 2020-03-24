import numpy as np
import networkx as nx
from pgmpy.models import BayesianModel
from pgmpy.factors.continuous import ConditionalKDE

import logging
import pickle

class KDEBayesianNetwork(BayesianModel):

    def __init__(self, ebunch=None):
        super(KDEBayesianNetwork, self).__init__()

        self.node_type = {}
        self.cpds = []

        if ebunch:
            self.add_edges_from(ebunch)

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
            if not isinstance(cpd, ConditionalKDE):
                raise ValueError("Only ConditionalKDE CPDs can be added.")

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
        if any(data.dtypes != 'float64'):
            raise ValueError("All columns should be continuous (float64 dtype).")
        cpds_list = []
        for node in self.nodes():
            parents = self.get_parents(node)
            node_data = data.loc[:, [node] + parents].dropna()

            bw_method = None
            if "bw_method" in kwargs:
                bw_method = kwargs["bw_method"]

            node_cpd = ConditionalKDE(node, node_data, evidence=parents, bw_method=bw_method)
            cpds_list.append(node_cpd)

        self.add_cpds(*cpds_list)


    def copy(self):
        model_copy = KDEBayesianNetwork()
        model_copy.add_nodes_from(self.nodes())
        model_copy.add_edges_from(self.edges())
        if self.cpds:
            model_copy.add_cpds(*[cpd.copy() for cpd in self.cpds])
        return model_copy

    # TODO: Add save model.

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

    def save_model(self, filename, save_parameters=False, protocol=pickle.HIGHEST_PROTOCOL):
        if self.cpds and save_parameters:
            self.save_parameters = save_parameters
        else:
            self.save_parameters = False

        if filename[-4:] != '.pkl':
            filename += '.pkl'

        with open(filename, 'wb') as pickle_file:
            pickle.dump(self, pickle_file, protocol)

        del self.save_parameters

    @classmethod
    def load_model(cls, filename):
        with open(filename, 'rb') as pickle_file:
            o = pickle.load(pickle_file)

        if not hasattr(o, 'cpds'):
            o.cpds = []

        return o

    def __getstate__(self):
        if self.save_parameters:
            state = self.__dict__.copy()
            del state['save_parameters']
            return self.__dict__
        else:
            state = self.__dict__.copy()
            del state['save_parameters']
            del state['cpds']
            return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def logpdf_dataset(self, data):
        logpdf = np.zeros((data.shape[0],))

        for n in self.nodes:
            cpd = self.get_cpds(n)
            logpdf += cpd.logpdf_dataset(data)

        return logpdf
