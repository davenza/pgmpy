from pgmpy.estimators import StructureEstimator, GaussianBicScore, BGeScore, K2Score, BdeuScore, BicScore
import numpy as np

class CachedHillClimbing(StructureEstimator):

    def __init__(self, data, scoring_method=None, blacklist=[], whitelist=[], **kwargs):
        """
        CacheDecomposableHillCliming implements a hill climbing algorithm caching the scores of different operators.
        This hill climbing procedure assumes the 'scoring_method' is decomposable, so it only updates the relevant
        operation scores when a change is applied to the graph.

        This method can blacklist arcs to prevent them to be included in the graphs.

        This method can whitelist arcs to ensure those arcs are included in the graph.

        :param data: Data to train the Bayesian Network.
        :param scoring_method: A decomposable score.
        :param blacklist: List of blacklisted arcs [(source_node, dest_node), (source_node, dest_node), ...]
        :param whitelist: List of blacklisted arcs [(source_node, dest_node), (source_node, dest_node), ...]
        :param kwargs:
        """
        continuous = False
        if np.all(data.dtypes == 'float64'):
            continuous = True

        if scoring_method is None:
            if continuous:
                self.scoring_method = GaussianBicScore(data, **kwargs)
            else:
                self.scoring_method = K2Score(data, **kwargs)
        else:
            self.scoring_method = scoring_method

        if continuous and not isinstance(self.scoring_method, (GaussianBicScore, BGeScore)):
            raise TypeError("Selected scoring_method {} incorrect for continuous data.".format(self.scoring_method))
        if not continuous and not isinstance(self.scoring_method, (BdeuScore, BicScore, K2Score)):
            raise TypeError("Selected scoring_method {} incorrect for discrete data".format(self.scoring_method))

        self.nodes = list(data.columns.values)
        self.nodes_indices = {var: index for index, var in enumerate(self.nodes)}
        self.blacklist = set(blacklist)
        self.whitelist = set(whitelist)

        nnodes = len(self.nodes)

        self.constraints_matrix = np.full((nnodes, nnodes), True, dtype=np.bool)
        np.fill_diagonal(self.constraints_matrix, False)
        for (source, dest) in self.blacklist:
            s = self.nodes_indices[source]
            d = self.nodes_indices[dest]
            self.constraints_matrix[s, d] = False

        for (source, dest) in self.whitelist:
            s = self.nodes_indices[source]
            d = self.nodes_indices[dest]
            self.constraints_matrix[s, d] = False
            self.constraints_matrix[d, s] = False

        self.total_num_arcs = self.constraints_matrix.sum()

        super(CachedHillClimbing, self).__init__(data, **kwargs)