#!/usr/bin/env python
from itertools import permutations

import numpy as np
import networkx as nx

from pgmpy.estimators import StructureEstimator, K2Score, GaussianBicScore, BdeuScore, BicScore
from pgmpy.base import DAG

import scipy

class HillClimbSearch(StructureEstimator):
    def __init__(self, data, scoring_method=None, blacklist=[], whitelist=[], **kwargs):
        """
        Class for heuristic hill climb searches for DAGs, to learn
        network structure from data. `estimate` attempts to find a model with optimal score.

        Parameters
        ----------
        data: pandas DataFrame object
            datafame object where each column represents one variable.
            (If some values in the data are missing the data cells should be set to `numpy.NaN`.
            Note that pandas converts each column containing `numpy.NaN`s to dtype `float`.)

        scoring_method: Instance of a `StructureScore`-subclass (`K2Score` is used as default)
            An instance of `K2Score`, `BdeuScore`, or `BicScore`.
            This score is optimized during structure estimation by the `estimate`-method.

        state_names: dict (optional)
            A dict indicating, for each variable, the discrete set of states (or values)
            that the variable can take. If unspecified, the observed values in the data set
            are taken to be the only possible states.

        complete_samples_only: bool (optional, default `True`)
            Specifies how to deal with missing data, if present. If set to `True` all rows
            that contain `np.Nan` somewhere are ignored. If `False` then, for each variable,
            every row where neither the variable nor its parents are `np.NaN` is used.
            This sets the behavior of the `state_count`-method.
        """
        continuous = False
        if np.all(data.dtypes == 'float64'):
            continuous = True

        if continuous and scoring_method not in [GaussianBicScore]:
            raise TypeError("Selected scoring_method {} incorrect for continuous data.".format(scoring_method))
        if not continuous and scoring_method not in [BdeuScore, BicScore, K2Score]:
            raise TypeError("Selected scoring_method {} incorrect for discrete data".format(scoring_method))

        if scoring_method is not None:
            self.scoring_method = scoring_method
        else:
            if continuous:
                self.scoring_method = GaussianBicScore
            else:
                self.scoring_method = K2Score(data, **kwargs)

        self.blacklist = set(blacklist)
        self.whitelist = set(whitelist)

        super(HillClimbSearch, self).__init__(data, **kwargs)


    def _legal_operations(self, model, tabu_list=[], max_indegree=None):
        """Generates a list of legal (= not in tabu_list) graph modifications
        for a given model, together with their score changes. Possible graph modifications:
        (1) add, (2) remove, or (3) flip a single edge. For details on scoring
        see Koller & Fridman, Probabilistic Graphical Models, Section 18.4.3.3 (page 818).
        If a number `max_indegree` is provided, only modifications that keep the number
        of parents for each node below `max_indegree` are considered."""

        local_score = self.scoring_method.local_score
        nodes = self.state_names.keys()
        potential_new_edges = (
            set(permutations(nodes, 2))
            - set(model.edges())
            - set([(Y, X) for (X, Y) in model.edges()])
        )

        for (X, Y) in potential_new_edges:  # (1) add single edge
            if nx.is_directed_acyclic_graph(nx.DiGraph(list(model.edges()) + [(X, Y)])):
                operation = ("+", (X, Y))
                if operation not in tabu_list:
                    old_parents = model.get_parents(Y)
                    new_parents = old_parents + [X]
                    if max_indegree is None or len(new_parents) <= max_indegree:
                        score_delta = local_score(Y, new_parents) - local_score(
                            Y, old_parents
                        )
                        yield (operation, score_delta)

        for (X, Y) in model.edges():  # (2) remove single edge
            operation = ("-", (X, Y))
            if operation not in tabu_list:
                old_parents = model.get_parents(Y)
                new_parents = old_parents[:]
                new_parents.remove(X)
                score_delta = local_score(Y, new_parents) - local_score(Y, old_parents)
                yield (operation, score_delta)

        for (X, Y) in model.edges():  # (3) flip single edge
            new_edges = list(model.edges()) + [(Y, X)]
            new_edges.remove((X, Y))
            if nx.is_directed_acyclic_graph(nx.DiGraph(new_edges)):
                operation = ("flip", (X, Y))
                if operation not in tabu_list and ("flip", (Y, X)) not in tabu_list:
                    old_X_parents = model.get_parents(X)
                    old_Y_parents = model.get_parents(Y)
                    new_X_parents = old_X_parents + [Y]
                    new_Y_parents = old_Y_parents[:]
                    new_Y_parents.remove(X)
                    if max_indegree is None or len(new_X_parents) <= max_indegree:
                        score_delta = (
                            local_score(X, new_X_parents)
                            + local_score(Y, new_Y_parents)
                            - local_score(X, old_X_parents)
                            - local_score(Y, old_Y_parents)
                        )
                        yield (operation, score_delta)

    def estimate(
        self, start=None, tabu_length=0, max_indegree=None, epsilon=1e-4, max_iter=1e6
    ):
        """
        Performs local hill climb search to estimates the `DAG` structure
        that has optimal score, according to the scoring method supplied in the constructor.
        Starts at model `start` and proceeds by step-by-step network modifications
        until a local maximum is reached. Only estimates network structure, no parametrization.

        Parameters
        ----------
        start: DAG instance
            The starting point for the local search. By default a completely disconnected network is used.

        tabu_length: int
            If provided, the last `tabu_length` graph modifications cannot be reversed
            during the search procedure. This serves to enforce a wider exploration
            of the search space. Default value: 100.

        max_indegree: int or None
            If provided and unequal None, the procedure only searches among models
            where all nodes have at most `max_indegree` parents. Defaults to None.

        epsilon: float (default: 1e-4)
            Defines the exit condition. If the improvement in score is less than `epsilon`,
            the learned model is returned.

        max_iter: int (default: 1e6)
            The maximum number of iterations allowed. Returns the learned model when the
            number of iterations is greater than `max_iter`.

        Returns
        -------
        model: `DAG` instance
            A `DAG` at a (local) score maximum.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pgmpy.estimators import HillClimbSearch, BicScore
        >>> # create data sample with 9 random variables:
        ... data = pd.DataFrame(np.random.randint(0, 5, size=(5000, 9)), columns=list('ABCDEFGHI'))
        >>> # add 10th dependent variable
        ... data['J'] = data['A'] * data['B']
        >>> est = HillClimbSearch(data, scoring_method=BicScore(data))
        >>> best_model = est.estimate()
        >>> sorted(best_model.nodes())
        ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        >>> best_model.edges()
        [('B', 'J'), ('A', 'J')]
        >>> # search a model with restriction on the number of parents:
        >>> est.estimate(max_indegree=1).edges()
        [('J', 'A'), ('B', 'J')]
        """
        nodes = self.state_names.keys()
        if start is None:
            start = DAG()
            start.add_nodes_from(nodes)
        elif not isinstance(start, DAG) or not set(start.nodes()) == set(nodes):
            raise ValueError(
                "'start' should be a DAG with the same variables as the data set, or 'None'."
            )

        tabu_list = []
        current_model = start

        iter_no = 0
        while iter_no <= max_iter:
            iter_no += 1

            best_score_delta = 0
            best_operation = None

            # FIXME: This implementation does not cache the delta scores for the changes!!!!!!!!
            for operation, score_delta in self._legal_operations(
                current_model, tabu_list, max_indegree
            ):
                if score_delta > best_score_delta:
                    best_operation = operation
                    best_score_delta = score_delta

            if best_operation is None or best_score_delta < epsilon:
                break
            elif best_operation[0] == "+":
                current_model.add_edge(*best_operation[1])
                tabu_list = ([("-", best_operation[1])] + tabu_list)[:tabu_length]
            elif best_operation[0] == "-":
                current_model.remove_edge(*best_operation[1])
                tabu_list = ([("+", best_operation[1])] + tabu_list)[:tabu_length]
            elif best_operation[0] == "flip":
                X, Y = best_operation[1]
                current_model.remove_edge(X, Y)
                current_model.add_edge(Y, X)
                tabu_list = ([best_operation] + tabu_list)[:tabu_length]

        return current_model


from collections import Counter

class CacheDecomposableScore(object):

    def __init__(self, graph, local_score, blacklist, whitelist):
        self.nodes = list(graph.nodes)
        self.nodes_indices = {var: index for index, var in enumerate(self.nodes)}
        self.blacklist = set(blacklist)
        self.whitelist = set(whitelist)
        self.graph = graph

        nnodes = self.graph.number_of_nodes()

        constraints_matrix = np.eye(nnodes)
        for (source, dest) in self.blacklist:
            s = self.nodes_indices[source]
            d = self.nodes_indices[dest]
            constraints_matrix[s, d] = 1

        for (source, dest) in self.whitelist:
            s = self.nodes_indices[source]
            d = self.nodes_indices[dest]
            constraints_matrix[s, d] = 1
            constraints_matrix[d, s] = 1

        self.reach_dict = self._node_reachability()

        scores = np.empty((nnodes, nnodes))
        constraints_matrix = self._add_acyclity_constraints(constraints_matrix)
        self.cache = np.ma.array(scores, mask=constraints_matrix)
        self._precompute_cache()
        self.local_score = local_score

    def _node_reachability(self):
        topo_list = list(nx.topological_sort(self.graph))
        print(self.graph.edges)
        reach_dict = {}
        for node in reversed(topo_list):
            reach_dict[node] = Counter(self.graph.successors(node))
            for child in self.graph.successors(node):
                reach_dict[node] = reach_dict[node] + (reach_dict[child])
        return reach_dict

    def _add_acyclity_constraints(self, constraints_matrix):
        for node in self.graph.nodes:
            for dest in self.reach_dict[node].keys() - self.graph.successors(node):
                constraints_matrix[self.nodes_indices[dest], self.nodes_indices[node]] = 1
        return constraints_matrix

    def _precompute_cache(self):
        for node in self.graph:
            parents = self.graph.get_parents(node)
            self._precompute_node(node, parents)

    def _precompute_node(self, node, parents):
        node_index = self.nodes_indices[node]
        mask = np.ma.getmask(self.cache)
        to_update = np.where(mask[:, node] == 0)[0]

        for index in to_update:
            updating_node = self.nodes[index]
            if updating_node in parents:
                # Delta score of removing arc 'updating_node' -> 'node'
                self.cache[index, node_index] = self.local_score(node, parents.remove(updating_node)) - \
                                                self.local_score(node, parents)
                # Delta score of reversing arc 'updating_node' -> 'node'
                if mask[node_index, index] == 0:
                    updating_node_parents = set(self.graph.predecessors(updating_node))
                    self.cache[node_index, index] = self.local_score(updating_node, updating_node_parents.add(node)) +\
                                                    self.local_score(node, parents.remove(updating_node)) -\
                                                    self.local_score(updating_node, updating_node_parents) -\
                                                    self.local_score(node, parents)
            else:
                # Delta score of adding arc 'updating_node' -> 'node'
                self.cache[index, node_index] = self.local_score(node, parents.add(updating_node)) - self.local_score(node, parents)

    def apply_operator(self, op):
        operation, source, dest = op

        if operation == "+":
            self.graph.add_edge(source, dest)
            # self.update_node_add(source, self.graph.predecessors(source))
        elif operation == "-":
            self.graph.remove_edge(source, dest)
            # self.update_node_remove(source, self.graph.predecessors(sour))
        else:
            self.graph.remove_edge(source, dest)
            self.graph.add_edge(dest, source)
            # self.update_nodes_flip(source, dest)
            # self._precompute_node(source, self.graph.predecessors())
            # self._precompute_node(dest, self.graph.predecessors())

    def best_operator(self):
        nnodes = self.graph.number_of_nodes()
        (source, dest) = np.unravel_index(self.cache.argmax(), (nnodes, nnodes))
        delta_score = self.cache[source, dest]
        node_source = self.nodes[source]
        node_dest = self.nodes[dest]
        if self.graph.has_edge(node_source, node_dest):
            return ("-", node_source, node_dest, delta_score)
        elif self.graph.has_edge(node_dest, node_source):
            return ("flip", node_dest, node_source, delta_score)
        else:
            return ("+", node_source, node_dest, delta_score)