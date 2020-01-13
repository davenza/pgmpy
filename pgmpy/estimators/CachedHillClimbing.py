#!/usr/bin/env python

import numpy as np
import networkx as nx

from pgmpy.estimators import StructureEstimator, K2Score, GaussianBicScore, BdeuScore, BicScore, BGeScore, \
    GaussianValidationLikelihood
from pgmpy.models import BayesianModel


class CachedHillClimbing(StructureEstimator):

    def __init__(self, data, scoring_method=None, blacklist=None, whitelist=None, **kwargs):
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
        if blacklist is None:
            blacklist = []

        if whitelist is None:
            whitelist = []

        continuous = False
        if np.all(data.dtypes == 'float64'):
            continuous = True
        elif np.any(data.dtypes == 'float64'):
            raise ValueError("CachedHillClimbing supports only-continuous or only-discrete data.")

        if scoring_method is None:
            if continuous:
                self.scoring_method = GaussianBicScore(data, **kwargs)
            else:
                self.scoring_method = K2Score(data, **kwargs)
        else:
            self.scoring_method = scoring_method

        if continuous and not isinstance(self.scoring_method,
                                         (GaussianBicScore, BGeScore, GaussianValidationLikelihood)):
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

        self.node_scores = np.empty((nnodes,))
        self.total_num_arcs = self.constraints_matrix.sum()

        super(CachedHillClimbing, self).__init__(data, **kwargs)

    def _precompute_cache_node_scores(self, model):
        local_score = self.scoring_method.local_score

        for node in model:
            node_index = self.nodes_indices[node]
            parents = set(model.get_parents(node))
            self.node_scores[node_index] = local_score(node, parents)

    def _precompute_cache_arcs(self, model, scores):
        """
        Precompute the scores for all the operations in the first iteration.
        :param model: Starting graph to compute node.
        :param scores:  Matrix n x n of scores to be filled up.
        :return:
        """
        for node in model:
            self._precompute_score_arcs_node(model, scores, node)

    def _precompute_score_arcs_node(self, model, scores, node):
        """
        Precompute the score for the node in the first iteration. This method is slightly different to
        update_node_score() because it is easier to deal with reversing scores at the start. In particular, when a
        change is produced in 'node', then 'update_node_score' needs to update the score of reversing a possible arc
        'other_node' -> 'node' or a possible arc 'node' -> 'other_node'.
        :param model: Starting graph to compute the score.
        :param scores: Matrix n x n of scores to be filled up with the scores of operations in node 'node'.
        :param node: Node for which operation scores on it will be computed.
        :return:
        """
        parents = set(model.get_parents(node))
        node_index = self.nodes_indices[node]
        to_update = np.where(self.constraints_matrix[:, node_index])[0]

        local_score = self.scoring_method.local_score

        for other_index in to_update:
            other_node = self.nodes[other_index]
            if model.has_edge(other_node, node):
                parents_new = parents.copy()
                parents_new.remove(other_node)
                # Delta score of removing arc 'other_node' -> 'node'
                scores[other_index, node_index] = \
                    local_score(node, parents_new) - self.node_scores[node_index]
                print("Caching removing arc " + other_node + " -> " + node +
                      " (" + str(scores[other_index, node_index]) + ")")
            # Delta score of reversing arc 'node' -> 'other_node'
            elif model.has_edge(node, other_node):
                other_node_parents = set(model.get_parents(other_node))

                parents_new = parents.copy()
                parents_new.add(other_node)

                other_node_parents_new = other_node_parents.copy()
                other_node_parents_new.remove(node)
                scores[other_index, node_index] = \
                    local_score(other_node, other_node_parents_new) + \
                    local_score(node, parents_new) - \
                    self.node_scores[other_index] - \
                    self.node_scores[node_index]
                print("Caching reversing arc " + node + " -> " + other_node +
                      " (" + str(scores[other_index, node_index]) + ")")
            else:
                # Delta score of adding arc 'other_node' -> 'node'
                parents_new = parents.copy()
                parents_new.add(other_node)
                scores[other_index, node_index] = local_score(node, parents_new) - self.node_scores[node_index]
                print("Caching adding arc " + other_node + " -> " + node +
                      " (" + str(scores[other_index, node_index]) + ")")

    def score_add_arc(self, model, source, dest):
        local_score = self.scoring_method.local_score

        dest_index = self.nodes_indices[dest]

        parents = set(model.get_parents(dest))
        parents_new = parents.copy()
        parents_new.add(source)

        return local_score(dest, parents_new) - self.node_scores[dest_index]

    def score_remove_arc(self, model, source, dest):
        local_score = self.scoring_method.local_score

        dest_index = self.nodes_indices[dest]

        parents = set(model.get_parents(dest))
        parents_new = parents.copy()
        parents_new.remove(source)

        return local_score(dest, parents_new) - self.node_scores[dest_index]

    def score_flip_arc(self, model, source, dest):
        return self.score_remove_arc(model, source, dest) + self.score_add_arc(model, dest, source)

    def update_node_score(self, model, node_set):
        local_score = self.scoring_method.local_score
        for n in node_set:
            parents = model.get_parents(n)
            n_index = self.nodes_indices[n]
            self.node_scores[n_index] = local_score(n, parents)

    def update_arc_scores(self, model, scores, arc_set):
        for (source, dest) in arc_set:
            source_index = self.nodes_indices[source]
            dest_index = self.nodes_indices[dest]

            if model.has_edge(source, dest):
                scores[source_index, dest_index] = self.score_remove_arc(model, source, dest)
            elif model.has_edge(dest, source):
                scores[source_index, dest_index] = self.score_flip_arc(model, dest, source)
            else:
                scores[source_index, dest_index] = self.score_add_arc(model, source, dest)

    def arcset_to_node(self, model, dest):
        dest_index = self.nodes_indices[dest]

        to_update = np.where(self.constraints_matrix[:, dest_index])[0]

        update_set = set()
        for other_index in to_update:
            other_node = self.nodes[other_index]

            if model.has_edge(other_node, dest):
                # Delta score of removing arc 'other_node' -> 'dest'
                update_set.add((other_node, dest))
                # Delta score of reversing arc 'other_node' -> 'dest'
                update_set.add((dest, other_node))
            # Delta score of reversing arc 'dest' -> 'other_node'
            elif model.has_edge(dest, other_node):
                update_set.add((other_node, dest))
            # Delta score of adding arc 'other_node' -> 'dest'
            else:
                update_set.add((other_node, dest))

        return update_set

    def apply_operator(self, op, model, scores):
        """
        Applies the operator 'op' to the graph. This implies updating the graph and the cached scores.
        :param op: Operation to apply (add, remove or reverse arcs).
        :param model: Graph to update.
        :param scores: Matrix n x n of scores to update. It just updates the relevant scores given the operation.
        :return:
        """
        operation, source, dest, _ = op

        to_update_nodes = set()
        to_update_arcs = set()
        if operation == "+":
            model.add_edge(source, dest)

            to_update_nodes.add(dest)
            to_update_arcs.update(self.arcset_to_node(model, dest))
        elif operation == "-":
            model.remove_edge(source, dest)

            to_update_nodes.add(dest)
            to_update_arcs.update(self.arcset_to_node(model, dest))
            to_update_arcs.add((dest, source))
        else:
            model.remove_edge(source, dest)
            model.add_edge(dest, source)

            to_update_nodes.add(source)
            to_update_nodes.add(dest)
            to_update_arcs.update(self.arcset_to_node(model, source))
            to_update_arcs.update(self.arcset_to_node(model, dest))

        self.update_node_score(model, to_update_nodes)
        self.update_arc_scores(model, scores, to_update_arcs)

    def best_operator(self, model, scores, epsilon):
        """
        Finds the best operator to apply to the graph.

        :param model: The current graph model.
        :param scores: A matrix of n x n where score[i,j] is the score of adding the arc i->j if the arc is not
        currently in the graph. If the arc i->j is currently in the graph, score[i,j] is the score of removing the arc
        and score[j,i] is the score of reversing the arc.
        :param epsilon: Minimum delta score.
        :return: The best operator (op, source_node, dest_node, delta_score).
        """
        nnodes = model.number_of_nodes()

        # Sort in descending order. That is, [::-1].
        sort_scores = np.unravel_index(np.argsort(scores.ravel())[::-1], (nnodes, nnodes))

        for i in range(self.total_num_arcs):
            source_index = sort_scores[0][i]
            dest_index = sort_scores[1][i]
            delta_score = scores[source_index, dest_index]

            if delta_score < epsilon:
                return None

            source_node = self.nodes[source_index]
            dest_node = self.nodes[dest_index]

            if model.has_edge(source_node, dest_node):
                return "-", source_node, dest_node, delta_score
            elif model.has_edge(dest_node, source_node):
                source_new_parents = model.get_parents(source_node)
                source_new_parents.remove(dest_node)
                dest_new_children = model.get_children(dest_node)
                dest_new_children.remove(source_node)

                must_check_for_cycle = False if not source_new_parents or not dest_new_children else True
                if must_check_for_cycle:
                    try:
                        model.remove_edge(dest_node, source_node)
                        model.add_edge(source_node, dest_node)
                        isdag = True
                        model.remove_edge(source_node, dest_node)
                        model.add_edge(dest_node, source_node)
                    except ValueError:
                        isdag = False
                        model.add_edge(dest_node, source_node)

                    if isdag:
                        return "flip", dest_node, source_node, delta_score
                    else:
                        continue
                else:
                    return "flip", dest_node, source_node, delta_score
            else:
                must_check_for_cycle = \
                    False if not model.get_parents(source_node) or not model.get_children(dest_node) else True

                if must_check_for_cycle:
                    try:
                        model.add_edge(source_node, dest_node)
                        isdag = True
                        model.remove_edge(source_node, dest_node)
                    except ValueError:
                        isdag = False

                    if isdag:
                        return "+", source_node, dest_node, delta_score
                    else:
                        continue
                else:
                    return "+", source_node, dest_node, delta_score
        return None

    def best_operator_validation(self, model, scores, epsilon, tabu):
        """
        Finds the best operator to apply to the graph.

        :param model: The current graph model.
        :param scores: A matrix of n x n where score[i,j] is the score of adding the arc i->j if the arc is not
        currently in the graph. If the arc i->j is currently in the graph, score[i,j] is the score of removing the arc
        and score[j,i] is the score of reversing the arc.
        :param epsilon: Minimum delta score.
        :param tabu: Set of forbidden operators.
        :return: The best operator (op, source_node, dest_node, delta_score).
        """
        nnodes = model.number_of_nodes()

        # Sort in descending order. That is, [::-1].
        sort_scores = np.unravel_index(np.argsort(scores.ravel())[::-1], (nnodes, nnodes))

        for i in range(self.total_num_arcs):
            source_index = sort_scores[0][i]
            dest_index = sort_scores[1][i]
            delta_score = scores[source_index, dest_index]

            if delta_score < epsilon:
                return None

            source_node = self.nodes[source_index]
            dest_node = self.nodes[dest_index]

            if model.has_edge(source_node, dest_node):
                if ("-", source_node, dest_node) not in tabu:
                    return "-", source_node, dest_node, delta_score
                else:
                    continue
            elif model.has_edge(dest_node, source_node):
                source_new_parents = model.get_parents(source_node)
                source_new_parents.remove(dest_node)
                dest_new_children = model.get_children(dest_node)
                dest_new_children.remove(source_node)

                must_check_for_cycle = False if not source_new_parents or not dest_new_children else True
                if must_check_for_cycle:
                    try:
                        model.remove_edge(dest_node, source_node)
                        model.add_edge(source_node, dest_node)
                        isdag = True
                        model.remove_edge(source_node, dest_node)
                        model.add_edge(dest_node, source_node)
                    except ValueError:
                        isdag = False
                        model.add_edge(dest_node, source_node)

                    if isdag:
                        if ("flip", dest_node, source_node) not in tabu:
                            return "flip", dest_node, source_node, delta_score
                        else:
                            continue
                    else:
                        continue
                else:
                    if ("flip", dest_node, source_node) not in tabu:
                        return "flip", dest_node, source_node, delta_score
                    else:
                        continue
            else:
                must_check_for_cycle = \
                    False if not model.get_parents(source_node) or not model.get_children(dest_node) else True

                if must_check_for_cycle:
                    try:
                        model.add_edge(source_node, dest_node)
                        isdag = True
                        model.remove_edge(source_node, dest_node)
                    except ValueError:
                        isdag = False

                    if isdag:
                        if ("+", source_node, dest_node) not in tabu:
                            return "+", source_node, dest_node, delta_score
                        else:
                            continue
                    else:
                        continue
                else:
                    if ("+", source_node, dest_node) not in tabu:
                        return "+", source_node, dest_node, delta_score
                    else:
                        continue
        return None

    def best_operator_max_indegree(self, graph, scores, epsilon, max_indegree):
        """
        Finds the best operator to apply to the graph. This is a version of self.best_operation() with checks for
        maximum indegree for maximum performance when indegree is not relevant.
        :param graph: The current graph model.
        :param scores: A matrix of n x n where score[i,j] is the score of adding the arc i->j if the arc is not
        currently in the graph. If the arc i->j is currently in the graph, score[i,j] is the score of removing the arc
        and score[j,i] is the score of reversing the arc.
        :param epsilon: Minimum delta score.
        :param max_indegree: Maximum indegree allowed.
        :return: The best operator (op, source_node, dest_node, delta_score).
        """
        nnodes = graph.number_of_nodes()

        # Sort in descending order. That is, [::-1].
        sort_scores = np.unravel_index(np.argsort(scores.ravel())[::-1], (nnodes, nnodes))

        for i in range(self.total_num_arcs):
            source_index = sort_scores[0][i]
            dest_index = sort_scores[1][i]
            delta_score = scores[source_index, dest_index]

            if delta_score < epsilon:
                return None

            source_node = self.nodes[source_index]
            dest_node = self.nodes[dest_index]

            if graph.has_edge(source_node, dest_node):
                return "-", source_node, dest_node, delta_score
            elif graph.has_edge(dest_node, source_node):
                if len(graph.get_parents(dest_node)) >= max_indegree:
                    continue

                graph.remove_edge(dest_node, source_node)
                graph.add_edge(source_node, dest_node)
                must_check_for_cycle = \
                    False if not any(graph.get_parents(source_node)) or not any(graph.get_children(dest_node)) else True
                if must_check_for_cycle:
                    isdag = nx.is_directed_acyclic_graph(graph)
                    graph.remove_edge(source_node, dest_node)
                    graph.add_edge(dest_node, source_node)

                    if isdag:
                        return "flip", dest_node, source_node, delta_score
                    else:
                        continue
                else:
                    graph.remove_edge(source_node, dest_node)
                    graph.add_edge(dest_node, source_node)
                    return "flip", dest_node, source_node, delta_score
            else:
                if len(graph.get_parents(dest_node)) >= max_indegree:
                    continue

                must_check_for_cycle = \
                    False if not any(graph.get_parents(source_node)) or not any(graph.get_children(dest_node)) else True

                if must_check_for_cycle:
                    graph.add_edge(source_node, dest_node)
                    isdag = nx.is_directed_acyclic_graph(graph)

                    graph.remove_edge(source_node, dest_node)
                    if isdag:
                        return "+", source_node, dest_node, delta_score
                    else:
                        continue
                else:
                    return "+", source_node, dest_node, delta_score
        return None

    def best_operator_max_indegree_validation(self, model, scores, epsilon, max_indegree, tabu):
        """
        Finds the best operator to apply to the graph.
        :param model: The current graph model.
        :param scores: A matrix of n x n where score[i,j] is the score of adding the arc i->j if the arc is not
        currently in the graph. If the arc i->j is currently in the graph, score[i,j] is the score of removing the arc
        and score[j,i] is the score of reversing the arc.
        :param epsilon: Minimum delta score.
        :param max_indegree: Maximum indegree allowed.
        :param tabu: Set of forbidden operators.
        :return: The best operator (op, source_node, dest_node, delta_score).
        """
        nnodes = model.number_of_nodes()

        # Sort in descending order. That is, [::-1].
        sort_scores = np.unravel_index(np.argsort(scores.ravel())[::-1], (nnodes, nnodes))

        for i in range(self.total_num_arcs):
            source_index = sort_scores[0][i]
            dest_index = sort_scores[1][i]
            delta_score = scores[source_index, dest_index]

            if delta_score < epsilon:
                return None

            source_node = self.nodes[source_index]
            dest_node = self.nodes[dest_index]

            if model.has_edge(source_node, dest_node):
                if ("-", source_node, dest_node) not in tabu:
                    return "-", source_node, dest_node, delta_score
                else:
                    continue
            elif model.has_edge(dest_node, source_node):
                if len(model.get_parents(dest_node)) >= max_indegree:
                    continue

                source_new_parents = model.get_parents(source_node)
                source_new_parents.remove(dest_node)
                dest_new_children = model.get_children(dest_node)
                dest_new_children.remove(source_node)

                must_check_for_cycle = False if not source_new_parents or not dest_new_children else True
                if must_check_for_cycle:
                    try:
                        model.remove_edge(dest_node, source_node)
                        model.add_edge(source_node, dest_node)
                        isdag = True
                        model.remove_edge(source_node, dest_node)
                        model.add_edge(dest_node, source_node)
                    except ValueError:
                        isdag = False
                        model.add_edge(dest_node, source_node)

                    if isdag:
                        if ("flip", dest_node, source_node) not in tabu:
                            return "flip", dest_node, source_node, delta_score
                        else:
                            continue
                    else:
                        continue
                else:
                    if ("flip", dest_node, source_node) not in tabu:
                        return "flip", dest_node, source_node, delta_score
                    else:
                        continue
            else:
                if len(model.get_parents(dest_node)) >= max_indegree:
                    continue

                must_check_for_cycle = \
                    False if not model.get_parents(source_node) or not model.get_children(dest_node) else True

                if must_check_for_cycle:
                    try:
                        model.add_edge(source_node, dest_node)
                        isdag = True
                        model.remove_edge(source_node, dest_node)
                    except ValueError:
                        isdag = False

                    if isdag:
                        if ("+", source_node, dest_node) not in tabu:
                            return "+", source_node, dest_node, delta_score
                        else:
                            continue
                    else:
                        continue
                else:
                    if ("+", source_node, dest_node) not in tabu:
                        return "+", source_node, dest_node, delta_score
                    else:
                        continue
        return None

    def _check_blacklist(self, graph):
        """
        Checks that blacklisted arcs are not included in the starting graph.
        :param graph: Starting graph.
        :return:
        """
        for edge in graph.edges:
            if edge in self.blacklist:
                raise ValueError("Blacklisted arc included in the starting graph.")

    def force_whitelist(self, graph):
        """
        Includes whitelisted arcs in the graph if they are not included.
        :param graph: Starting graph.
        :return:
        """
        for edge in self.whitelist:
            graph.add_edge(edge[0], edge[1])

        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("Whitelisted arcs generates cycles in the starting graph.")

    # FIXME: Implement tabu.
    def estimate_parametric(
        self, start=None, tabu_length=0, max_indegree=None, epsilon=1e-4, max_iter=1e6, callbacks=None
    ):
        """
        This method runs the hill climbing algorithm.

        :param start: Starting graph structure. If None, it starts with an empty graph.
        :param tabu_length:
        :param max_indegree: Maximum indegree allowed for each node. If None, no maximum_indegree restriction is
        applied.
        :param epsilon: Minimum delta score necessary to continue iterating.
        :param max_iter: Maximum number of iterations to apply.
        :param callbacks. A list of callbacks to be executed at each iteration of the algorithm.
        :return: Best graph structure found by the algorithm.
        """
        if start is None:
            start = BayesianModel()
            start.add_nodes_from(self.nodes)

        if callbacks is None:
            callbacks = []

        if max_indegree is None:
            best_operator_fun = lambda g, s: self.best_operator(g, s, epsilon)
        else:
            best_operator_fun = lambda g, s: self.best_operator_max_indegree(g, s, epsilon, max_indegree)

        self._check_blacklist(start)
        self.force_whitelist(start)

        nnodes = len(self.nodes)
        scores = np.empty((nnodes, nnodes))
        self._precompute_cache_node_scores(start)
        self._precompute_cache_arcs(start, scores)
        # Mark constraints with the lowest value.
        maximum_fill_value = np.ma.maximum_fill_value(scores)
        scores[~self.constraints_matrix] = maximum_fill_value

        current_model = start

        iter_no = 0
        # last_delta = np.finfo(np.float64).max

        current_score = self._total_score()
        print("Starting score: " + str(current_score) + " (" + str(current_score / len(self.data)) + " / instance)")

        for callback in callbacks:
            callback.call(current_model, None, self.scoring_method, iter_no)

        while iter_no <= max_iter:
            iter_no += 1
            op = best_operator_fun(current_model, scores)

            if op is None:
                break

            print("Best op: " + str(op))
            self.apply_operator(op, current_model, scores)

            new_score = self._total_score()

            if not np.isclose(new_score, current_score + op[3]):
                print("Error on scores")
                input()

            for callback in callbacks:
                callback.call(current_model, op, self.scoring_method, iter_no)

            print("Current score: " + str(new_score) + " (" + str(new_score / len(self.data)) + " / instance)")
            current_score = new_score

        for callback in callbacks:
            callback.call(current_model, None, self.scoring_method, iter_no)

        final_score = self._total_score()
        print("Final score: " + str(final_score) + " (" + str(final_score / len(self.data)) + " / instance)")
        return current_model

        # FIXME: Implement tabu.
    def estimate_validation(
            self, start=None, tabu_length=0, max_indegree=None, epsilon=1e-4, max_iter=1e6, patience=0, callbacks=None
    ):
        """
        This method runs the hill climbing algorithm.

        :param start: Starting graph structure. If None, it starts with an empty graph.
        :param tabu_length:
        :param max_indegree: Maximum indegree allowed for each node. If None, no maximum_indegree restriction is
        applied.
        :param epsilon: Minimum delta score necessary to continue iterating.
        :param max_iter: Maximum number of iterations to apply.
        :param patience: Maximum number of iterations without improving the score.
        :param callbacks. A list of callbacks to be executed at each iteration of the algorithm.
        :return: Best graph structure found by the algorithm.
        """
        if not isinstance(self.scoring_method, GaussianValidationLikelihood):
            raise TypeError("estimate_validation() must be executed using a scoring method ValidationLikelihood.")

        if start is None:
            start = BayesianModel()
            start.add_nodes_from(self.nodes)

        if callbacks is None:
            callbacks = []

        tabu_last = set()

        if max_indegree is None:
            best_operator_fun = lambda g, s: self.best_operator_validation(g, s, epsilon, tabu_last)
        else:
            best_operator_fun = lambda g, s: self.best_operator_max_indegree_validation(g, s,
                                                                                        epsilon,
                                                                                        max_indegree,
                                                                                        tabu_last)

        self._check_blacklist(start)
        self.force_whitelist(start)

        nnodes = len(self.nodes)
        scores = np.empty((nnodes, nnodes))

        self._precompute_cache_node_scores(start)
        self._precompute_cache_arcs(start, scores)
        # Mark constraints with the lowest value.
        maximum_fill_value = np.ma.maximum_fill_value(scores)
        scores[~self.constraints_matrix] = maximum_fill_value

        current_model = start

        n_train_instances = len(self.scoring_method.data)
        n_validation_instances = len(self.scoring_method.validation_data)

        iter_no = 0
        iter_no_improvement = 0

        current_score = self._total_score()
        best_validation_score = self._total_validation_score(current_model)
        best_model_index = 0

        print("Starting score: " + str(current_score) + " (" + str(current_score / n_train_instances) + " / instance)")
        print("Validation score: " + str(best_validation_score) +
              " (" + str(best_validation_score / n_validation_instances) + " / instance)")

        models = [current_model.copy()]
        scores_history = [current_score]
        scores_validation_history = [best_validation_score]

        for callback in callbacks:
            callback.call(current_model, None, self.scoring_method, iter_no)

        while iter_no <= max_iter:
            iter_no += 1
            print("Iteration " + str(iter_no))
            print("----------------------")

            op = best_operator_fun(current_model, scores)

            if op is None:
                print("----------------------------")
                print("Best validation score: " + str(best_validation_score))
                print("----------------------------")
                current_model = models[best_model_index]
                self._precompute_cache_node_scores(current_model)
                break

            print("Best op: " + str(op))
            print()

            self.apply_operator(op, current_model, scores)

            new_score = self._total_score()
            new_validation_score = self._total_validation_score(current_model)

            models.append(current_model.copy())
            scores_history.append(new_score)
            scores_validation_history.append(new_validation_score)

            if best_validation_score > new_validation_score:
                iter_no_improvement += 1
                if iter_no_improvement > patience:
                    print("----------------------------")
                    print("Best validation score: " + str(best_validation_score))
                    print("New validation score: " + str(new_validation_score))
                    print("----------------------------")
                    current_model = models[best_model_index]
                    self._precompute_cache_node_scores(current_model)
                    break

                if op[0] == "+":
                    tabu_last.add(("-", op[1], op[2]))
                elif op[0] == "-":
                    tabu_last.add(("+", op[1], op[2]))
                elif op[0] == "flip":
                    tabu_last.add(("flip", op[2], op[1]))
            else:
                iter_no_improvement = 0
                best_model_index = iter_no
                best_validation_score = new_validation_score
                tabu_last.clear()

            if not np.isclose(new_score, current_score + op[3]):
                print("Error on scores")
                input()

            for callback in callbacks:
                callback.call(current_model, op, self.scoring_method, iter_no)

            print("Current score: " + str(new_score) + " (" + str(new_score / n_train_instances) + " / instance)")
            val_score = self._total_validation_score(current_model)
            print("Current validation score: " + str(val_score) +
                  " (" + str(val_score / n_validation_instances) + " / instance)")
            current_score = new_score

        for callback in callbacks:
            callback.call(current_model, None, self.scoring_method, iter_no)

        final_score = self._total_score()
        print("Final score: " + str(final_score) + " (" + str(final_score / n_train_instances) + " / instance)")
        val_score = self._total_validation_score(current_model)
        print("Final validation score: " + str(val_score) +
              " (" + str(val_score / n_validation_instances) + " / instance)")
        return current_model

    def estimate(self, start=None, tabu_length=0, max_indegree=None, epsilon=1e-4, max_iter=1e6,
                 patience=0, callbacks=None):
        if isinstance(self.scoring_method, (GaussianBicScore, BGeScore)):
            return self.estimate_parametric(start=start, tabu_length=tabu_length, max_indegree=max_indegree,
                                            epsilon=epsilon, max_iter=max_iter, callbacks=callbacks)
        elif isinstance(self.scoring_method, GaussianValidationLikelihood):
            return self.estimate_validation(start=start, tabu_length=tabu_length, max_indegree=max_indegree,
                                            epsilon=epsilon, max_iter=max_iter, patience=patience,
                                            callbacks=callbacks)

    def _total_score(self):
        """
        Computes the total score in the network. As the score method is decomposable. The total score is the sum of
        the local scores.
        :return: Total score of the network.
        """
        return self.node_scores.sum()

    def _total_validation_score(self, model):
        """
        Computes the total score in the network. As the score method is decomposable. The total score is the sum of
        the local scores.
        :param model: Graph to be evaluated.
        :return: Total score of the network.
        """
        total_score = 0
        for node in model.nodes:
            parents = model.get_parents(node)
            a = self.scoring_method.validation_local_score(node, parents)

            total_score += a

        return total_score
