#!/usr/bin/env python

import numpy as np
import networkx as nx

from pgmpy.estimators import StructureEstimator, K2Score, GaussianBicScore, BdeuScore, BicScore, BGeScore, PredictiveLikelihood
from pgmpy.base import DAG

from pgmpy.factors.continuous import NodeType

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

        if continuous and not isinstance(self.scoring_method, (GaussianBicScore, BGeScore, PredictiveLikelihood)):
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

    def _precompute_cache(self, graph, scores):
        """
        Precompute the scores for all the operations in the first iteration.
        :param graph: Starting graph to compute node.
        :param scores:  Matrix n x n of scores to be filled up.
        :return:
        """
        for node in graph:
            self._precompute_score_node(graph, scores, node)

    def _precompute_score_node(self, graph, scores, node):
        """
        Precompute the score for the node in the first iteration. This method is slightly different to update_node_score()
        because it is easier to deal with reversing scores at the start. In particular, when a change is produced in
        'node', then 'update_node_score' needs to update the score of reversing a possible arc 'other_node' -> 'node' or
        a possible arc 'node' -> 'other_node'.
        :param graph: Starting graph to compute the score.
        :param scores: Matrix n x n of scores to be filled up with the scores of operations in node 'node'.
        :param node: Node for which operation scores on it will be computed.
        :return:
        """
        parents = set(graph.get_parents(node))
        node_index = self.nodes_indices[node]
        to_update = np.where(self.constraints_matrix[:, node_index])[0]

        if isinstance(self.scoring_method, PredictiveLikelihood):
            local_score = lambda node, parents: self.scoring_method.local_score(node, parents,
                                                            NodeType.GAUSSIAN, {p: NodeType.GAUSSIAN for p in parents})
        else:
            local_score = self.scoring_method.local_score

        for other_index in to_update:
            other_node = self.nodes[other_index]
            if graph.has_edge(other_node, node):
                parents_new = parents.copy()
                parents_new.remove(other_node)
                # Delta score of removing arc 'other_node' -> 'node'
                scores[other_index, node_index] = local_score(node, parents_new) - \
                                                local_score(node, parents)
            # Delta score of reversing arc 'node' -> 'other_node'
            elif graph.has_edge(node, other_node):
                other_node_parents = set(graph.ancestors(other_node))

                parents_new = parents.copy()
                parents_new.add(other_node)

                other_node_parents_new = other_node_parents.copy()
                other_node_parents_new.remove(node)
                scores[other_index, node_index] = local_score(other_node, other_node_parents_new) +\
                                                local_score(node, parents_new) -\
                                                local_score(other_node, other_node_parents) -\
                                                local_score(node, parents)
            else:
                # Delta score of adding arc 'other_node' -> 'node'
                parents_new = parents.copy()
                parents_new.add(other_node)
                scores[other_index, node_index] = local_score(node, parents_new) - local_score(node, parents)

    def update_node_score(self, graph, scores, node):
        """
        Updates the relevant scores for a given node. When a change is produced, only scores related with dest_node
        need to be updated. This method updates only those scores (page 818 of Koller & Friedman, 2009). Take into
        account that whitelisted/blacklisted score movements need not to be updated. In particular:

        * if an arc i->j is blacklisted:
            - source[i,j] is never computed.
            - source[j,i] is computed.
        * if an arc i->j is whitelisted:
            - source[i,j] is not computed.
            - source[j,i] is not computed.

        :param graph: Graph of the current model.
        :param scores: Matrix n x n of scores to be updated.
        :param node: Node where there was a change in the graph that needs updating scores.
        :return:
        """
        parents = set(graph.get_parents(node))
        node_index = self.nodes_indices[node]
        to_update = np.where(self.constraints_matrix[:, node_index])[0]

        if isinstance(self.scoring_method, PredictiveLikelihood):
            local_score = lambda node, parents: self.scoring_method.local_score(node, parents,
                                                            NodeType.GAUSSIAN, {p: NodeType.GAUSSIAN for p in parents})
        else:
            local_score = self.scoring_method.local_score

        for other_index in to_update:
            other_node = self.nodes[other_index]

            if graph.has_edge(other_node, node):
                parents_new = parents.copy()
                parents_new.remove(other_node)
                # Delta score of removing arc 'other_node' -> 'node'
                scores[other_index, node_index] = local_score(node, parents_new) - \
                                                local_score(node, parents)

                # Delta score of reversing arc 'other_node' -> 'node'
                other_node_parents = set(graph.get_parents(other_node))
                other_node_parents_new = other_node_parents.copy()
                other_node_parents_new.add(node)

                scores[node_index, other_index] = local_score(other_node, other_node_parents_new) +\
                                                local_score(node, parents_new) -\
                                                local_score(other_node, other_node_parents) - \
                                                local_score(node, parents)

            # Delta score of reversing arc 'node' -> 'other_node'
            elif graph.has_edge(node, other_node):
                other_node_parents = set(graph.get_parents(other_node))

                parents_new = parents.copy()
                parents_new.add(other_node)

                other_node_parents_new = other_node_parents.copy()
                other_node_parents_new.remove(node)
                scores[other_index, node_index] = local_score(other_node, other_node_parents_new) +\
                                                local_score(node, parents_new) -\
                                                local_score(other_node, other_node_parents) -\
                                                local_score(node, parents)

            # Delta score of adding arc 'other_node' -> 'node'
            else:
                parents_new = parents.copy()
                parents_new.add(other_node)
                scores[other_index, node_index] = local_score(node, parents_new) - local_score(node, parents)

    def apply_operator(self, op, graph, scores):
        """
        Applies the operator 'op' to the graph. This implies updating the graph and the cached scores.
        :param op: Operation to apply (add, remove or reverse arcs).
        :param graph: Graph to update.
        :param scores: Matrix n x n of scores to update. It just updates the relevant scores given the operation.
        :return:
        """
        operation, source, dest, _ = op

        if operation == "+":
            graph.add_edge(source, dest)
            self.update_node_score(graph, scores, dest)
        elif operation == "-":
            graph.remove_edge(source, dest)
            self.update_node_score(graph, scores, dest)
        else:
            graph.remove_edge(source, dest)
            graph.add_edge(dest, source)
            # TODO FIXME: The local score for reversing the arc 'source' -> 'dest' is computed twice, once for each call to update_node_score().
            self.update_node_score(graph, scores, source)
            self.update_node_score(graph, scores, dest)

    def best_operator(self, graph, scores):
        """
        Finds the best operator to apply to the graph.
        :param graph: The current graph model.
        :param scores: A matrix of n x n where score[i,j] is the score of adding the arc i->j if the arc is not currently
        in the graph. If the arc i->j is currently in the graph, score[i,j] is the score of removing the, and
        score[j,i] is the score of reversing the arc.
        :return: The best operator (op, source_node, dest_node, delta_score).
        """
        nnodes = graph.number_of_nodes()

        # Sort in descending order. That is, [::-1].
        sort_scores = np.unravel_index(np.argsort(scores.ravel())[::-1], (nnodes, nnodes))

        for i in range(self.total_num_arcs):
            source_index = sort_scores[0][i]
            dest_index = sort_scores[1][i]
            delta_score = scores[source_index, dest_index]
            source_node = self.nodes[source_index]
            dest_node = self.nodes[dest_index]

            if graph.has_edge(source_node, dest_node):
                return ("-", source_node, dest_node, delta_score)
            elif graph.has_edge(dest_node, source_node):
                graph.remove_edge(dest_node, source_node)
                graph.add_edge(source_node, dest_node)
                must_check_for_cycle = False if not any(graph.get_parents(source_node)) or \
                                               not any(graph.get_children(dest_node)) else True

                if must_check_for_cycle:
                    isdag = nx.is_directed_acyclic_graph(graph)

                    graph.remove_edge(source_node, dest_node)
                    graph.add_edge(dest_node, source_node)

                    if isdag:
                        return ("flip", dest_node, source_node, delta_score)
                    else:
                        continue
                else:
                    graph.remove_edge(source_node, dest_node)
                    graph.add_edge(dest_node, source_node)
                    return ("flip", dest_node, source_node, delta_score)
            else:
                must_check_for_cycle = False if not any(graph.get_parents(source_node)) or \
                                               not any(graph.get_children(dest_node)) else True

                if must_check_for_cycle:
                    graph.add_edge(source_node, dest_node)
                    isdag = nx.is_directed_acyclic_graph(graph)

                    graph.remove_edge(source_node, dest_node)
                    if isdag:
                        return ("+", source_node, dest_node, delta_score)
                    else:
                        continue
                else:
                    return ("+", source_node, dest_node, delta_score)
        return None

    def best_operator_max_indegree(self, graph, scores, max_indegree):
        """
        Finds the best operator to apply to the graph. This is a version of self.best_operation() with checks for
        maximum indegree for maximum performance when indegree is not relevant.
        :param graph: The current graph model.
        :param scores: A matrix of n x n where score[i,j] is the score of adding the arc i->j if the arc is not currently
        in the graph. If the arc i->j is currently in the graph, score[i,j] is the score of removing the, and
        score[j,i] is the score of reversing the arc.
        :return: The best operator (op, source_node, dest_node, delta_score).
        """
        nnodes = graph.number_of_nodes()

        # Sort in descending order. That is, [::-1].
        sort_scores = np.unravel_index(np.argsort(scores.ravel())[::-1], (nnodes, nnodes))

        for i in range(self.total_num_arcs):
            source_index = sort_scores[0][i]
            dest_index = sort_scores[1][i]
            delta_score = scores[source_index, dest_index]
            source_node = self.nodes[source_index]
            dest_node = self.nodes[dest_index]

            if graph.has_edge(source_node, dest_node):
                return ("-", source_node, dest_node, delta_score)
            elif graph.has_edge(dest_node, source_node):
                if len(graph.get_parents(dest_node)) >= max_indegree:
                    continue

                graph.remove_edge(dest_node, source_node)
                graph.add_edge(source_node, dest_node)
                must_check_for_cycle = False if not any(graph.get_parents(source_node)) or \
                                               not any(graph.get_children(dest_node)) else True
                if must_check_for_cycle:
                    isdag = nx.is_directed_acyclic_graph(graph)
                    graph.remove_edge(source_node, dest_node)
                    graph.add_edge(dest_node, source_node)

                    if isdag:
                        return ("flip", dest_node, source_node, delta_score)
                    else:
                        continue
                else:
                    graph.remove_edge(source_node, dest_node)
                    graph.add_edge(dest_node, source_node)
                    return ("flip", dest_node, source_node, delta_score)
            else:
                if len(graph.get_parents(dest_node)) >= max_indegree:
                    continue

                must_check_for_cycle = False if not any(graph.get_parents(source_node)) or \
                                               not any(graph.get_children(dest_node)) else True

                if must_check_for_cycle:
                    graph.add_edge(source_node, dest_node)
                    isdag = nx.is_directed_acyclic_graph(graph)

                    graph.remove_edge(source_node, dest_node)
                    if isdag:
                        return ("+", source_node, dest_node, delta_score)
                    else:
                        continue
                else:
                    return ("+", source_node, dest_node, delta_score)
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
    def estimate(
        self, start=None, tabu_length=0, max_indegree=None, epsilon=1e-4, max_iter=1e6
    ):
        """
        This method runs the hill climbing algorithm.

        :param start: Starting graph structure. If None, it starts with an empty graph.
        :param tabu_length:
        :param max_indegree: Maximum indegree allowed for each node. If None, no maximum_indegree restriction is applied.
        :param epsilon: Minimum delta score necessary to continue iterating.
        :param max_iter: Maximum number of iterations to apply.
        :return: Best graph structure found by the algorithm.
        """
        if start is None:
            start = DAG()
            start.add_nodes_from(self.nodes)

        if max_indegree is None:
            best_operator_fun = self.best_operator
        else:
            best_operator_fun = lambda graph, scores: self.best_operator_max_indegree(graph, scores, max_indegree)


        self._check_blacklist(start)
        self.force_whitelist(start)

        nnodes = len(self.nodes)
        scores = np.empty((nnodes, nnodes))
        self._precompute_cache(start, scores)
        # Mark constraints with the lowest value.
        maximum_fill_value = np.ma.maximum_fill_value(scores)
        scores[~self.constraints_matrix] = maximum_fill_value

        current_model = start

        iter_no = 0
        # last_delta = np.finfo(np.float64).max
        print("Starting score: " + str(self._total_score(current_model)))
        while iter_no <= max_iter:
            iter_no += 1
            op = best_operator_fun(current_model, scores)

            if op is None:
                break

            delta_score = op[3]
            if delta_score < epsilon:
                break

            print("Best op: " + str(op))
            self.apply_operator(op, current_model, scores)
            self._draw(current_model, op, iter_no)
            print("Current score: " + str(self._total_score(current_model)))

        self._draw(current_model, None, iter_no)
        final_score = self._total_score(current_model)
        print("Final score: " + str(final_score))
        return current_model

    def _total_score(self, graph):
        """
        Computes the total score in the network. As the score method is decomposable. The total score is the sum of
        the local scores.
        :param graph: Graph to be evaluated.
        :return: Total score of the network.
        """
        total_score = 0
        for node in graph.nodes:
            parents = graph.get_parents(node)

            if isinstance(self.scoring_method, PredictiveLikelihood):
                local_score = lambda node, parents: self.scoring_method.local_score(node, parents,
                                                                                    NodeType.GAUSSIAN, {p: NodeType.GAUSSIAN for p in parents})
            else:
                local_score = self.scoring_method.local_score

            total_score += local_score(node, parents)

        return total_score

    def _draw(self, graph, best_op, iter):

        if isinstance(self.scoring_method, PredictiveLikelihood):
            local_score = lambda node, parents: self.scoring_method.local_score(node, parents,
                                                                                NodeType.GAUSSIAN, {p: NodeType.GAUSSIAN for p in parents})
        else:
            local_score = self.scoring_method.local_score

        total_score = 0
        for node in graph.nodes:
            parents = graph.get_parents(node)
            total_score += local_score(node, parents)

        if best_op is None:
            A = nx.nx_agraph.to_agraph(graph)
            A.graph_attr.update(label="Score {:0.3f}".format(total_score), labelloc="t", fontsize='25')
            A.write('iterations/{:03d}.dot'.format(iter))
            A.clear()
        else:
            operation, source, dest, score = best_op

            graph_copy = graph.copy()
            if operation == '+':
                graph_copy.edges[source, dest]['color'] = 'green3'
                graph_copy.edges[source, dest]['label'] = "{:0.3f}".format(score)
            elif operation == '-':
                graph_copy.add_edge(source, dest)
                graph_copy.edges[source, dest]['color'] = 'firebrick1'
                graph_copy.edges[source, dest]['label'] = "{:0.3f}".format(score)
            elif operation == 'flip':
                graph_copy.edges[dest, source]['color'] = 'dodgerblue'
                graph_copy.edges[dest, source]['label'] = "{:0.3f}".format(score)

            A = nx.nx_agraph.to_agraph(graph_copy)
            A.graph_attr.update(label="Score {:0.3f}".format(total_score), labelloc="t", fontsize='25')
            A.write('iterations/{:03d}.dot'.format(iter))
            A.clear()

        import subprocess
        subprocess.run(["dot", "-Tpdf", "iterations/{:03d}.dot".format(iter), "-o",
                        "iterations/{:03d}.pdf".format(iter)])

        import os
        os.remove('iterations/{:03d}.dot'.format(iter))