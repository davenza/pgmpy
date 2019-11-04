#!/usr/bin/env python

import numpy as np
import networkx as nx

from pgmpy.estimators import StructureEstimator, PredictiveLikelihood
from pgmpy.base import DAG
from pgmpy.models import HybridContinuousModel
from pgmpy.factors.continuous import NodeType


class HybridCachedHillClimbing(StructureEstimator):

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
        if not np.all(data.dtypes == 'float64'):
            raise ValueError("HybridContinuousModel can only be trained from all-continuous data.")

        if scoring_method is None:
            self.scoring_method = PredictiveLikelihood(data)
        else:
            if not isinstance(scoring_method, PredictiveLikelihood):
                raise TypeError("HybridContinuousModel can only be trained with PredictiveLikelihood score.")

            self.scoring_method = scoring_method

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

        super(HybridCachedHillClimbing, self).__init__(data, **kwargs)

    def _precompute_cache_node_scores(self, model):
        local_score = self.scoring_method.local_score

        for node in model:
            node_index = self.nodes_indices[node]
            parents = set(model.get_parents(node))
            node_type = model.node_type[node]
            self.node_scores[node_index] = local_score(node, parents, node_type, model.node_type)

    def _precompute_cache_arcs(self, model, scores):
        """
        Precompute the scores for all the operations in the first iteration.
        :param model: Starting graph to compute node.
        :param scores:  Matrix n x n of scores to be filled up.
        :return:
        """
        for node in model:
            self._precompute_score_arcs_node(model, scores, node)

    def _precompute_cache_types(self, model, scores):
        for node in model:
            self._precompute_score_types_node(model, scores, node)

    def _precompute_score_arcs_node(self, model, scores, node):
        """
        Precompute the score for the node in the first iteration. This method is slightly different to update_node_score()
        because it is easier to deal with reversing scores at the start. In particular, when a change is produced in
        'node', then 'update_node_score' needs to update the score of reversing a possible arc 'other_node' -> 'node' or
        a possible arc 'node' -> 'other_node'.
        :param model: Starting graph to compute the score.
        :param scores: Matrix n x n of scores to be filled up with the scores of operations in node 'node'.
        :param node: Node for which operation scores on it will be computed.
        :return:
        """
        parents = set(model.get_parents(node))
        node_index = self.nodes_indices[node]
        to_update = np.where(self.constraints_matrix[:, node_index])[0]
        local_score = self.scoring_method.local_score

        node_type = model.node_type[node]

        for other_index in to_update:
            other_node = self.nodes[other_index]
            if model.has_edge(other_node, node):
                parents_new = parents.copy()
                parents_new.remove(other_node)
                # Delta score of removing arc 'other_node' -> 'node'
                scores[other_index, node_index] = local_score(node, parents_new, node_type, model.node_type) - \
                                                    self.node_scores[node_index]

                print("Caching removing arc " + other_node + " -> " + node + " (" + str(scores[other_index, node_index]) + ")")


                # str_p = ""
                # str_p_new = ""
                # for p in parents:
                #     str_p += "[" + p + "," + NodeType.str(model.node_type[p]) + "], "
                # for p in parents_new:
                #     str_p_new += "[" + p + "," + NodeType.str(model.node_type[p]) + "], "
                # print("\t P([" + node + "," + NodeType.str(model.node_type[node]) + "] | " + str_p_new + ") - " +
                #            "P([" + node + "," + NodeType.str(model.node_type[node]) + "] | " + str_p + ") = " +
                #       str(local_score(node, parents_new, node_type, model.node_type)) + " - " +
                #       str(local_score(node, parents, node_type, model.node_type))
                #       )
            # Delta score of reversing arc 'node' -> 'other_node'
            elif model.has_edge(node, other_node):
                other_node_parents = set(model.get_parents(other_node))

                parents_new = parents.copy()
                parents_new.add(other_node)

                other_node_parents_new = other_node_parents.copy()
                other_node_parents_new.remove(node)

                other_node_type = model.node_type[other_node]

                scores[node_index, other_index] = local_score(other_node, other_node_parents_new, other_node_type, model.node_type) +\
                                                local_score(node, parents_new, node_type, model.node_type) -\
                                                self.node_scores[other_index] -\
                                                self.node_scores[node_index]
                print("Caching reversing arc " + node + " -> " + other_node + " (" + str(scores[node_index, other_index]) + ")")

                # str_p = ""
                # str_other_p = ""
                # str_p_new = ""
                # str_other_p_new = ""
                # for p in parents:
                #     str_p += "[" + p + "," + NodeType.str(model.node_type[p]) + "], "
                # for p in parents_new:
                #     str_p_new += "[" + p + "," + NodeType.str(model.node_type[p]) + "], "
                # for p in other_node_parents:
                #     str_other_p += "[" + p + "," + NodeType.str(model.node_type[p]) + "], "
                # for p in other_node_parents_new:
                #     str_other_p_new += "[" + p + "," + NodeType.str(model.node_type[p]) + "], "
                #
                # print("\t P([" + other_node + "," + NodeType.str(model.node_type[other_node]) + "] | " + str_other_p_new + ") + " +
                #            "P([" + node + "," + NodeType.str(model.node_type[node]) + "] | " + str_p_new + ") - " +
                #         "P([" + other_node + "," + NodeType.str(model.node_type[other_node]) + "] | " + str_other_p + ") - " +
                #         "P([" + node + "," + NodeType.str(model.node_type[node]) + "] | " + str_p + ") = " +
                #       str(local_score(other_node, other_node_parents_new, node_type, model.node_type)) + " + " +
                #       str(local_score(node, parents_new, node_type, model.node_type)) + " - " +
                #       str(local_score(other_node, other_node_parents, node_type, model.node_type)) + " - " +
                #       str(local_score(node, parents, node_type, model.node_type))
                #       )

            else:
                # Delta score of adding arc 'other_node' -> 'node'
                parents_new = parents.copy()
                parents_new.add(other_node)
                scores[other_index, node_index] = local_score(node, parents_new, node_type, model.node_type) -\
                                                  self.node_scores[node_index]
                print("Caching adding arc " + other_node + " -> " + node + " (" + str(scores[other_index, node_index]) + ")")

                # str_p = ""
                # str_p_new = ""
                # for p in parents:
                #     str_p += "[" + p + "," + NodeType.str(model.node_type[p]) + "], "
                # for p in parents_new:
                #     str_p_new += "[" + p + "," + NodeType.str(model.node_type[p]) + "], "
                # print("\t P([" + node + "," + NodeType.str(model.node_type[node]) + "] | " + str_p_new + ") - " +
                #            "P([" + node + "," + NodeType.str(model.node_type[node]) + "] | " + str_p + ") = " +
                #       str(local_score(node, parents_new, node_type, model.node_type)) + " - " +
                #       str(local_score(node, parents, node_type, model.node_type))
                #       )

    def _precompute_score_types_node(self, model, scores, node):

        node_index = self.nodes_indices[node]
        local_score = self.scoring_method.local_score
        parents = set(model.get_parents(node))
        parents_type = model.node_type
        if model.node_type[node] == NodeType.GAUSSIAN:
            scores[node_index] = local_score(node, parents, NodeType.CKDE, parents_type) - \
                                 self.node_scores[node_index]
            print("Caching changing type of node " + node + " to CKDE (" + str(scores[node_index]) + ")")

            # str_p = ""
            # for p in parents:
            #     str_p += "[" + p + "," + NodeType.str(model.node_type[p]) + "], "
            #
            # print("\t P([" + node + "," + NodeType.str(NodeType.CKDE) + "] | " + str_p + ") - " +
            #       "P([" + node + "," + NodeType.str(NodeType.GAUSSIAN) + "] | " + str_p + ") = " +
            #       str(local_score(node, parents, NodeType.CKDE, parents_type)) + " - " +
            #       str(local_score(node, parents, NodeType.GAUSSIAN, parents_type))
            #       )
        elif model.node_type[node] == NodeType.CKDE:
            scores[node_index] = local_score(node, parents, NodeType.GAUSSIAN, parents_type) - \
                                 self.node_scores[node_index]
            print("Caching changing type of node " + node + " to Gaussian (" + str(scores[node_index]) + ")")

            # str_p = ""
            # for p in parents:
            #     str_p += "[" + p + "," + NodeType.str(model.node_type[p]) + "], "
            #
            # print("\t P([" + node + "," + NodeType.str(NodeType.GAUSSIAN) + "] | " + str_p + ") - " +
            #       "P([" + node + "," + NodeType.str(NodeType.CKDE) + "] | " + str_p + ") = " +
            #       str(local_score(node, parents, NodeType.GAUSSIAN, parents_type)) + " - " +
            #       str(local_score(node, parents, NodeType.CKDE, parents_type))
            #       )
        else:
            raise ValueError("Wrong node type for HybridContinuousModel.")

    def update_node_score_arcs(self, model, scores, node):
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

        :param model: Graph of the current model.
        :param scores: Matrix n x n of scores to be updated.
        :param node: Node where there was a change in the graph that needs updating scores.
        :return:
        """
        parents = set(model.get_parents(node))
        node_index = self.nodes_indices[node]
        to_update = np.where(self.constraints_matrix[:, node_index])[0]

        local_score = self.scoring_method.local_score

        node_type = model.node_type[node]

        for other_index in to_update:
            other_node = self.nodes[other_index]

            if model.has_edge(other_node, node):
                parents_new = parents.copy()
                parents_new.remove(other_node)

                # Delta score of removing arc 'other_node' -> 'node'
                scores[other_index, node_index] = local_score(node, parents_new, node_type, model.node_type) - \
                                                  self.node_scores[node_index]
                print("Updating removing arc " + other_node + " -> " + node + " (" + str(scores[other_index, node_index]) + ")")

                # str_p = ""
                # str_p_new = ""
                # for p in parents:
                #     str_p += "[" + p + "," + NodeType.str(model.node_type[p]) + "], "
                # for p in parents_new:
                #     str_p_new += "[" + p + "," + NodeType.str(model.node_type[p]) + "], "
                # print("\t P([" + node + "," + NodeType.str(model.node_type[node]) + "] | " + str_p_new + ") - " +
                #            "P([" + node + "," + NodeType.str(model.node_type[node]) + "] | " + str_p + ") = " +
                #       str(local_score(node, parents_new, node_type, model.node_type)) + " - " +
                #       str(local_score(node, parents, node_type, model.node_type))
                #       )

                # Delta score of reversing arc 'other_node' -> 'node'
                other_node_parents = set(model.get_parents(other_node))
                other_node_parents_new = other_node_parents.copy()
                other_node_parents_new.add(node)

                other_node_type = model.node_type[other_node]

                scores[node_index, other_index] = local_score(other_node, other_node_parents_new, other_node_type, model.node_type) +\
                                                local_score(node, parents_new, node_type, model.node_type) - \
                                                self.node_scores[other_index] - \
                                                self.node_scores[node_index]

                print("Updating reversing arc " + other_node + " -> " + node + " (" + str(scores[node_index, other_index]) + ")")

                # str_p = ""
                # str_other_p = ""
                # str_p_new = ""
                # str_other_p_new = ""
                # for p in parents:
                #     str_p += "[" + p + "," + NodeType.str(model.node_type[p]) + "], "
                # for p in parents_new:
                #     str_p_new += "[" + p + "," + NodeType.str(model.node_type[p]) + "], "
                # for p in other_node_parents:
                #     str_other_p += "[" + p + "," + NodeType.str(model.node_type[p]) + "], "
                # for p in other_node_parents_new:
                #     str_other_p_new += "[" + p + "," + NodeType.str(model.node_type[p]) + "], "
                #
                # print("\t P([" + other_node + "," + NodeType.str(model.node_type[other_node]) + "] | " + str_other_p_new + ") + " +
                #            "P([" + node + "," + NodeType.str(model.node_type[node]) + "] | " + str_p_new + ") - " +
                #         "P([" + other_node + "," + NodeType.str(model.node_type[other_node]) + "] | " + str_other_p + ") - " +
                #         "P([" + node + "," + NodeType.str(model.node_type[node]) + "] | " + str_p + ") = " +
                #       str(local_score(other_node, other_node_parents_new, other_node_type, model.node_type)) + " + " +
                #       str(local_score(node, parents_new, node_type, model.node_type)) + " - " +
                #       str(local_score(other_node, other_node_parents, other_node_type, model.node_type)) + " - " +
                #       str(local_score(node, parents, node_type, model.node_type))
                #       )

            # Delta score of reversing arc 'node' -> 'other_node'
            elif model.has_edge(node, other_node):
                other_node_parents = set(model.get_parents(other_node))

                parents_new = parents.copy()
                parents_new.add(other_node)

                other_node_parents_new = other_node_parents.copy()
                other_node_parents_new.remove(node)

                other_node_type = model.node_type[other_node]

                scores[other_index, node_index] = local_score(other_node, other_node_parents_new, other_node_type, model.node_type) +\
                                                local_score(node, parents_new, node_type, model.node_type) - \
                                                self.node_scores[other_index] - \
                                                self.node_scores[node_index]

                print("Updating reversing arc " + node + " -> " + other_node + " (" + str(scores[other_index, node_index]) + ")")

                # str_p = ""
                # str_other_p = ""
                # str_p_new = ""
                # str_other_p_new = ""
                # for p in parents:
                #     str_p += "[" + p + "," + NodeType.str(model.node_type[p]) + "], "
                # for p in parents_new:
                #     str_p_new += "[" + p + "," + NodeType.str(model.node_type[p]) + "], "
                # for p in other_node_parents:
                #     str_other_p += "[" + p + "," + NodeType.str(model.node_type[p]) + "], "
                # for p in other_node_parents_new:
                #     str_other_p_new += "[" + p + "," + NodeType.str(model.node_type[p]) + "], "
                #
                # print("\t P([" + other_node + "," + NodeType.str(model.node_type[other_node]) + "] | " + str_other_p_new + ") + " +
                #            "P([" + node + "," + NodeType.str(model.node_type[node]) + "] | " + str_p_new + ") - " +
                #         "P([" + other_node + "," + NodeType.str(model.node_type[other_node]) + "] | " + str_other_p + ") - " +
                #         "P([" + node + "," + NodeType.str(model.node_type[node]) + "] | " + str_p + ") = " +
                #       str(local_score(other_node, other_node_parents_new, other_node_type, model.node_type)) + " + " +
                #       str(local_score(node, parents_new, node_type, model.node_type)) + " - " +
                #       str(local_score(other_node, other_node_parents, other_node_type, model.node_type)) + " - " +
                #       str(local_score(node, parents, node_type, model.node_type))
                #       )

            # Delta score of adding arc 'other_node' -> 'node'
            else:
                parents_new = parents.copy()
                parents_new.add(other_node)
                scores[other_index, node_index] = local_score(node, parents_new, node_type, model.node_type) - \
                                                  self.node_scores[node_index]
                print("Updating adding arc " + other_node + " -> " + node + " (" + str(scores[other_index, node_index]) + ")")

                # str_p = ""
                # str_p_new = ""
                # for p in parents:
                #     str_p += "[" + p + "," + NodeType.str(model.node_type[p]) + "], "
                # for p in parents_new:
                #     str_p_new += "[" + p + "," + NodeType.str(model.node_type[p]) + "], "
                # print("\t P([" + node + "," + NodeType.str(model.node_type[node]) + "] | " + str_p_new + ") - " +
                #            "P([" + node + "," + NodeType.str(model.node_type[node]) + "] | " + str_p + ") = " +
                #       str(local_score(node, parents_new, node_type, model.node_type)) + " - " +
                #       str(local_score(node, parents, node_type, model.node_type))
                #       )

    def update_node_score_types(self, model, type_scores, node):
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

        :param model: Graph of the current model.
        :param scores: Matrix n x n of scores to be updated.
        :param node: Node where there was a change in the graph that needs updating scores.
        :return:
        """
        parents = set(model.get_parents(node))
        node_index = self.nodes_indices[node]
        local_score = self.scoring_method.local_score
        node_type = model.node_type[node]

        if node_type == NodeType.GAUSSIAN:
            other_node_type = NodeType.CKDE
            print("Updating changing type of node " + node + " to CKDE", end='')
        elif node_type == NodeType.CKDE:
            other_node_type = NodeType.GAUSSIAN
            print("Updating changing type of node " + node + " to Gaussian", end='')
        else:
            raise ValueError("Wrong node type detected on node {}.", node)

        type_scores[node_index] = local_score(node, parents, other_node_type, model.node_type) - \
                                  self.node_scores[node_index]

        children = model.get_children(node)

        for child in children:
            if model.node_type[child] == NodeType.CKDE:
                child_parents = set(model.get_parents(child))
                new_parent_type = model.node_type.copy()
                new_parent_type[node] = other_node_type
                child_index = self.nodes_indices[child]
                type_scores[node_index] += local_score(child, child_parents, model.node_type[child], new_parent_type) - \
                                           self.node_scores[child_index]


        # str_appendix_children = ""
        # str_appendix_numbers = ""
        # for child in children:
        #
        #     if model.node_type[child] == NodeType.CKDE:
        #         str_parents_child_new = ""
        #         str_parents_child_old = ""
        #         child_parents = model.get_parents(child)
        #         for p in child_parents:
        #             if p == node:
        #                 str_parents_child_new += "[" + p + "," + NodeType.str(other_node_type) + "], "
        #             str_parents_child_old += "[" + p + "," + NodeType.str(model.node_type[p]) + "], "
        #
        #
        #         str_appendix_children += "P([" + child + "," + NodeType.str(model.node_type[child]) + "] | " + str_parents_child_new + ") - " + \
        #                                  "P([" + child + "," + NodeType.str(model.node_type[child]) + "] | " + str_parents_child_old + ")"
        #
        #         new_parent_type = model.node_type.copy()
        #         new_parent_type[node] = other_node_type
        #
        #         str_appendix_numbers += str(local_score(child, child_parents, model.node_type[child], new_parent_type)) + " - " + \
        #                                 str(local_score(child, child_parents, model.node_type[child], model.node_type))
        #
        #
        # print(" (" + str(type_scores[node_index]) + ")")

        # str_p = ""
        # for p in parents:
        #     str_p += "[" + p +  "," + NodeType.str(model.node_type[p]) + "], "
        # print("\t P([" + node + "," + NodeType.str(other_node_type) + "] | " + str_p + ") - " +
        #       "P([" + node + "," + NodeType.str(node_type) + "] | " + str_p + ") " + str_appendix_children +
        #       " = " +
        #       str(local_score(node, parents, other_node_type, model.node_type)) + " - " +
        #       str(local_score(node, parents, node_type, model.node_type)) + " + " + str_appendix_numbers
        #       )

    def apply_operator(self, op, model, scores, type_scores):
        """
        Applies the operator 'op' to the graph. This implies updating the graph and the cached scores.
        :param op: Operation to apply (add, remove or reverse arcs).
        :param model: Graph to update.
        :param scores: Matrix n x n of scores to update. It just updates the relevant scores given the operation.
        :return:
        """
        operation, source, dest, _ = op

        local_score = self.scoring_method.local_score
        if operation == "+":
            model.add_edge(source, dest)

            dest_index = self.nodes_indices[dest]
            parents = model.get_parents(dest)
            self.node_scores[dest_index] = local_score(dest, parents, model.node_type[dest], model.node_type)

            self.update_node_score_arcs(model, scores, dest)
            self.update_node_score_types(model, type_scores, dest)
        elif operation == "-":
            model.remove_edge(source, dest)

            dest_index = self.nodes_indices[dest]
            parents = model.get_parents(dest)
            self.node_scores[dest_index] = local_score(dest, parents, model.node_type[dest], model.node_type)

            self.update_node_score_arcs(model, scores, dest)
            self.update_node_score_types(model, type_scores, dest)
        elif operation == "flip":
            model.remove_edge(source, dest)
            model.add_edge(dest, source)

            dest_index = self.nodes_indices[dest]
            parents = model.get_parents(dest)
            self.node_scores[dest_index] = local_score(dest, parents, model.node_type[dest], model.node_type)

            source_index = self.nodes_indices[source]
            parents = model.get_parents(source)
            self.node_scores[source_index] = local_score(source, parents, model.node_type[source], model.node_type)


            # TODO FIXME: The local score for reversing the arc 'source' -> 'dest' is computed twice, once for each call to update_node_score().
            self.update_node_score_arcs(model, scores, source)
            self.update_node_score_arcs(model, scores, dest)
            self.update_node_score_types(model, type_scores, dest)
            self.update_node_score_types(model, type_scores, source)
        elif operation == "type":
            model.node_type[source] = dest

            source_index = self.nodes_indices[source]
            parents = model.get_parents(source)
            self.node_scores[source_index] = local_score(source, parents, model.node_type[source], model.node_type)

            children = model.get_children(source)

            for child in children:
                if model.node_type[child] == NodeType.CKDE:
                    child_index = self.nodes_indices[dest]
                    parents = model.get_parents(dest)
                    self.node_scores[child_index] = local_score(child, parents, model.node_type[child_index], model.node_type)

            self.update_node_score_arcs(model, scores, source)
            self.update_node_score_types(model, type_scores, source)


    def best_operator(self, model, scores, type_scores):
        best_type = self.best_operator_types(model, type_scores)
        best_arc = self.best_operator_arcs(model, scores)

        delta_score_type = best_type[3]
        delta_score_arcs = best_arc[3]

        if delta_score_arcs is None or delta_score_type > delta_score_arcs:
            return best_type
        else:
            return best_arc

    def best_operator_max_indegree(self, model, scores, type_scores, max_indegree):
        best_type = self.best_operator_types(model, type_scores)
        best_arc = self.best_operator_arcs_max_indegree(model, scores, max_indegree)

        delta_score_type = best_type[3]
        delta_score_arcs = best_arc[3]

        if delta_score_arcs is None or delta_score_type > delta_score_arcs:
            return best_type
        else:
            return best_arc

    def best_operator_types(self, model, type_scores):
        index = type_scores.argmax()
        node = self.nodes[index]

        if model.node_type[node] == NodeType.GAUSSIAN:
            return ("type", node, NodeType.CKDE, type_scores[index])
        else:
            return ("type", node, NodeType.GAUSSIAN, type_scores[index])

    def best_operator_arcs(self, model, scores):
        """
        Finds the best operator to apply to the graph.
        :param model: The current graph model.
        :param scores: A matrix of n x n where score[i,j] is the score of adding the arc i->j if the arc is not currently
        in the graph. If the arc i->j is currently in the graph, score[i,j] is the score of removing the, and
        score[j,i] is the score of reversing the arc.
        :return: The best operator (op, source_node, dest_node, delta_score).
        """
        nnodes = model.number_of_nodes()

        # Sort in descending order. That is, [::-1].
        sort_scores = np.unravel_index(np.argsort(scores.ravel())[::-1], (nnodes, nnodes))

        for i in range(self.total_num_arcs):
            source_index = sort_scores[0][i]
            dest_index = sort_scores[1][i]
            delta_score = scores[source_index, dest_index]
            source_node = self.nodes[source_index]
            dest_node = self.nodes[dest_index]

            if model.has_edge(source_node, dest_node):
                return ("-", source_node, dest_node, delta_score)
            elif model.has_edge(dest_node, source_node):
                source_new_parents = model.get_parents(source_node)
                source_new_parents.remove(dest_node)
                dest_new_children = model.get_children(dest_node)
                dest_new_children.remove(source_node)

                must_check_for_cycle = False if not source_new_parents or \
                                                not dest_new_children else True

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
                        return ("flip", dest_node, source_node, delta_score)
                    else:
                        continue
                else:
                    return ("flip", dest_node, source_node, delta_score)

            else:
                must_check_for_cycle = False if not model.get_parents(source_node) or \
                                                not model.get_children(dest_node) else True

                if must_check_for_cycle:
                    try:
                        model.add_edge(source_node, dest_node)
                        isdag = True
                        model.remove_edge(source_node, dest_node)
                    except ValueError:
                        isdag = False

                    if isdag:
                        return ("+", source_node, dest_node, delta_score)
                    else:
                        continue
                else:
                    return ("+", source_node, dest_node, delta_score)
        return None

    def best_operator_arcs_max_indegree(self, model, scores, max_indegree):
        """
        Finds the best operator to apply to the graph. This is a version of self.best_operation() with checks for
        maximum indegree for maximum performance when indegree is not relevant.
        :param graph: The current graph model.
        :param scores: A matrix of n x n where score[i,j] is the score of adding the arc i->j if the arc is not currently
        in the graph. If the arc i->j is currently in the graph, score[i,j] is the score of removing the, and
        score[j,i] is the score of reversing the arc.
        :return: The best operator (op, source_node, dest_node, delta_score).
        """
        nnodes = model.number_of_nodes()

        # Sort in descending order. That is, [::-1].
        sort_scores = np.unravel_index(np.argsort(scores.ravel())[::-1], (nnodes, nnodes))

        for i in range(self.total_num_arcs):
            source_index = sort_scores[0][i]
            dest_index = sort_scores[1][i]
            delta_score = scores[source_index, dest_index]
            source_node = self.nodes[source_index]
            dest_node = self.nodes[dest_index]

            if model.has_edge(source_node, dest_node):
                return ("-", source_node, dest_node, delta_score)
            elif model.has_edge(dest_node, source_node):
                if len(model.get_parents(dest_node)) >= max_indegree:
                    continue

                source_new_parents = model.get_parents(source_node)
                source_new_parents.remove(dest_node)
                dest_new_children = model.get_children(dest_node)
                dest_new_children.remove(source_node)
                must_check_for_cycle = False if not source_new_parents or \
                                                not dest_new_children else True

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
                        return ("flip", dest_node, source_node, delta_score)
                    else:
                        continue
                else:
                    return ("flip", dest_node, source_node, delta_score)
            else:
                if len(model.get_parents(dest_node)) >= max_indegree:
                    continue

                must_check_for_cycle = False if not model.get_parents(source_node) or \
                                                not model.get_children(dest_node) else True

                if must_check_for_cycle:
                    try:
                        model.add_edge(source_node, dest_node)
                        isdag = True
                        model.remove_edge(source_node, dest_node)
                    except ValueError:
                        isdag = False

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
        self, start=None, tabu_length=0, max_indegree=None, epsilon=1e-4, max_iter=1e6, seed=0, seed_iters=15
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
            start = HybridContinuousModel()
            start.add_nodes_from(self.nodes)

        if max_indegree is None:
            best_operator_fun = self.best_operator
        else:
            best_operator_fun = lambda graph, scores, type_scores: self.best_operator_max_indegree(graph, scores, type_scores, max_indegree)


        self._check_blacklist(start)
        self.force_whitelist(start)

        nnodes = len(self.nodes)
        scores = np.empty((nnodes, nnodes))
        type_scores = np.empty((nnodes,))
        self._precompute_cache_node_scores(start)
        self._precompute_cache_arcs(start, scores)
        self._precompute_cache_types(start, type_scores)
        # Mark constraints with the lowest value.
        maximum_fill_value = np.ma.maximum_fill_value(scores)
        scores[~self.constraints_matrix] = maximum_fill_value

        current_model = start

        iter_no = 0
        # last_delta = np.finfo(np.float64).max
        print("Starting score: " + str(self._total_score(current_model)))

        while iter_no <= max_iter:
            iter_no += 1

            # if iter_no % seed_iters == 0:
            #     # Change seed
            #     seed = np.random.randint(0, max_int_value)
            #     self.scoring_method.change_seed(seed)
            #     self._precompute_cache_arcs(current_model, scores)
            #     self._precompute_cache_types(current_model, type_scores)

            op = best_operator_fun(current_model, scores, type_scores)

            if op is None:
                break

            delta_score = op[3]
            if delta_score < epsilon:
                break

            print("Best op: " + str(op))
            print("")
            self.apply_operator(op, current_model, scores, type_scores)
            self._draw(current_model, op, iter_no)
            print("Current score: " + str(self._total_score(current_model)))

        self._draw(current_model, None, iter_no)
        final_score = self._total_score(current_model)
        print("Final score: " + str(final_score))
        return current_model

    def _total_score(self, model):
        """
        Computes the total score in the network. As the score method is decomposable. The total score is the sum of
        the local scores.
        :param model: Graph to be evaluated.
        :return: Total score of the network.
        """
        print("")
        total_score = 0
        for node in model.nodes:
            parents = model.get_parents(node)
            total_score += self.scoring_method.local_score(node, parents, model.node_type[node], model.node_type)

            str_p = ""
            for p in parents:
                str_p += "[" + p + "," + NodeType.str(model.node_type[p]) + "], "
            print(
                "P([" + node + "," + NodeType.str(model.node_type[node]) + "] | " + str_p + ") = " +
                str(self.scoring_method.local_score(node, parents, model.node_type[node], model.node_type))
            )

        return total_score

    def _draw(self, graph, best_op, iter):
        graph_copy = graph.copy()

        total_score = 0
        for node in graph.nodes:
            parents = graph.get_parents(node)
            total_score += self.scoring_method.local_score(node, parents, graph.node_type[node], graph.node_type)

        for n in graph_copy.nodes:
            if graph_copy.node_type[n] == NodeType.CKDE:
                graph_copy.node[n]['style'] = 'filled'
                graph_copy.node[n]['fillcolor'] = 'gray'


        if best_op is None:
            # nx.nx_agraph.write_dot(graph_copy, 'iterations/{:03d}.dot'.format(iter))
            A = nx.nx_agraph.to_agraph(graph_copy)
            A.graph_attr.update(label="Score {:0.3f}".format(total_score), labelloc="t", fontsize='25')
            A.write('iterations/{:03d}.dot'.format(iter))
            A.clear()
        else:
            operation, source, dest, score = best_op

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
            elif operation == 'type' and dest == NodeType.CKDE:
                graph_copy.node[source]['style'] = 'filled'
                graph_copy.node[source]['fillcolor'] = 'darkolivegreen1'
                graph_copy.node[source]['label'] = "{}\n{:0.3f}".format(source, score)
            elif operation == 'type' and dest == NodeType.GAUSSIAN:
                graph_copy.node[source]['style'] = 'filled'
                graph_copy.node[source]['fillcolor'] = 'yellow'
                graph_copy.node[source]['label'] = "{}\n{:0.3f}".format(source, score)

            # nx.nx_agraph.write_dot(graph_copy, 'iterations/{:03d}.dot'.format(iter))
            A = nx.nx_agraph.to_agraph(graph_copy)
            A.graph_attr.update(label="Score {:0.3f}".format(total_score), labelloc="t", fontsize='25')
            A.write('iterations/{:03d}.dot'.format(iter))
            A.clear()

        import subprocess
        subprocess.run(["dot", "-Tpdf", "iterations/{:03d}.dot".format(iter), "-o",
                        "iterations/{:03d}.pdf".format(iter)])

        import os
        os.remove('iterations/{:03d}.dot'.format(iter))