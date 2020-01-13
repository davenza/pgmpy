#!/usr/bin/env python

import numpy as np
from scipy.optimize import brentq as root
import networkx as nx

from pgmpy.estimators import StructureEstimator, CVPredictiveLikelihood, ValidationLikelihood
from pgmpy.base import DAG
from pgmpy.models import HybridContinuousModel
from pgmpy.factors.continuous import NodeType

from scipy import stats

import math

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
            self.scoring_method = CVPredictiveLikelihood(data)
        else:
            if not isinstance(scoring_method, (CVPredictiveLikelihood, ValidationLikelihood)):
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
            # Delta score of reversing arc 'node' -> 'other_node'
            elif model.has_edge(node, other_node):
                other_node_parents = set(model.get_parents(other_node))

                parents_new = parents.copy()
                parents_new.add(other_node)

                other_node_parents_new = other_node_parents.copy()
                other_node_parents_new.remove(node)

                other_node_type = model.node_type[other_node]

                scores[other_index, node_index] = local_score(other_node, other_node_parents_new, other_node_type, model.node_type) +\
                                                local_score(node, parents_new, node_type, model.node_type) -\
                                                self.node_scores[other_index] -\
                                                self.node_scores[node_index]
                print("Caching reversing arc " + node + " -> " + other_node + " (" + str(scores[other_index, node_index]) + ")")
            else:
                # Delta score of adding arc 'other_node' -> 'node'
                parents_new = parents.copy()
                parents_new.add(other_node)
                scores[other_index, node_index] = local_score(node, parents_new, node_type, model.node_type) -\
                                                  self.node_scores[node_index]
                print("Caching adding arc " + other_node + " -> " + node + " (" + str(scores[other_index, node_index]) + ")")

    def _precompute_score_types_node(self, model, scores, node):
        node_index = self.nodes_indices[node]
        local_score = self.scoring_method.local_score
        parents = set(model.get_parents(node))
        if model.node_type[node] == NodeType.GAUSSIAN:
            other_node_type = NodeType.CKDE
            print("Caching changing type of node " + node + " to CKDE", end='')
        elif model.node_type[node] == NodeType.CKDE:
            other_node_type = NodeType.GAUSSIAN
            print("Caching changing type of node " + node + " to Gaussian", end='')

        else:
            raise ValueError("Wrong node type for HybridContinuousModel.")

        scores[node_index] = local_score(node, parents, other_node_type, model.node_type) - \
                                 self.node_scores[node_index]

        children = model.get_children(node)
        for child in children:
            if model.node_type[child] == NodeType.CKDE:
                child_parents = model.get_parents(child)
                new_parent_type = model.node_type.copy()
                new_parent_type[node] = other_node_type
                child_index = self.nodes_indices[child]

                scores[node_index] += local_score(child, child_parents, model.node_type[child], new_parent_type) - \
                                           self.node_scores[child_index]

        print(" (" + str(scores[node_index]) + ")")

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
            # print("Updating changing type of node " + node + " to CKDE", end='')
        elif node_type == NodeType.CKDE:
            other_node_type = NodeType.GAUSSIAN
            # print("Updating changing type of node " + node + " to Gaussian", end='')
        else:
            raise ValueError("Wrong node type detected on node {}.", node)

        type_scores[node_index] = local_score(node, parents, other_node_type, model.node_type) - \
                                  self.node_scores[node_index]

        children = model.get_children(node)

        for child in children:
            if model.node_type[child] == NodeType.CKDE:
                child_parents = model.get_parents(child)
                new_parent_type = model.node_type.copy()
                new_parent_type[node] = other_node_type
                child_index = self.nodes_indices[child]
                type_scores[node_index] += local_score(child, child_parents, model.node_type[child], new_parent_type) - \
                                           self.node_scores[child_index]


    def score_add_arc(self, model, source, dest):
        local_score = self.scoring_method.local_score

        dest_index = self.nodes_indices[dest]

        parents = set(model.get_parents(dest))
        parents_new = parents.copy()
        parents_new.add(source)

        return local_score(dest, parents_new, model.node_type[dest], model.node_type) -\
               self.node_scores[dest_index]

    def score_remove_arc(self, model, source, dest):
        local_score = self.scoring_method.local_score

        dest_index = self.nodes_indices[dest]

        parents = set(model.get_parents(dest))
        parents_new = parents.copy()
        parents_new.remove(source)

        return local_score(dest, parents_new, model.node_type[dest], model.node_type) -\
               self.node_scores[dest_index]

    def score_flip_arc(self, model, source, dest):
        return self.score_remove_arc(model, source, dest) + self.score_add_arc(model, dest, source)

    def set_update_arcs_to_node(self, model, dest):
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

    def set_update_arcs_from_ckde(self, model, source):
        source_index = self.nodes_indices[source]

        to_update = np.where(self.constraints_matrix[source_index, :])[0]

        update_set = set()
        for other_index in to_update:
            other_node = self.nodes[other_index]

            if model.node_type[other_node] != NodeType.CKDE:
                continue

            if model.has_edge(source, other_node):
                # Delta score of removing arc 'source' -> 'other_node'
                update_set.add((source, other_node))

            elif not model.has_edge(other_node, source):
                update_set.add((source, other_node))
                # Delta score of adding arc 'source' -> 'other_node'

        return update_set

    def update_node_scores(self, model, node_set):
        local_score = self.scoring_method.local_score
        for n in node_set:
            parents = model.get_parents(n)
            n_index = self.nodes_indices[n]
            self.node_scores[n_index] = local_score(n, parents, model.node_type[n], model.node_type)

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

    def update_type_scores(self, model, type_scores, node_set):
        for n in node_set:
            self.update_node_score_types(model, type_scores, n)


    def apply_operator(self, op, model, scores, type_scores):
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
        to_update_types = set()

        if operation == "+":
            model.add_edge(source, dest)

            to_update_nodes.add(dest)
            to_update_arcs.update(self.set_update_arcs_to_node(model, dest))
            to_update_types.add(dest)

            if model.node_type[dest] == NodeType.CKDE:
                for p in model.get_parents(dest):
                    to_update_types.add(p)

        elif operation == "-":
            model.remove_edge(source, dest)

            to_update_nodes.add(dest)
            to_update_arcs.update(self.set_update_arcs_to_node(model, dest))
            # Update the score of adding the score in the opposite direction (dest -> source).
            to_update_arcs.add((dest, source))

            to_update_types.add(dest)

            if model.node_type[dest] == NodeType.CKDE:
                for p in model.get_parents(dest):
                    to_update_types.add(p)

        elif operation == "flip":
            model.remove_edge(source, dest)
            model.add_edge(dest, source)

            to_update_nodes.update((source, dest))

            to_update_arcs.update(self.set_update_arcs_to_node(model, dest))
            to_update_arcs.update(self.set_update_arcs_to_node(model, source))

            to_update_types.update((dest, source))

            if model.node_type[dest] == NodeType.CKDE:
                to_update_types.update(model.get_parents(dest))
            if model.node_type[source] == NodeType.CKDE:
                to_update_types.update(model.get_parents(source))

        elif operation == "type":
            model.node_type[source] = dest

            to_update_nodes.add(source)

            children = model.get_children(source)
            for child in children:
                if model.node_type[child] == NodeType.CKDE:
                    to_update_nodes.add(child)
                    to_update_arcs.update(self.set_update_arcs_to_node(model, child))
                    to_update_types.update(model.get_parents(child))

            to_update_arcs.update(self.set_update_arcs_to_node(model, source))
            to_update_arcs.update(self.set_update_arcs_from_ckde(model, source))

            to_update_types.update(model.get_parents(source) + children)
            to_update_types.add(source)

        self.update_node_scores(model, to_update_nodes)
        self.update_arc_scores(model, scores, to_update_arcs)
        self.update_type_scores(model, type_scores, to_update_types)

    def ttest_onesample_sample_size_power(self, d, alpha, power, starting_N, alternative, maximum_samples):
        """
        From pwr.t.test R function.
        :param d:
        :param alpha:
        :param power:
        :param type:
        :param alternative:
        :return:
        """
        if alternative == "greater":
            power_function = self.ttest_power_greater
        elif alternative == "two.sided":
            power_function = self.ttest_power_twosided
            d = math.fabs(d)
        elif alternative == "less":
            power_function = self.ttest_power_less
        else:
            raise ValueError("alternative should be an string with one of the following values (\"less\", \"two.sided\", \"greater\")")


        lower_power = power_function(starting_N, d, alpha)

        if lower_power >= power:
            return starting_N, lower_power

        higher_power = power_function(maximum_samples, d, alpha)

        if higher_power < power:
            # Can't calculate enough samples
            return maximum_samples, higher_power

        needed_N = math.ceil(root(lambda n: power_function(n, d, alpha) - power, starting_N, maximum_samples))

        return needed_N, power_function(needed_N, d, alpha)

    def ttest_power_greater(self, n, d, alpha):
        nu = n - 1
        return 1-stats.nct.cdf(stats.t.ppf(1-alpha, nu), nu, math.sqrt(n)*d)

    def ttest_power_twosided(self, n, d, alpha):
        nu = n - 1
        qu = stats.t.ppf(1-alpha, nu)

        return 1-stats.nct.cdf(qu, nu, math.sqrt(n)*d) + (stats.nct.cdf(-qu, nu, math.sqrt(n)*d))

    def ttest_power_less(self, n, d, alpha):
        nu = n - 1
        return stats.nct.cdf(stats.t.ppf(alpha, nu), nu, math.sqrt(n)*d)

    def sample_crossvalidation_power_operation(self, model, op, scores, epsilon, d, alpha, starting_N):
        N = starting_N
        operation = op[0]

        if operation == "+" or operation == "-" or operation == "flip":
            dest_type = model.node_type[op[2]]
            multiple_operation_fun = self.multiple_crossvalidation_operation_arcs
        elif operation == "type":
            dest_type = op[2]
            multiple_operation_fun = self.multiple_crossvalidation_operation_types
        else:
            raise ValueError("Wrong operator.")

        if dest_type == NodeType.CKDE:
            maximum_samples = 100
        elif dest_type == NodeType.GAUSSIAN:
            maximum_samples = 3000

        needed_N, power = self.ttest_onesample_sample_size_power(d, alpha, 0.8, N, "two.sided", maximum_samples)

        if needed_N is None:
            return scores, N, None

        while needed_N > N:
            needed_scores = multiple_operation_fun(model, op, needed_N - N)
            scores = np.hstack((scores, needed_scores))
            N = needed_N

            d = (scores.mean() - epsilon) / scores.std()

            if d <= 0:
                return scores, N, None

            needed_N, power = self.ttest_onesample_sample_size_power(d, alpha, 0.8, N, "greater", maximum_samples)

            if needed_N is None:
                return scores, N, None

        return scores, needed_N, power

    def sample_crossvalidation_power_convergence(self, model, scores, total_score, d, alpha, starting_N):
        N = starting_N

        if any(n == NodeType.CKDE for n in model.node_type.values()):
            maximum_samples = 100
        else:
            maximum_samples = 3000

        needed_N, power = self.ttest_onesample_sample_size_power(d, alpha, 0.8, N, "two.sided", maximum_samples)

        if needed_N is None:
            return scores, N, None

        max_uint = np.iinfo(np.int32).max
        while needed_N > N:
            additional_scores = np.empty((needed_N-N,))
            print()
            for i in range(needed_N-N):
                print("\rMore convergence samples (" + str(needed_N - N) + "): " + str(i+1) + "/" + str(needed_N-N), end='')
                self.scoring_method.change_seed(np.random.randint(0, max_uint))
                additional_scores[i] = self._total_score()
            print()

            scores = np.hstack((scores, additional_scores))
            N = needed_N

            d = (scores.mean() - total_score) / scores.std()

            if d <= 0:
                return scores, N, None

            needed_N, power = self.ttest_onesample_sample_size_power(d, alpha, 0.8, N, "two.sided", maximum_samples)

            if needed_N is None:
                return scores, N, None

        return scores, needed_N, power

    def is_significant_operator(self, model, op, starting_N, epsilon, alpha):
        print("Checking significance for operator: " + str(op))
        operation = op[0]
        N = starting_N

        if operation == "+" or operation == "-" or operation == "flip":
            scores = self.multiple_crossvalidation_operation_arcs(model, op, N)
        elif operation == "type":
           scores = self.multiple_crossvalidation_operation_types(model, op, N)
        else:
            raise ValueError("Wrong operator")

        d = (scores.mean() - epsilon) / scores.std()

        print("Scores mean: " + str(scores.mean()))
        print("Scores std: " + str(scores.std()))

        t_statistic = (scores.mean() - epsilon) * math.sqrt(N) / scores.std()

        print("Cross validated_scores: " + str(scores))
        print("p-value: ", 1-stats.t.cdf(t_statistic, N-1))

        power = self.ttest_power_greater(N, d, alpha)
        print("power: ", power)

        if stats.t.cdf(t_statistic, N-1) > 1-alpha:
            if d > 0:
                scores, N_new, power = self.sample_crossvalidation_power_operation(model, op, scores, epsilon, d, alpha, starting_N)

                if N_new > N:
                    t_statistic = (scores.mean() - epsilon) * math.sqrt(N_new) / scores.std()
                    print("p-value: ", 1-stats.t.cdf(t_statistic, N_new-1))
                    print("power: ", power)

                    if stats.t.cdf(t_statistic, N_new-1) > 1-alpha:
                        return True
                    else:
                        return False

            return True
        else:
            return False

    def multiple_crossvalidation_operation_arcs(self, model, op, N):

        operation, source, dest, _ = op

        old_parents = model.get_parents(dest)

        new_parents = old_parents.copy()
        if operation == "+":
            new_parents.append(source)
        else:
            new_parents.remove(source)

        max_uint = np.iinfo(np.int32).max
        old_seed = self.scoring_method.seed

        scores = np.empty((N,))
        seeds = np.empty((N,), dtype=np.int32)

        print()
        for i in range(N):
            print("\rConvergence samples arcs: {}/{}".format(i+1, N), end='')
            seeds[i] = np.random.randint(0, max_uint)
            self.scoring_method.change_seed(seeds[i])
            scores[i] = self.scoring_method.local_score(dest, new_parents, model.node_type[dest], model.node_type) - \
                        self.scoring_method.local_score(dest, old_parents, model.node_type[dest], model.node_type)

        if operation == "flip":
            source_parents = model.get_parents(source)
            source_parents_new = source_parents.copy()
            source_parents_new.append(dest)
            for i in range(N):
                print("\rConvergence samples arcs: {}/{} (flip)".format(i+1, N), end='')
                self.scoring_method.change_seed(seeds[i])
                scores[i] += self.scoring_method.local_score(source, source_parents_new, model.node_type[source], model.node_type) - \
                             self.scoring_method.local_score(source, source_parents, model.node_type[source], model.node_type)

        print()
        self.scoring_method.change_seed(old_seed)
        return scores

    def multiple_crossvalidation_operation_types(self, model, op, N):

        _, node, new_type, _ = op
        parents = model.get_parents(node)
        children = model.get_children(node)

        current_type = model.node_type[node]

        max_uint = np.iinfo(np.int32).max
        old_seed = self.scoring_method.seed

        scores = np.empty((N,))
        seeds = np.empty((N,), dtype=np.int32)

        for i in range(N):
            print("\r {}/{}".format(i+1, N), end='')
            seeds[i] = np.random.randint(0, max_uint)
            self.scoring_method.change_seed(seeds[i])
            scores[i] = self.scoring_method.local_score(node, parents, new_type, model.node_type) - \
                        self.scoring_method.local_score(node, parents, current_type, model.node_type)

            for child in children:
                if model.node_type[child] == NodeType.CKDE:
                    child_parents = model.get_parents(child)
                    new_parent_type = model.node_type.copy()
                    new_parent_type[node] = new_type
                    scores[i] += self.scoring_method.local_score(child, child_parents, model.node_type[child], new_parent_type) - \
                                 self.scoring_method.local_score(child, child_parents, model.node_type[child], model.node_type)

        print()
        self.scoring_method.change_seed(old_seed)
        return scores

    def best_operator(self, model, scores, type_scores, epsilon, significant_threshold, alpha):
        best_type = self.best_operator_types(model, type_scores)
        best_arc = self.best_operator_arcs(model, scores)

        delta_score_type = best_type[3]
        delta_score_arcs = best_arc[3]

        if delta_score_arcs is None or delta_score_type > delta_score_arcs:
            best_op = best_type
        else:
            best_op = best_arc

        if (best_op[3] - epsilon) > significant_threshold:
            return best_op
        elif math.fabs(best_op[3] - epsilon) < significant_threshold:
            significative = self.is_significant_operator(model, best_op, 10000, epsilon, alpha)
            if significative:
                return best_op
            else:
                return None
        else:
            return None

    def best_operator_validation(self, model, scores, type_scores, epsilon, tabu):
        best_type = self.best_operator_types_tabu(model, type_scores, tabu)
        best_arc = self.best_operator_arcs_tabu(model, scores, tabu)

        if best_arc is None:
            if best_type is None or best_type[3] < epsilon:
                best_op = None
            else:
                best_op = best_type
        else:
            if best_type is None:
                if best_arc[3] < epsilon:
                    best_op = None
                else:
                    best_op = best_arc
            else:
                if max(best_arc[3], best_type[3]) < epsilon:
                    best_op = None
                else:
                    if best_arc[3] >= best_type[3]:
                        best_op = best_arc
                    else:
                        best_op = best_type

        return best_op

    def validation_delta(self, model, op):
        op, source, dest, _ = op

        if op == "+":
            node_type = model.node_type[dest]

            parents = set(model.get_parents(dest))
            parents_new = parents.copy()
            parents_new.add(source)

            return self.scoring_method.validation_local_score(dest, parents_new, node_type, model.node_type) - \
                   self.scoring_method.validation_local_score(dest, parents, node_type, model.node_type)

        elif op == "-":
            node_type = model.node_type[dest]

            parents = set(model.get_parents(dest))
            parents_new = parents.copy()
            parents_new.remove(source)

            return self.scoring_method.validation_local_score(dest, parents_new, node_type, model.node_type) - \
                   self.scoring_method.validation_local_score(dest, parents, node_type, model.node_type)
        elif op == "flip":
            node_type = model.node_type[dest]
            other_node_type = model.node_type[source]

            parents = set(model.get_parents(dest))
            other_node_parents = set(model.get_parents(source))

            parents_new = parents.copy()
            parents_new.remove(source)
            other_node_parents_new = other_node_parents.copy()
            other_node_parents_new.add(dest)

            return self.scoring_method.validation_local_score(dest, parents_new, node_type, model.node_type) + \
                   self.scoring_method.validation_local_score(source, other_node_parents_new, other_node_type, model.node_type) - \
                   self.scoring_method.validation_local_score(dest, parents, node_type, model.node_type) - \
                   self.scoring_method.validation_local_score(source, other_node_parents, other_node_type, model.node_type)

        elif op == "type":
            node_type = model.node_type[source]
            node_type_new = dest

            new_node_type = model.node_type.copy()
            new_node_type[source] = node_type_new

            parents = model.get_parents(source)
            children = model.get_children(source)

            validation_score = self.scoring_method.validation_local_score(source, parents, node_type_new, new_node_type) - \
                               self.scoring_method.validation_local_score(source, parents, node_type, model.node_type)

            for child in children:
                if model.node_type[child] == NodeType.CKDE:
                    child_parents = model.get_parents(child)
                    child_type = model.node_type[child]
                    validation_score += self.scoring_method.validation_local_score(child, child_parents, child_type, new_node_type) - \
                                        self.scoring_method.validation_local_score(child, child_parents, child_type, model.node_type)

            return validation_score
        else:
            raise ValueError("Wrong operator.")


    def best_operator_max_indegree(self, model, scores, type_scores, max_indegree, epsilon, significant_threshold, alpha):
        best_type = self.best_operator_types(model, type_scores)
        best_arc = self.best_operator_arcs_max_indegree(model, scores, max_indegree)

        delta_score_type = best_type[3]
        delta_score_arcs = best_arc[3]

        if delta_score_arcs is None or delta_score_type > delta_score_arcs:
            best_op = best_type
        else:
            best_op = best_arc

        if (best_op[3] - epsilon) > significant_threshold:
            return best_op
        elif math.fabs(best_op[3] - epsilon) < significant_threshold:
            significative = self.is_significant_operator(model, best_op, 200, epsilon, alpha)
            if significative:
                return best_op
            else:
                return None
        else:
            return None

    def best_operator_types(self, model, type_scores):
        index = type_scores.argmax()
        node = self.nodes[index]

        if model.node_type[node] == NodeType.GAUSSIAN:
            return ("type", node, NodeType.CKDE, type_scores[index])
        else:
            return ("type", node, NodeType.GAUSSIAN, type_scores[index])

    def best_operator_types_tabu(self, model, type_scores, tabu):
        sorted_scores = np.argsort(type_scores)[::-1]
        for i in sorted_scores:
            node = self.nodes[i]

            if model.node_type[node] == NodeType.GAUSSIAN:
                o = ("type", node, NodeType.CKDE)
                if o not in tabu:
                    return ("type", node, NodeType.CKDE, type_scores[i])
                else:
                    continue
            else:
                o = ("type", node, NodeType.GAUSSIAN)
                if o not in tabu:
                    return ("type", node, NodeType.GAUSSIAN, type_scores[i])
                else:
                    continue

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

    def best_operator_arcs_tabu(self, model, scores, tabu):
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
                if ("-", source_node, dest_node) not in tabu:
                    return ("-", source_node, dest_node, delta_score)
                else:
                    continue
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
                        if ("flip", dest_node, source_node) not in tabu:
                            return ("flip", dest_node, source_node, delta_score)
                        else:
                            continue
                    else:
                        continue
                else:
                    if ("flip", dest_node, source_node) not in tabu:
                        return ("flip", dest_node, source_node, delta_score)
                    else:
                        continue

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
                        if ("+", source_node, dest_node) not in tabu:
                            return ("+", source_node, dest_node, delta_score)
                        else:
                            continue
                    else:
                        continue
                else:
                    if ("+", source_node, dest_node) not in tabu:
                        return ("+", source_node, dest_node, delta_score)
                    else:
                        continue
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


    def has_converged(self, model, starting_N, alpha):
        total_score = self.node_scores.sum()

        N = starting_N

        scores = np.empty((N,))
        max_uint = np.iinfo(np.int32).max

        print()
        for i in range(N):
            print("\rChecking convergence (" + str(N) + "): " + str(i+1) + "/" + str(N), end='')
            self.scoring_method.change_seed(np.random.randint(0, max_uint))
            scores[i] = self._total_score()
        print()

        d = (scores.mean() - total_score) / scores.std()

        scores, N, power = self.sample_crossvalidation_power_convergence(model, scores, total_score, d, alpha, starting_N)

        t_statistic = (scores.mean() - total_score) * math.sqrt(N) / scores.std()
        cdf_t = stats.t.cdf(t_statistic, N-1)

        if cdf_t >= 0.5:
            p_value = (1-cdf_t)*2
        else:
            p_value = cdf_t*2

        print("Other scores: " + str(scores))
        print()
        print("Convergence analysis:")
        print("--------------------------")
        print("Model score: " + str(total_score))
        print("Scores mean: " + str(scores.mean()))
        print("Scores std: " + str(scores.std()))
        print("p-value: " + str(p_value))
        print("power: " + str(power))
        if cdf_t >= alpha/2 and cdf_t <= (1-alpha/2):
            return True
        else:
            print("Not converged")
            return False

    # FIXME: Implement tabu.
    def estimate_cv(
        self, start=None, tabu_length=0, max_indegree=None, epsilon=1e-4, max_iter=1e6,
            significant_threshold=5, significant_alpha=0.05, callbacks=[]
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
            best_operator_fun = lambda graph, scores, type_scores: self.best_operator(graph,
                                                                                      scores,
                                                                                      type_scores,
                                                                                      epsilon,
                                                                                      significant_threshold,
                                                                                      significant_alpha)
        else:
            best_operator_fun = lambda graph, scores, type_scores: self.best_operator_max_indegree(graph,
                                                                                                   scores,
                                                                                                   type_scores,
                                                                                                   max_indegree,
                                                                                                   epsilon,
                                                                                                   significant_threshold,
                                                                                                   significant_alpha)

        self._check_blacklist(start)
        self.force_whitelist(start)

        nnodes = len(self.nodes)
        scores = np.empty((nnodes, nnodes))
        type_scores = np.empty((nnodes,))

        self._precompute_cache_node_scores(start)
        self._precompute_cache_arcs(start, scores)
        # Mark constraints with the lowest value.
        maximum_fill_value = np.ma.maximum_fill_value(scores)
        scores[~self.constraints_matrix] = maximum_fill_value
        self._precompute_cache_types(start, type_scores)

        current_model = start

        iter_no = 0
        print("Starting score: " + str(self._total_score_print(current_model)))

        max_uint = np.iinfo(np.int32).max

        iter_no_improvement = 0
        current_score = self._total_score()

        for callback in callbacks:
            callback.call(current_model, None, self.scoring_method, iter_no)

        while iter_no <= max_iter and iter_no_improvement <= 5:
            iter_no += 1
            print("Iteration " + str(iter_no))
            print("----------------------")

            op = best_operator_fun(current_model, scores, type_scores)

            if op is None:
                if self.has_converged(current_model, 30, significant_alpha):
                    break
                else:
                    iter_no_improvement += 1
                    self.scoring_method.change_seed(np.random.randint(0, max_uint))
                    self._precompute_cache_node_scores(current_model)
                    self._precompute_cache_arcs(current_model, scores)
                    self._precompute_cache_types(current_model, type_scores)
                    current_score = self.node_scores.sum()
                    print("Current score: " + str(self.node_scores.sum()))
                    continue

            iter_no_improvement = 0

            print("Best op: " + str(op))
            print()

            self.apply_operator(op, current_model, scores, type_scores)
            new_score = self._total_score()

            if not np.isclose(new_score, current_score + op[3]):
                print("Error on scores")
                input()

            for callback in callbacks:
                callback.call(current_model, op, self.scoring_method, iter_no)

            print("Current score: " + str(self.node_scores.sum()))
            current_score = new_score

        for callback in callbacks:
            callback.call(current_model, None, self.scoring_method, iter_no)


        final_score = self._total_score_print(current_model)
        print("Final score: " + str(final_score))
        return current_model

    # FIXME: Implement tabu.
    def estimate_validation(
            self, start=None, tabu_length=0, max_indegree=None, epsilon=1e-4, max_iter=1e6, patience=0,
            callbacks=[]
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
        if not isinstance(self.scoring_method, ValidationLikelihood):
            raise TypeError("estimate_validation() must be executed using a scoring method ValidationLikelihood.")

        if start is None:
            start = HybridContinuousModel()
            start.add_nodes_from(self.nodes)

        tabu_last = set()

        if max_indegree is None:
            best_operator_fun = lambda graph, scores, type_scores: self.best_operator_validation(graph,
                                                                                      scores,
                                                                                      type_scores,
                                                                                      epsilon,
                                                                                      tabu_last
                                                                                    )
        else:
            raise NotImplementedError("Implement max indegree execution.")

        self._check_blacklist(start)
        self.force_whitelist(start)

        nnodes = len(self.nodes)
        scores = np.empty((nnodes, nnodes))
        type_scores = np.empty((nnodes,))

        self._precompute_cache_node_scores(start)
        self._precompute_cache_arcs(start, scores)
        # Mark constraints with the lowest value.
        maximum_fill_value = np.ma.maximum_fill_value(scores)
        scores[~self.constraints_matrix] = maximum_fill_value
        self._precompute_cache_types(start, type_scores)

        current_model = start

        n_train_instances = len(self.scoring_method.data)
        n_validation_instances = len(self.scoring_method.validation_data)

        iter_no = 0
        iter_no_improvement = 0

        current_score = self._total_score()
        best_validation_score = self._total_validation_score(current_model)
        best_model_index = 0

        print("Starting score: " + str(current_score) + " (" + str(current_score / n_train_instances) + " / instance)")
        print("Validation score: " + str(best_validation_score) + " (" + str(best_validation_score / n_validation_instances) + " / instance)")

        models = [current_model.copy()]
        scores_history = [current_score]
        scores_validation_history = [best_validation_score]

        for callback in callbacks:
            callback.call(current_model, None, self.scoring_method, iter_no)

        while iter_no <= max_iter:
            iter_no += 1
            print("Iteration " + str(iter_no))
            print("----------------------")

            op = best_operator_fun(current_model, scores, type_scores)

            if op is None:
                print("----------------------------")
                print("Best validation score: " + str(best_validation_score))
                print("----------------------------")
                current_model = models[best_model_index]
                self._precompute_cache_node_scores(current_model)
                break

            print("Best op: " + str(op))
            print()

            self.apply_operator(op, current_model, scores, type_scores)

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
                elif op[0] == "type":
                    if op[2] == NodeType.CKDE:
                        other_type = NodeType.GAUSSIAN
                    elif op[2] == NodeType.GAUSSIAN:
                        other_type = NodeType.CKDE

                    tabu_last.add(("type", op[1], other_type))
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

            print("Current score: " + str(new_score) + " (" +str(new_score / n_train_instances) + " / instance)")
            val_score = self._total_validation_score(current_model)
            print("Current validation score: " + str(val_score) + " (" +str(val_score / n_validation_instances) + " / instance)")
            current_score = new_score

        for callback in callbacks:
            callback.call(current_model, None, self.scoring_method, iter_no)

        final_score = self._total_score()
        print("Final score: " + str(final_score) + " (" + str(final_score / n_train_instances) + " / instance)")
        val_score = self._total_validation_score(current_model)
        print("Final validation score: " + str(val_score) + " (" + str(val_score / n_validation_instances) + " / instance)")
        return current_model

    def estimate(self, start=None, tabu_length=0, max_indegree=None, epsilon=1e-4, max_iter=1e6, patience=0,
                         significant_threshold=5, significant_alpha=0.05, callbacks=[]):

        if isinstance(self.scoring_method, ValidationLikelihood):
            return self.estimate_validation(start=start, tabu_length=tabu_length, max_indegree=max_indegree,
                                            epsilon=epsilon, max_iter=max_iter, patience=patience,
                                            callbacks=callbacks)
        elif isinstance(self.scoring_method, CVPredictiveLikelihood):
            return self.estimate_cv(start=start, tabu_length=tabu_length, max_indegree=max_indegree, epsilon=epsilon,
                                    max_iter=max_iter, significant_threshold=significant_threshold,
                                    significant_alpha=significant_alpha, callbacks=callbacks)

    def _total_score(self):
        """
        Computes the total score in the network. As the score method is decomposable. The total score is the sum of
        the local scores.
        :param model: Graph to be evaluated.
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
            a = self.scoring_method.validation_local_score(node, parents, model.node_type[node], model.node_type)

            total_score += a

        return total_score

    def _total_score_print(self, model):
        """
        Computes the total score in the network. As the score method is decomposable. The total score is the sum of
        the local scores.
        :param model: Graph to be evaluated.
        :return: Total score of the network.
        """
        print("")
        for node in model.nodes:
            parents = model.get_parents(node)
            node_index = self.nodes_indices[node]

            str_p = ""
            for p in parents:
                str_p += "[" + p + "," + NodeType.str(model.node_type[p]) + "], "
            print(
                "P([" + node + "," + NodeType.str(model.node_type[node]) + "] | " + str_p + ") = " +
                str(self.node_scores[node_index])
            )

        return self.node_scores.sum()