import numpy as np
np.seterr(all='raise')
import pandas as pd

import networkx as nx

from pgmpy.estimators import CachedHillClimbing, GaussianBicScore, BGeScore
from pgmpy.estimators.BGeScorePy import BGeScore as BGeScorePy

from pgmpy.cython_backend import linear_algebra

import timeit

import pgmpy

# Example data a -> c <- b
a = np.random.normal(0, 1, 5000)
b = np.random.normal(3.2, np.sqrt(1.8), 5000)
c = -1.2 + 2.3*a + 1.5*b + np.random.normal(0, np.sqrt(0.5), 5000)

parents = sorted(['a', 'b'])
node = 'c'
data = pd.DataFrame({'a': a, 'b': b, 'c': c})
data_train = data.iloc[:3000]
data_predict = data.drop('c', axis=1, inplace=False).iloc[3000:]

data_discrete = pd.DataFrame(np.random.randint(0, 5, size=(5000, 9)), columns=list('ABCDEFGHI'))

ecoli_data = pd.read_csv('ecoli_data.csv')
ecoli_data.set_index("index", inplace=True)

def to_bnlearn_str(graph):

    for node in graph.nodes:
        print("[" + node, end='')
        parents = graph.get_parents(node)
        if parents:
            print("|" + parents[0], end='')
            for parent in parents[1:]:
                print(":" + parent, end='')
        print("]", end='')

    print()

def total_score(graph, scoring_method):
    """
    Computes the total score in the network. As the score method is decomposable. The total score is the sum of
    the local scores.
    :param graph: Graph to be evaluated.
    :return: Total score of the network.
    """
    total_score = 0
    for node in graph.nodes:
        parents = graph.get_parents(node)
        total_score += scoring_method.local_score(node, parents)

    return total_score

def det_str(rows, cols, positive, in_str):
    if len(rows) == 2 and len(cols) == 2:
        if positive:
            return in_str + "[" + str(rows[0]) + "," + str(cols[0]) + "]*" + \
                    in_str + "[" + str(rows[1]) + "," + str(cols[1]) + "] - " + \
                       in_str + "[" + str(rows[0]) + "," + str(cols[1]) + "]*" + \
                       in_str + "[" + str(rows[1]) + "," + str(cols[0]) + "]"
        else:
            return in_str + "[" + str(rows[0]) + "," + str(cols[1]) + "]*" + \
                    in_str + "[" + str(rows[1]) + "," + str(cols[0]) + "] - " + \
                    in_str + "[" + str(rows[0]) + "," + str(cols[0]) + "]*" + \
                    in_str + "[" + str(rows[1]) + "," + str(cols[1]) + "]"

    else:
        output_str = ""
        for i in range(len(cols)):
            removed_rows = rows.copy()
            del removed_rows[0]
            removed_cols = cols.copy()
            del removed_cols[i]
            if (positive and i % 2 == 0) or ((not positive) and i % 2 == 1):
                output_str += (in_str + "[" + str(rows[0]) + "," + str(cols[i]) + "]*(" + det_str(removed_rows, removed_cols, True, in_str) + ")")
            else:
                output_str += (in_str + "[" + str(rows[0]) + "," + str(cols[i]) + "]*(" + det_str(removed_rows, removed_cols, False, in_str) + ")")
            if i < len(cols)-1:
                output_str += " + \n\t\t"

        return output_str

def adjugate_str(rows, cols, in_str, out_str):
    res = ""
    for i in range(len(rows)):
        for j in range(len(cols)):
            removed_rows = rows.copy()
            del removed_rows[i]
            removed_cols = cols.copy()
            del removed_cols[j]
            if ((i+j) % 2) == 0:
                res += out_str + "[" + str(i) + "," + str(j) + "] = (" + det_str(removed_cols, removed_rows, True, in_str) + ")\n"
            else:
                res +=  out_str + "[" + str(i) + "," + str(j) + "] = -(" + det_str(removed_cols, removed_rows, True, in_str)  + ")\n"

            res += '\n'

    return res

def generate_tmp_name(rows, cols):
    return "det_" + ''.join(map(str, sorted(rows))) + "_" + ''.join(map(str, sorted(cols)))

def cache_adjugate(rows, cols, positive, in_str):
    cache = {}
    if len(rows) == 2 and len(cols) == 2:
        cache[(generate_tmp_name(rows, cols),2)] = in_str + "[" + str(rows[0]) + "," + str(cols[0]) + "]*" + \
                in_str + "[" + str(rows[1]) + "," + str(cols[1]) + "] - " + \
                   in_str + "[" + str(rows[0]) + "," + str(cols[1]) + "]*" + \
                   in_str + "[" + str(rows[1]) + "," + str(cols[0]) + "]"
    else:
        for i in range(len(cols)):
            removed_rows = rows.copy()
            del removed_rows[0]
            removed_cols = cols.copy()
            del removed_cols[i]

            if (positive and i % 2 == 0) or ((not positive) and i % 2 == 1):
                partial_cache = cache_adjugate(removed_rows, removed_cols, True, in_str)
                cache.update(partial_cache)

                if not (generate_tmp_name(rows, cols), len(rows)) in cache.keys():
                    cache[(generate_tmp_name(rows, cols), len(rows))] = (in_str + "[" + str(rows[0]) + "," + str(cols[i]) + "]*(" + generate_tmp_name(removed_rows, removed_cols) + ")")
                else:
                    cache[(generate_tmp_name(rows, cols), len(rows))] += (' \\\n\t\t +' + in_str + "[" + str(rows[0]) + "," + str(cols[i]) + "]*(" + generate_tmp_name(removed_rows, removed_cols) + ")")
            else:
                partial_cache = cache_adjugate(removed_rows, removed_cols, True, in_str)
                cache.update(partial_cache)

                if not (generate_tmp_name(rows, cols), len(rows)) in cache.keys():
                    cache[(generate_tmp_name(rows, cols), len(rows))] = (in_str + "[" + str(rows[0]) + "," + str(cols[i]) + "]*(-" + generate_tmp_name(removed_rows, removed_cols) + ")")
                else:
                    cache[(generate_tmp_name(rows, cols), len(rows))] += (' \\\n\t\t +' + in_str + "[" + str(rows[0]) + "," + str(cols[i]) + "]*(-" + generate_tmp_name(removed_rows, removed_cols) + ")")

    return cache

def det_str_cached(rows, cols, positive, in_str):
    if len(rows) == 2 and len(cols) == 2:
        return generate_tmp_name(rows, cols)
    else:
        output_str = ""
        for i in range(len(cols)):
            removed_rows = rows.copy()
            del removed_rows[0]
            removed_cols = cols.copy()
            del removed_cols[i]

            if (positive and i % 2 == 0) or ((not positive) and i % 2 == 1):
                if output_str == "":
                    output_str += (in_str + "[" + str(rows[0]) + "," + str(cols[i]) + "]*(" + generate_tmp_name(removed_rows, removed_cols) + ")")
                else:
                    output_str += (" \\\n\t\t + " + in_str + "[" + str(rows[0]) + "," + str(cols[i]) + "]*(" + generate_tmp_name(removed_rows, removed_cols) + ")")
            else:
                if output_str == "":
                    output_str += (in_str + "[" + str(rows[0]) + "," + str(cols[i]) + "]*(-" + generate_tmp_name(removed_rows, removed_cols) + ")")
                else:
                    output_str += (" \\\n\t\t + " + in_str + "[" + str(rows[0]) + "," + str(cols[i]) + "]*(-" + generate_tmp_name(removed_rows, removed_cols) + ")")


    return output_str

def adjugate_str_cached(rows, cols, in_str, out_str):
    cache = {}

    res = ""
    # for j in range(len(cols)):
    for j in range(1):
        for i in range(len(rows)):
            removed_rows = rows.copy()
            del removed_rows[i]
            removed_cols = cols.copy()
            del removed_cols[j]
            if ((i + j) % 2) == 0:
                partial_cache = cache_adjugate(removed_rows, removed_cols, True, in_str)
                cache.update(partial_cache)
            else:
                partial_cache = cache_adjugate(removed_rows, removed_cols, True, in_str)
                cache.update(partial_cache)

    sorted_keys = sorted(cache.keys(), key=lambda kv: (kv[1], kv[0]))

    res += "cdef double " + sorted_keys[0][0]
    for key in sorted_keys[1:]:
        if key[1] < len(rows) - 1:
            res += "," + key[0]

    res += '\n\n'

    for key in sorted_keys:
        if key[1] < len(rows) - 1:
            res += key[0] + " = " + cache[key] + '\n'

    res += '\n\n'
    # for j in range(len(cols)):
    for j in range(1):
        for i in range(len(rows)):
            removed_rows = rows.copy()
            del removed_rows[i]
            removed_cols = cols.copy()
            del removed_cols[j]
            if ((i+j) % 2) == 0:
                res += out_str + "[" + str(j) + "][" + str(i) + "] = " + det_str_cached(removed_rows, removed_cols, True, in_str) + "\n"
            else:
                res += out_str + "[" + str(j) + "][" + str(i) + "] = " + det_str_cached(removed_rows, removed_cols, False, in_str) + "\n"

            res += '\n'

    return res, cache



if __name__ == '__main__':
    # rows = cols = list(range(4))
    # code4, cache4 = adjugate_str_cached(rows, cols, "mat", "tmp_inv")
    # print(code4)
    # rows = cols = list(range(4))
    # code5, cache5 = adjugate_str_cached(rows, cols, "mat", "tmp_inv")
    # print(code5)
    # cache = cache4
    # cache.update(cache5)
    # sorted_keys = sorted(cache.keys(), key=lambda kv: (kv[1], kv[0]))
    #
    # print("cdef double " + sorted_keys[0][0], end='')
    # for key in sorted_keys[1:]:
    #     if key[1] < 4:
    #         print(", " + key[0], end='')
    # print()


    #
    #
    #
    # bge = BGeScore(ecoli_data)
    # print("Score aceB -> asnA " + str(bge.local_score("asnA", ["aceB"])))
    # print("Score asnA " + str(bge.local_score("asnA", [])))
    # print("Delta " + str(bge.local_score("asnA", ["aceB"]) - bge.local_score("asnA", [])))
    # print(bge.benchmark("cspG", ["atpD"]))

    # print(bge.benchmark("cspG", []))
    # print(bge.local_score("cspG", ["atpD"]))

    hc = CachedHillClimbing(ecoli_data, BGeScore(ecoli_data))
    bn = hc.estimate()
    print(type(bn))
    to_bnlearn_str(bn)




