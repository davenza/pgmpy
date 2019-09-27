import numpy as np
# np.seterr(all='raise')
import pandas as pd

import networkx as nx

from pgmpy.estimators import CachedHillClimbing, GaussianBicScore, BGeScore, HybridCachedHillClimbing, PredictiveLikelihood
from pgmpy.estimators.BGeScorePy import BGeScore as BGeScorePy
from pgmpy.models import HybridContinuousModel

from pgmpy.cython_backend import linear_algebra

import timeit

import pgmpy
import matplotlib.pyplot as plt

# Example Gaussian data a -> c <- b
def basic_data():
    a = np.random.normal(0, 1, 5000)
    b = np.random.normal(3.2, np.sqrt(1.8), 5000)
    c = -1.2 + 2.3*a + 1.5*b + np.random.normal(0, np.sqrt(0.5), 5000)
    data = pd.DataFrame({'a': a, 'b': b, 'c': c})
    return data

basic_data = basic_data()

def discrete_data():
    return pd.DataFrame(np.random.randint(0, 5, size=(5000, 9)), columns=list('ABCDEFGHI'))

discrete_data = discrete_data()

def ecoli_data():
    ecoli_data = pd.read_csv('ecoli_data.csv')
    ecoli_data.set_index("index", inplace=True)
    return ecoli_data

ecoli_data = ecoli_data()

def mixture_data(n_instances):
    a = np.random.normal(0, 1, n_instances)
    b = np.random.normal(3.2, np.sqrt(1.8), n_instances)

    a_negative = a < 0
    c = np.empty_like(a)
    c[a_negative] = -1.2 + 2.3*a[a_negative] + 1.5*b[a_negative] + np.random.normal(0, np.sqrt(0.5), a_negative.sum())
    c[~a_negative] = 2.3 + -0.7*a[~a_negative] + 3.0*b[~a_negative] + np.random.normal(0, np.sqrt(1), n_instances - a_negative.sum())

    data = pd.DataFrame({'a': a, 'b': b, 'c': c})
    return data

mixture_data = mixture_data(500)

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

if __name__ == '__main__':

    import gc

    gc.set_debug(gc.DEBUG_COLLECTABLE)
    
    hc = HybridCachedHillClimbing(mixture_data)
    bn = hc.estimate()

    print("=======================")
    print("=======================")

    hc = CachedHillClimbing(mixture_data)
    bn = hc.estimate()

    #
    # pred_log = PredictiveLikelihood(mixture_data)
    # print("P([b,GAUSSIAN]) = " + str(pred_log.local_score('b', set(), HybridContinuousModel.NodeType.GAUSSIAN, set())))
    # print("P([b,GAUSSIAN] | [c,GAUSSIAN) = " + str(pred_log.local_score('b', set('c'),
    #                                     HybridContinuousModel.NodeType.GAUSSIAN, {'c': HybridContinuousModel.NodeType.GAUSSIAN})))
    #
    # print("P([b,GAUSSIAN] | [c,CKDE) = " + str(pred_log.local_score('b', set('c'),
    #                                     HybridContinuousModel.NodeType.GAUSSIAN, {'c': HybridContinuousModel.NodeType.CKDE})))

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

    # hc = CachedHillClimbing(ecoli_data, BGeScore(ecoli_data))
    # bn = hc.estimate()
    # print(type(bn))
    # to_bnlearn_str(bn)




