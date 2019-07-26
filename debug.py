import numpy as np
np.seterr(all='raise')
import pandas as pd

import networkx as nx

from pgmpy.estimators import CachedHillClimbing, GaussianBicScore, BGeScore
from pgmpy.estimators.BGeScorePy import BGeScore as BGeScorePy

import timeit

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




if __name__ == '__main__':
    bge = BGeScore(ecoli_data)
    bge.local_score("asnA", ["aceB"])

    # bge.local_score("aceB", ["atpD", "b1583", "b1963", "cspG", "dnaG", "dnaJ", "dnaK", "eutG", "fixC", "flgD"])
    # hc_ecoli = CachedHillClimbing(ecoli_data, scoring_method=BGeScore(ecoli_data))
    # network = hc_ecoli.estimate()
    # to_bnlearn_str(network)
    # print(network.edges)
    #
    # print("Num edges: " + str(len(network.edges)))


