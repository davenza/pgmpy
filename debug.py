import numpy as np
# np.seterr(all='raise')
np.random.seed(0)
import pandas as pd

from pgmpy.estimators import MaximumLikelihoodEstimator, CachedHillClimbing, HybridCachedHillClimbing, PredictiveLikelihood
from pgmpy.factors.continuous import NodeType, CKDE_CPD
from pgmpy.models import HybridContinuousModel

from time import time

# Example Gaussian data a -> c <- b
def basic_data_f():
    a = np.random.normal(0, 1, 5000)
    b = np.random.normal(3.2, np.sqrt(1.8), 5000)
    c = -1.2 + 2.3*a + 1.5*b + np.random.normal(0, np.sqrt(0.5), 5000)
    data = pd.DataFrame({'a': a, 'b': b, 'c': c})
    return data

basic_data = basic_data_f()

def discrete_data_f():
    return pd.DataFrame(np.random.randint(0, 5, size=(5000, 9)), columns=list('ABCDEFGHI'))

discrete_data = discrete_data_f()

def ecoli_data_f():
    ecoli_data = pd.read_csv('ecoli_data.csv')
    ecoli_data.set_index("index", inplace=True)
    return ecoli_data

ecoli_data = ecoli_data_f()

def mixture_data_f(n_instances):
    a = np.random.normal(0, 1, n_instances)
    b = np.random.normal(3.2, np.sqrt(1.8), n_instances)

    a_negative = a < 0
    c = np.empty_like(a)
    c[a_negative] = -1.2 + 2.3*a[a_negative] + 1.5*b[a_negative] + np.random.normal(0, np.sqrt(0.5), a_negative.sum())
    c[~a_negative] = 2.3 + -0.7*a[~a_negative] + 3.0*b[~a_negative] + np.random.normal(0, np.sqrt(1), n_instances - a_negative.sum())

    d = c*0.67 + a*0.1 + 7.2 + np.random.normal(0, 0.5, n_instances)
    e = c + b + np.random.normal(0, 3, n_instances)

    data = pd.DataFrame({'a': a, 'b': b, 'c': c, 'd': d, 'e': e})
    return data

mixture_data = mixture_data_f(10000)

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

def test_ckde_results(variable, evidence, node_type):
    print("Configuration variable " + str(variable) + ", evidence " + str(evidence) + ", node_type " + str(node_type))
    print("------------------------------------------------------")

    iterate_test_results(evidence, node_type, variable)
    print()
    iterate_train_high_memory(evidence, node_type, variable)
    print()
    iterate_train_low_memory(evidence, node_type, variable)
    print()


def iterate_train_low_memory(evidence, node_type, variable):
    if not evidence:
        ckde_fun = CKDE_CPD._logdenominator_dataset
    elif all([node_type[e] == NodeType.GAUSSIAN for e in evidence]):
        ckde_fun = CKDE_CPD._logdenominator_dataset_onlygaussian
    elif all([node_type[e] == NodeType.CKDE for e in evidence]):
        ckde_fun = CKDE_CPD._logdenominator_dataset_onlykde
    else:
        ckde_fun = CKDE_CPD._logdenominator_dataset_mix

    print("Test > Train Low Memory")
    print("..............")
    train_dataset = mixture_data_f(100)
    train_dataset.index = range(2, 102)
    test_dataset = mixture_data_f(10000000)
    test_dataset.index = range(2, 10000002)
    ckde = MaximumLikelihoodEstimator.ckde_estimate_with_parents(variable, evidence, node_type,
                                                                 train_dataset[[variable] + evidence])
    start = time()
    py_result = ckde._logdenominator_dataset_python(test_dataset[:10])
    end = time()
    py_time = end - start
    print("Python implementation (extrapolation): " + str(py_time * 1000000))
    reorder_dataset = test_dataset[[variable] + evidence]
    start = time()
    rust_result = ckde_fun(ckde, reorder_dataset)
    end = time()
    rust_time = end - start
    print("Rust implementation: " + str(rust_time))
    print("Ratio: " + str((py_time * 1000000) / rust_time))
    print("Python result: " + str(py_result))
    if isinstance(rust_result, np.ndarray):
        rust_result = rust_result[:10].sum()
    print("Rust result: " + str(rust_result))
    assert np.isclose(py_result, rust_result), \
        "Wrong result for variable " + str(variable) + ", evidence " + str(evidence) + ", node_type " + str(node_type) + \
        " in Test > Train Low Memory"


def iterate_train_high_memory(evidence, node_type, variable):
    print("Test > Train High Memory")
    print("..............")
    train_dataset = mixture_data_f(100)
    train_dataset.index = range(2, 102)
    test_dataset = mixture_data_f(500)
    test_dataset.index = range(2, 502)
    ckde = MaximumLikelihoodEstimator.ckde_estimate_with_parents(variable, evidence, node_type,
                                                                 train_dataset[[variable] + evidence])
    start = time()
    py_result = ckde._logdenominator_dataset_python(test_dataset)
    end = time()
    py_time = end - start
    print("Python implementation: " + str(py_time))
    reorder_dataset = test_dataset[[variable] + evidence]
    start = time()
    rust_result = ckde._logdenominator_dataset(reorder_dataset)
    end = time()
    rust_time = end - start
    print("Rust implementation: " + str(rust_time))
    print("Ratio: " + str(py_time / rust_time))
    print("Python result: " + str(py_result))
    if isinstance(rust_result, np.ndarray):
        rust_result = rust_result.sum()
    print("Rust result: " + str(rust_result))
    assert np.isclose(py_result, rust_result), \
        "Wrong result for variable " + str(variable) + ", evidence " + str(evidence) + ", node_type " + str(node_type) + \
        " in Test > Train High Memory"


def iterate_test_results(evidence, node_type, variable):
    print("Train > test")
    print("..............")
    train_dataset = mixture_data_f(100000)
    train_dataset.index = range(2, 100002)
    test_dataset = mixture_data_f(1000)
    test_dataset.index = range(2, 1002)
    ckde = MaximumLikelihoodEstimator.ckde_estimate_with_parents(variable, evidence, node_type,
                                                                 train_dataset[[variable] + evidence])
    start = time()
    py_result = ckde._logdenominator_dataset_python(test_dataset)
    end = time()
    py_time = end - start
    print("Python implementation: " + str(py_time))
    reorder_dataset = test_dataset[[variable] + evidence]
    start = time()
    rust_result = ckde._logdenominator_dataset(reorder_dataset)
    end = time()
    rust_time = end - start
    print("Rust implementation: " + str(rust_time))
    print("Ratio: " + str(py_time / rust_time))
    print("Python result: " + str(py_result))
    if isinstance(rust_result, np.ndarray):
        rust_result = rust_result.sum()
    print("Rust result: " + str(rust_result))
    assert np.isclose(py_result, rust_result), \
        "Wrong result for variable " + str(variable) + ", evidence " + str(evidence) + ", node_type " + str(node_type) + \
        " in Train > Test"


def total_score_pl(graph, pl):
    """
    Computes the total score in the network. As the score method is decomposable. The total score is the sum of
    the local scores.
    :param graph: Graph to be evaluated.
    :return: Total score of the network.
    """
    total_score = 0
    for node in graph.nodes:
        parents = graph.get_parents(node)
        ls = pl.local_score(node, parents, graph.node_type[node], {p: graph.node_type[p] for p in parents})
        total_score += ls

    return total_score

if __name__ == '__main__':

    # test_ckde_results('c', [], {})
    # test_ckde_results('c', ['a'], {'a' : NodeType.GAUSSIAN})
    # test_ckde_results('c', ['a', 'b'], {'a': NodeType.GAUSSIAN, 'b': NodeType.GAUSSIAN})
    # test_ckde_results('c', ['a'], {'a' : NodeType.CKDE})
    # test_ckde_results('c', ['a', 'b'], {'a': NodeType.CKDE, 'b': NodeType.CKDE})
    # test_ckde_results('c', ['a', 'b'], {'a': NodeType.CKDE, 'b': NodeType.GAUSSIAN})
    # test_ckde_results('c', ['a', 'b'], {'a': NodeType.GAUSSIAN, 'b': NodeType.CKDE})

    scoring_method = PredictiveLikelihood(ecoli_data, k=2, seed=0)

    a = scoring_method.local_score('yceP', ['fixC', 'ibpB'], NodeType.CKDE, {'fixC': NodeType.GAUSSIAN, 'ibpB': NodeType.GAUSSIAN})
    # Out[33]: -930.5869349859206
    b = scoring_method.local_score('yceP', ['ibpB', 'fixC', ], NodeType.CKDE, {'fixC': NodeType.GAUSSIAN, 'ibpB': NodeType.GAUSSIAN})
    # Out[34]: -853.3000746577968

    print("a = " + str(a) + ", b = " + str(b))


    # start = HybridContinuousModel()
    # start.add_nodes_from(list(ecoli_data.columns.values))
    # start.node_type['eutG'] = NodeType.CKDE
    # start.node_type['yjbO'] = NodeType.CKDE
    # start.node_type['yfaD'] = NodeType.CKDE
    # start.node_type['yedE'] = NodeType.CKDE
    # start.node_type['b1191'] = NodeType.CKDE
    # start.node_type['icdA'] = NodeType.CKDE
    # start.node_type['gltA'] = NodeType.CKDE
    # start.node_type['lpdA'] = NodeType.CKDE
    # start.node_type['ygcE'] = NodeType.CKDE
    # start.node_type['yheI'] = NodeType.CKDE
    # start.node_type['aceB'] = NodeType.CKDE
    # start.node_type['lacA'] = NodeType.CKDE
    # start.node_type['yfiA'] = NodeType.CKDE
    # start.node_type['folK'] = NodeType.CKDE
    # start.node_type['dnaG'] = NodeType.CKDE
    # start.node_type['asnA'] = NodeType.CKDE
    # start.node_type['yceP'] = NodeType.CKDE
    # start.node_type['tnaA'] = NodeType.CKDE
    # start.node_type['cchB'] = NodeType.CKDE
    #
    # start.add_edge('eutG', 'ibpB')
    #
    # start.add_edge('ibpB', 'yfaD')
    # start.add_edge('ibpB', 'yceP')
    #
    # start.add_edge('yfaD', 'atpG')
    # start.add_edge('yfaD', 'flgD')
    # start.add_edge('yfaD', 'sucD')
    # start.add_edge('yfaD', 'fixC')
    #
    # start.add_edge('atpG', 'flgD')
    #
    # start.add_edge('flgD', 'sucD')
    #
    # start.add_edge('sucD', 'sucA')
    #
    # start.add_edge('yedE', 'yecO')
    #
    # start.add_edge('b1191', 'tnaA')
    # start.add_edge('b1191', 'fixC')
    # start.add_edge('b1191', 'folK')
    # start.add_edge('b1191', 'ygcE')
    # start.add_edge('b1191', 'icdA')
    #
    # start.add_edge('sucA', 'tnaA')
    # start.add_edge('sucA', 'yheI')
    # start.add_edge('sucA', 'icdA')
    # start.add_edge('sucA', 'dnaJ')
    # start.add_edge('sucA', 'gltA')
    # start.add_edge('sucA', 'ygcE')
    # start.add_edge('sucA', 'yhdM')
    #
    # start.add_edge('yecO', 'pspA')
    # start.add_edge('yecO', 'lpdA')
    #
    # start.add_edge('icdA', 'ygcE')
    # start.add_edge('icdA', 'asnA')
    #
    # start.add_edge('lpdA', 'cspG')
    # start.add_edge('lpdA', 'pspB')
    # start.add_edge('lpdA', 'nmpC')
    # start.add_edge('lpdA', 'yheI')
    #
    # start.add_edge('ygcE', 'yheI')
    # start.add_edge('ygcE', 'aceB')
    # start.add_edge('ygcE', 'lacA')
    # start.add_edge('ygcE', 'asnA')
    #
    # start.add_edge('cspG', 'yfiA')
    # start.add_edge('cspG', 'yaeM')
    #
    # start.add_edge('yheI', 'ftsJ')
    # start.add_edge('yheI', 'fixC')
    # start.add_edge('yheI', 'mopB')
    #
    # start.add_edge('lacA', 'lacY')
    # start.add_edge('lacA', 'nuoM')
    # start.add_edge('lacA', 'lacZ')
    #
    # start.add_edge('yfiA', 'cspA')
    #
    # start.add_edge('ftsJ', 'dnaK')
    #
    # start.add_edge('lacY', 'nuoM')
    #
    # start.add_edge('mopB', 'dnaG')
    #
    # start.add_edge('cspA', 'hupB')
    #
    # start.add_edge('dnaK', 'folK')
    #
    # start.add_edge('lacZ', 'b1583')
    #
    # start.add_edge('folK', 'ycgX')
    # start.add_edge('folK', 'b1963')
    #
    # start.add_edge('ycgX', 'fixC')
    #
    # start.add_edge('b1963', 'dnaG')
    # start.add_edge('b1963', 'asnA')
    #
    # start.add_edge('fixC', 'yceP')
    # start.add_edge('fixC', 'tnaA')
    # start.add_edge('fixC', 'ygbD')
    # start.add_edge('fixC', 'cchB')
    #
    # start.add_edge('dnaG', 'atpD')
    #
    # start.add_edge('yceP', 'b1583')
    #
    #
    # print("=======================")
    # print("=======================")
    #
    # db = ecoli_data.iloc[:1000]
    #
    # pl = PredictiveLikelihood(db, k=2, seed=0)
    # # pl = PredictiveLikelihood(mixture_data)
    #
    # hc = HybridCachedHillClimbing(db, scoring_method=pl)
    # # bn = hc.estimate(start=start)
    # bn = hc.estimate()
    #
    # # ghc = CachedHillClimbing(ecoli_data, scoring_method=pl)
    # # gbn = ghc.estimate()
    # #
    # to_bnlearn_str(bn)
    #
    # scores = []
    # max_int_value = np.iinfo(np.int32).max
    # for i in range(50):
    #     seed = np.random.randint(0, max_int_value)
    #     pl.change_seed(seed)
    #
    #     scores.append(total_score_pl(bn, pl))
    #
    # print("Other cross validation scores: " + str(scores))


    # pass

    # print("Score summary: ")
    # print("-------------------")
    # total_score_pl(gbn, pl)
    #
    #
    # max_int_value = np.iinfo(np.int32).max
    # seed = np.random.randint(0, max_int_value)
    # pl.change_seed(seed)
    #
    # print("Score summary (other seed): ")
    # print("-------------------")
    # total_score_pl(gbn, pl)