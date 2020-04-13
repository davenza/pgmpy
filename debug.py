import numpy as np
# np.seterr(all='raise')
np.random.seed(0)
import pandas as pd

from pgmpy.estimators import MaximumLikelihoodEstimator, CachedHillClimbing, HybridCachedHillClimbing, CVPredictiveLikelihood, ValidationLikelihood, GaussianValidationLikelihood
from pgmpy.estimators.callbacks import DrawModel, SaveModel
from pgmpy.factors.continuous import NodeType, CKDE_CPD
from pgmpy.models import HybridContinuousModel, BayesianModel
from pgmpy.estimators import GaussianBicScore, MaximumLikelihoodEstimator

from time import time
from sklearn.model_selection import KFold

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
    # if not evidence:
    #     ckde_fun = CKDE_CPD._logdenominator_dataset
    # elif all([node_type[e] == NodeType.GAUSSIAN for e in evidence]):
    #     ckde_fun = CKDE_CPD._logdenominator_dataset_onlygaussian
    # elif all([node_type[e] == NodeType.CKDE for e in evidence]):
    #     ckde_fun = CKDE_CPD._logdenominator_dataset_onlykde
    # else:
    #     ckde_fun = CKDE_CPD._logdenominator_dataset_mix

    ckde_fun = CKDE_CPD._logdenominator_dataset

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
    print("Python result: " + str(py_result.sum()))
    if isinstance(rust_result, np.ndarray):
        rust_result = rust_result[:10]
    print("Rust result: " + str(rust_result.sum()))
    assert np.all(np.isclose(py_result, rust_result)), \
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
    print("Python result: " + str(py_result.sum()))
    if isinstance(rust_result, np.ndarray):
        rust_result = rust_result
    print("Rust result: " + str(rust_result.sum()))
    assert np.all(np.isclose(py_result, rust_result)), \
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
    print("Python result: " + str(py_result.sum()))
    if isinstance(rust_result, np.ndarray):
        rust_result = rust_result
    print("Rust result: " + str(rust_result.sum()))
    assert np.all(np.isclose(py_result, rust_result)), \
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

def remove_crossvalidated_nan(train_data, folds):

    to_delete = set()
    for k in folds:
        vl = ValidationLikelihood(train_data, k=k, seed=0)
        for (train_indices, _) in vl.fold_indices:
            cv_data = train_data.iloc[train_indices,:]
            d = cv_data.columns[cv_data.var() == 0].tolist()
            to_delete.update(d)

    return to_delete

def linear_dependent_features(dataset):
    to_remove = set()
    rank = np.linalg.matrix_rank(dataset)
    min_dim = min(dataset.shape[0], dataset.shape[1])

    tmp_dataset = dataset.copy()
    for i in range(min_dim - rank):
        for column in tmp_dataset.columns:
            cpy = tmp_dataset.drop(column, axis=1)
            if np.linalg.matrix_rank(cpy) == rank:
                to_remove.add(column)
                tmp_dataset = cpy
                break

    return to_remove

ctg = pd.read_csv("CTG.csv")
if "class" in ctg.columns:
    ctg = ctg.drop("class", axis=1)
ctg = ctg.astype('float64')

to_remove_features = remove_crossvalidated_nan(ctg, [2, 3, 5, 10])
ctg = ctg.drop(to_remove_features, axis=1)

dependent_features = linear_dependent_features(ctg)
ctg = ctg.drop(dependent_features, axis=1)

blocks = pd.read_csv("page-blocks.csv")
if "class" in blocks.columns:
    blocks = blocks.drop("class", axis=1)
blocks = blocks.astype('float64')

to_remove_features = remove_crossvalidated_nan(blocks, [2, 3, 5, 10])
blocks = blocks.drop(to_remove_features, axis=1)

dependent_features = linear_dependent_features(blocks)
blocks = blocks.drop(dependent_features, axis=1)

if __name__ == '__main__':



    # test_ckde_results('c', [], {})
    # test_ckde_results('c', ['a'], {'a': NodeType.GAUSSIAN})
    # test_ckde_results('c', ['a', 'b'], {'a': NodeType.GAUSSIAN, 'b': NodeType.GAUSSIAN})
    # test_ckde_results('c', ['b', 'a'], {'b': NodeType.GAUSSIAN, 'a': NodeType.GAUSSIAN})
    # test_ckde_results('c', ['a'], {'a': NodeType.CKDE})
    # test_ckde_results('c', ['a', 'b'], {'a': NodeType.CKDE, 'b': NodeType.CKDE})
    # test_ckde_results('c', ['b', 'a'], {'b': NodeType.CKDE, 'a': NodeType.CKDE})
    # test_ckde_results('c', ['a', 'b'], {'a': NodeType.CKDE, 'b': NodeType.GAUSSIAN})
    # test_ckde_results('c', ['b', 'a'], {'b': NodeType.GAUSSIAN, 'a': NodeType.CKDE})
    test_ckde_results('c', ['a', 'b'], {'a': NodeType.GAUSSIAN, 'b': NodeType.CKDE})
    # test_ckde_results('c', ['b', 'a'], {'b': NodeType.CKDE, 'a': NodeType.GAUSSIAN})

    # print("all blocks = " + str(len(blocks)))


    # folds = list(KFold(10, shuffle=True, random_state=0).split(blocks))
    #
    # fold0_train, fold0_test = folds[0]
    # train_dataset = blocks.iloc[fold0_train,:]
    # # print("train dataset = " + str(len(train_dataset)))
    # vl = ValidationLikelihood(train_dataset, k=2, seed=0)
    #
    # k = vl.local_score('blackpix', ['length'], NodeType.CKDE, {'length': NodeType.GAUSSIAN})
    #
    # i = 0
    # while True:
    #     i += 1
    #     print("\r{}".format(i), end='')
    #     vl2 = ValidationLikelihood(train_dataset, k=2, seed=0)
    #
    #     k2 = vl2.local_score('blackpix', ['length'], NodeType.CKDE, {'length': NodeType.GAUSSIAN})
    #
    #     if np.any(~np.isclose(k, k2)):
    #         import os
    #         os.system('play -nq -t alsa synth {} sine {}'.format(1, 400))


    #
    # folds = list(KFold(10, shuffle=True, random_state=0).split(blocks))
    #
    # fold0_train, fold0_test = folds[0]
    # train_dataset = blocks.iloc[fold0_train,:]
    # # print("train dataset = " + str(len(train_dataset)))
    # vl = ValidationLikelihood(train_dataset, k=2, seed=0)
    #
    # k = vl.local_score('blackpix', ['length', 'eccen', 'p_black', 'p_and'], NodeType.CKDE, {'length': NodeType.GAUSSIAN,
    #                                                                             'eccen': NodeType.CKDE,
    #                                                                             'p_black': NodeType.GAUSSIAN,
    #                                                                             'p_and': NodeType.GAUSSIAN})
    #
    # i = 0
    #
    # vl2 = ValidationLikelihood(train_dataset, k=2, seed=0)
    #
    # k2 = vl2.local_score('blackpix', ['p_black', 'p_and', 'length', 'eccen'], NodeType.CKDE, {'p_black': NodeType.GAUSSIAN,
    #                                                                                  'p_and': NodeType.GAUSSIAN,
    #                                                                                  'length': NodeType.GAUSSIAN,
    #                                                                                  'eccen': NodeType.CKDE})
    #
    # vl3 = ValidationLikelihood(train_dataset, k=2, seed=0)
    # k3 = vl2.local_score('blackpix', ['eccen', 'p_and', 'p_black', 'length'], NodeType.CKDE, {'eccen': NodeType.CKDE,
    #                                                                                  'p_and': NodeType.GAUSSIAN,
    #                                                                                  'p_black': NodeType.GAUSSIAN,
    #                                                                                  'length': NodeType.GAUSSIAN})
    #
    # print("k = " + str(k))
    # print("k2 = " + str(k2))
    # print("k3 = " + str(k3))
    #
    #









    # np = vl.local_score('Nzeros', [], NodeType.CKDE, {})
    # p = vl.local_score('Nzeros', ['ASTV'], NodeType.CKDE, {'ASTV': NodeType.GAUSSIAN})
    # print("Delta: " + str(p - np))
    # print(np)
    # print(p)

    # hc = HybridCachedHillClimbing(train_dataset, scoring_method=vl)

    # start = HybridContinuousModel.load_model('iterations/000080.pkl')
    # pl = ValidationLikelihood(ecoli_data, k=10, seed=0)
    # # pl = CVPredictiveLikelihood(ecoli_data, k=10, seed=0)
    # hc = HybridCachedHillClimbing(ecoli_data, scoring_method=pl)
    # cb_draw = DrawModel('iterations')
    # cb_save = SaveModel('iterations')
    # # start = HybridContinuousModel.load_model('iterations/000099.pkl')
    # bn = hc.estimate(callbacks=[cb_draw, cb_save], significant_threshold=np.inf)

    # p = {'a': NodeType.CKDE, 'b': NodeType.CKDE, 'c': NodeType.GAUSSIAN}
    # a = MaximumLikelihoodEstimator.ckde_estimate_with_parents('a', ['b', 'c'], p, mixture_data)
    #
    # inst = pd.DataFrame({'a': [2.3, -0.5, 1.7], 'b': [0.3, 1.2, -3.1], 'c': [-0.3, 2.2, 0.9]})
    # log = a.logpdf_dataset(inst)
    # while True:
    #     if np.any(log != a.logpdf_dataset(inst)):
    #         print("Error")
    #         input()

    # cv_indices = list(KFold(10, shuffle=True, random_state=0).split(blocks))
    #
    # parents = ["blackpix", "wb_trans", "p_black", "p_and", "eccen", "area"]
    # parents_new = ["blackpix", "wb_trans", "p_black", "p_and", "eccen", "area", "length"]
    #
    # train_blocks = blocks.iloc[cv_indices[7][0],:]
    #
    # pl = CVPredictiveLikelihood(train_blocks, k=2, seed=924231285)
    #
    # node_type = {'mean_tr': NodeType.GAUSSIAN, "area": NodeType.CKDE, "length": NodeType.CKDE,
    #              "wb_trans": NodeType.CKDE, "eccen": NodeType.CKDE, "blackpix": NodeType.CKDE,
    #              "p_and": NodeType.CKDE, "p_black": NodeType.CKDE, "height": NodeType.GAUSSIAN,
    #              "blackand": NodeType.CKDE}
    #
    # score = pl.local_score("blackand", parents, node_type["blackand"], node_type)
    # new_score = pl.local_score("blackand", parents_new, node_type["blackand"], node_type)
    #
    # print(str(new_score - score))
    # # print(pl.fold_indices)