import numpy as np
np.random.seed(1)
import pandas as pd

from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import LinearGaussianBayesianNetwork

import scipy.stats as stats
from scipy.linalg import solve_triangular

def sample_mixture(prior_prob, means, variances, n_instances):
    p = np.asarray(prior_prob)
    c = np.cumsum(p)
    m = np.asarray(means)
    v = np.asarray(variances)

    s = np.random.uniform(size=n_instances)

    digitize = np.digitize(s, c)

    res = np.random.normal(m[digitize], np.sqrt(v[digitize]))

    return res


def mixture_data_f(n_instances):
    c = sample_mixture([0.2, 0.15, 0.3, 0.35], [-0.7, 0, 2.3, 7.8], [1.8, 1, 0.5, 0.8], n_instances)
    # c_value = 54

    a = np.random.normal(2.3 + 1.2*c, 0.5, size=n_instances)
    b = np.random.normal(-0.7 - 0.3*a + 0.6*c, 0.9, size=n_instances)

    data = pd.DataFrame({'a': a, 'b': b, 'c': c})
    return data

mixture_data = mixture_data_f(1000)
test_mixture_data = mixture_data_f(3)

def mixture_data_f_c_value(n_instances, c_value):
    c = np.full((n_instances,), c_value)

    a = np.random.normal(2.3 + 1.2*c, 0.5, size=n_instances)
    b = np.random.normal(-0.7 - 0.3*a + 0.6*c, 0.9, size=n_instances)

    data = pd.DataFrame({'a': a, 'b': b, 'c': c})
    return data

def learn_regressions(data, gaussians, starting_parents=None):

    if starting_parents is None:
        parents = []
    else:
        parents = [starting_parents]

    gaussian_cpds = []
    for g in gaussians:
        cpd = MaximumLikelihoodEstimator.gaussian_estimate_with_parents(g, parents, data)
        gaussian_cpds.append(cpd)
        parents = parents.copy()
        parents.append(g)

    return gaussian_cpds

def logpdf_regression(data, regressions):
    lpdf = np.zeros(data.shape[0])
    for gr in regressions:
        lpdf += gr.logpdf_dataset(data)

    return lpdf

def denominator_ckde(data, ckde):
    return ckde._logdenominator_dataset(data)

def denominator_mvn(data, training_data, regressions, gaussian_variables, kde_variable, h):
    den = np.zeros(data.shape[0])

    means = np.zeros((training_data.shape[0], len(regressions)))

    for (i,(_,row)) in enumerate(data.iterrows()):


        for j, (_, training_row) in enumerate(training_data.iterrows()):
            grks = [gr.reduce([(kde_variable, training_row[kde_variable])], inplace=False) for gr in regressions]

            edges = [(e, g.variable) for g in grks for e in g.evidence]
            lgn = LinearGaussianBayesianNetwork(edges)

            lgn.add_cpds(grks)

            gaussian_distribution = lgn.to_joint_gaussian()

            means[j] = np.squeeze(gaussian_distribution.mean)
            covariance = gaussian_distribution.covariance


        gaussian_values = row[gaussian_variables]

        f_values = stats.multivariate_normal(gaussian_values, covariance).pdf(means)
        g_values = stats.norm(0, h).pdf(training_data[kde_variable])

        den[i] = np.log(f_values.sum() * g_values.sum()) - np.log(training_data.shape[0])

    return den

def logpdf_mvn(data, regressions, gaussian_variables, kde_variable):
    lpdf = np.zeros(data.shape[0])

    for (i,(_,row)) in enumerate(data.iterrows()):
        grks = [gr.reduce([(kde_variable, row[kde_variable])], inplace=False) for gr in regressions]

        edges = [(e, g.variable) for g in grks for e in g.evidence]
        lgn = LinearGaussianBayesianNetwork(edges)

        lgn.add_cpds(grks)

        gaussian_distribution = lgn.to_joint_gaussian()
        # print("Mean = " + str(np.squeeze(gaussian_distribution.mean)))
        # print("Covariance = " + str(gaussian_distribution.covariance))
        lpdf[i] = stats.multivariate_normal(np.squeeze(gaussian_distribution.mean), gaussian_distribution.covariance). \
            logpdf(row[gaussian_variables])

    return lpdf

def logpdf_cte(data, mean, covariance, gammas, gaussian_variables, kde_variable):
    lpdf = np.zeros(data.shape[0])

    chol = np.linalg.cholesky(covariance)

    cte = -0.5*len(gaussian_variables)*(np.log(2*np.pi)) -\
          np.log(np.diag(chol)).sum()


    for (i,(_,row)) in enumerate(data.iterrows()):
        k = row[kde_variable]

        g_diff = row[gaussian_variables] - mean
        gamma_vec = gammas * k

        g_solve = solve_triangular(chol, g_diff)
        gamma_solve = solve_triangular(chol, gamma_vec)

        # coeff = -0.5 * (np.sum(g_solve*g_solve) - np.sum(gamma_solve*gamma_solve))

        lpdf[i] = -0.5 * (np.sum(g_solve*g_solve) + np.sum(gamma_solve*gamma_solve))
        # lpdf[i] = -0.5 * (np.sum(g_solve*g_solve) + np.sum(gamma_solve*gamma_solve))

    return lpdf + cte


def regression_covariance(regressions, gaussian_variables):
    d = np.eye(len(regressions))
    u = np.eye(len(regressions))

    for i, gaussian in enumerate(gaussian_variables):
        for gr in regressions:
            if gr.variable == gaussian:
                d[i,i] = 1/gr.variance

                for j, gaussian_evidence in enumerate(gaussian_variables):
                    if gaussian_evidence in gr.evidence:
                        u[j,i] = -gr.beta[gr.evidence.index(gaussian_evidence)+1]

    cov = np.linalg.inv(u.dot(d).dot(u.T))

    return cov

def regression_means(regressions, gaussian_variables, evidence):
    means = np.zeros((evidence.shape[0], len(gaussian_variables),))

    for i, gr in enumerate(regressions):
        means[:, i] = gr.beta[0]

        for e in gr.evidence:

            if e in evidence.columns:
                means[:, i] += gr.beta[gr.evidence.index(e)+1]*evidence.loc[:,e]
            else:
                means[:, i] += gr.beta[gr.evidence.index(e)+1]*means[:, gaussian_variables.index(e)]

    return means

def regression_means_cte(regressions, gaussian_variables, evidence):
    means = np.zeros((len(gaussian_variables),))

    for i, gr in enumerate(regressions):
        means[i] = gr.beta[0]

        for e in gr.evidence:

            if e in evidence.columns:
                continue
            else:
                means[i] += gr.beta[gr.evidence.index(e)+1]*means[gaussian_variables.index(e)]

    return means