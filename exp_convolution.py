import numpy as np
np.random.seed(0)
import pandas as pd

from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.factors.continuous import NodeType

import scipy.stats as stats
from scipy.linalg import solve_triangular
from scipy.special import logsumexp



import exp_helper


def logsum_g(train_data, test_instance, cholesky_cov, grs, gaussian_variables, kde_variable):

    gaussian_values = test_instance[gaussian_variables].to_numpy()
    means = exp_helper.regression_means(grs, gaussian_variables, train_data.loc[:,kde_variable].to_frame())

    difference = gaussian_values - means

    y = solve_triangular(cholesky_cov, difference.T, lower=True)

    coeffs = -0.5*np.sum(y*y, axis=0)

    return logsumexp(coeffs)


def logsum_h(train_data, kde_variable, h):

    var = train_data.loc[:,kde_variable].to_numpy()
    coeffs = -0.5*var*var / (h**2)
    return logsumexp(coeffs)



def logden_convolution(train_data, test_data, grs, gaussian_variables, kde_variable, h):

    covariance = exp_helper.regression_covariance(grs, gaussian_variables)
    cholesky_cov = np.linalg.cholesky(covariance)


    cte = -np.log(train_data.shape[0]) - \
          0.5*(len(gaussian_variables)+1)*np.log(2*np.pi) - \
          0.5*np.log(np.linalg.det(covariance)) - \
          np.log(h)

    res = np.full((test_data.shape[0],), cte)
    for i, (_, instance) in enumerate(test_data.iterrows()):
        res[i] += logsum_g(train_data, instance, cholesky_cov, grs, gaussian_variables, kde_variable)
        res[i] += logsum_h(train_data, kde_variable, h)

    return res

def logden_convolution_cte(train_data, test_data, grs, gaussian_variables, kde_variable, h):

    covariance = exp_helper.regression_covariance(grs, gaussian_variables)
    means = exp_helper.regression_means(grs, gaussian_variables, train_data.loc[:, kde_variable].to_frame())

    res = np.zeros((test_data.shape[0],))

    for i, (_, instance) in enumerate(test_data.iterrows()):
        gaussian_values = instance[gaussian_variables].to_numpy()

        g_sum = stats.multivariate_normal(gaussian_values, covariance).pdf(means).sum()
        h_sum = stats.norm(0, h).pdf(train_data.loc[:, kde_variable].to_numpy()).sum()

        res[i] = np.log(g_sum) + np.log(h_sum)

    res -= np.log(train_data.shape[0])


    return res



if __name__ == '__main__':
    gaussian_variables = ['a', 'b']
    kde_variable = 'c'

    data = exp_helper.mixture_data_f(10)
    data_test = exp_helper.mixture_data_f(3)
    data_test = data_test.append([{'a': data_test.loc[0, 'a'], 'b': data_test.loc[0, 'b'], 'c': -5}])
    print(data_test)


    grs = exp_helper.learn_regressions(data, ['a', 'b'], starting_parents='c')


    ckde = MaximumLikelihoodEstimator.ckde_estimate_with_parents(kde_variable,
                                                                 gaussian_variables,
                                                                 {
                                                                     'a': NodeType.GAUSSIAN,
                                                                     'b': NodeType.GAUSSIAN
                                                                  },
                                                                 data
                                                                )
    logden_py = ckde._logdenominator_dataset_python(data_test)
    print("Logdenominator Python " + str(logden_py))

    logden = ckde._logdenominator_dataset(data_test.loc[:, [kde_variable] + gaussian_variables])
    print("Logdenominator OpenCL " + str(logden))


    h = np.sqrt(ckde.covariance[0,0])

    logden_conv = logden_convolution(data, data_test, grs, gaussian_variables, kde_variable, h)
    print("Logdenominator conv" + str(logden_conv))
    logden_conv_cte = logden_convolution_cte(data, data_test, grs, gaussian_variables, kde_variable, h)
    print("Logdenominator conv cte" + str(logden_conv_cte))


    print("Difference: " + str(logden_conv - logden))
