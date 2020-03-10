import numpy as np
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.factors.continuous import NodeType
import exp_helper


if __name__ == '__main__':

    gaussian_variables = ['a', 'b']
    kde_variable = 'c'

    mixture_data = exp_helper.mixture_data_f(10000)
    test_mixture_data = exp_helper.mixture_data_f(3)

    ckde = MaximumLikelihoodEstimator.ckde_estimate_with_parents('c', ['a', 'b'], {'a': NodeType.GAUSSIAN,
                                                                                   'b': NodeType.GAUSSIAN}, mixture_data)
    grs = exp_helper.learn_regressions(mixture_data, gaussian_variables, kde_variable)


    for gr in grs:
        print(gr)

    print("udu = " + str(exp_helper.regression_covariance(grs, gaussian_variables)))

    print("--------------")

    print("Regress")
    print("logpdf "+ str(exp_helper.logpdf_regression(test_mixture_data, grs)))
    print()
    print("MVN")
    print("logpdf "+ str(exp_helper.logpdf_mvn(test_mixture_data, grs, gaussian_variables, kde_variable)))


    means_cte = exp_helper.regression_means_cte(grs, gaussian_variables, mixture_data.loc[:, kde_variable].to_frame())
    gammas = np.asarray([gr.beta[gr.evidence.index(kde_variable)+1] for gr in grs])

    covariance = exp_helper.regression_covariance(grs, gaussian_variables)

    print()
    print("MVN cte")
    print("logpdf " + str(exp_helper.logpdf_cte(test_mixture_data, means_cte, covariance, gammas, gaussian_variables, kde_variable)))
