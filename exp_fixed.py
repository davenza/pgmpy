import numpy as np
np.random.seed(1)

import exp_helper

if __name__ == '__main__':


    gaussian_variables = ['a', 'b']
    kde_variable = 'c'

    for c_value in [3, -0.7, 50, -20, -2.5781]:
        print("c = " + str(c_value))
        print("===================")
        c_dataset = exp_helper.mixture_data_f_c_value(1000, c_value)
        c_dataset_test = exp_helper.mixture_data_f_c_value(3, c_value)

        grs = exp_helper.learn_regressions(c_dataset[gaussian_variables], gaussian_variables, None)
        print(grs[0])
        print(grs[1])

        gr_covariance = exp_helper.regression_covariance(grs, gaussian_variables)

        print()
        print("MLE covariance: " + str(c_dataset[gaussian_variables].cov().to_numpy()))
        print("Least squares covariance: " + str(gr_covariance))
        print()


