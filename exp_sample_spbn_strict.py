import exp_helper

from pgmpy.estimators import MaximumLikelihoodEstimator
import matplotlib.pyplot as plt


N_TRAIN = 10000
N_TEST = 1000

train_data = exp_helper.mixture_data_f(N_TRAIN)
test_data = exp_helper.mixture_data_f(N_TEST)


plt.scatter(train_data.loc[:, 'a'].to_numpy(), train_data.loc[:,'b'].to_numpy(),
            c=train_data.loc[:,'c'].to_numpy())





spbn = MaximumLikelihoodEstimator.spbn_strict_estimate_with_parents('c', ['a', 'b'], train_data)


sampled = spbn.sample(test_data.shape[0], test_data)
