import pandas as pd
import numpy as np
import exp_helper

from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.factors.continuous import NodeType
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde

N_TRAIN = 10000
N_TEST = 10

train_data = exp_helper.mixture_data_f(N_TRAIN)
test_data = exp_helper.mixture_data_f(N_TEST)

a_value = 12.5
b_value = 0

plt.scatter(train_data.loc[:, 'a'].to_numpy(), train_data.loc[:,'b'].to_numpy(),
            c=train_data.loc[:,'c'].to_numpy())

plt.scatter(a_value, b_value, c="red")



ckde = MaximumLikelihoodEstimator.ckde_estimate_with_parents('c', ['a', 'b'], {'a': NodeType.GAUSSIAN,
                                                                               'b': NodeType.GAUSSIAN}, train_data)

domain = np.linspace(-3, 10, 2000)

plt.figure()

parent_values = pd.DataFrame({'a': [a_value]*10000, 'b': [b_value]*10000})

sampled = ckde.sample(10000, parent_values)

k = gaussian_kde(sampled)


plt.plot(np.linspace(-3, 10, 2000),
         exp_helper.mixture_data_cond_distribution(np.linspace(-3, 10, 2000), a_value, b_value))


parent_values_serie = pd.Series([a_value, b_value], index=['a', 'b'])

plt.plot(np.linspace(-3, 10, 2000),
         ckde.sample_distribution(np.linspace(-3, 10, 2000), parent_values_serie))

plt.plot(np.linspace(-3, 10, 2000),
         k(np.linspace(-3, 10, 2000)))
plt.show()


