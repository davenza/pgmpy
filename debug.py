# import pandas as pd
# import numpy as np
# from pgmpy.estimators import HillClimbSearch, BicScore
#
#
# data = pd.DataFrame(np.random.randint(0, 5, size=(5000, 9)), columns=list('ABCDEFGHI'))
# # add 10th dependent variable
#
# data['J'] = data['A'] * data['B']
# est = HillClimbSearch(data, scoring_method=BicScore(data))
# best_model = est.estimate()
# sorted(best_model.nodes())
#
# best_model.edges()
#
# # search a model with restriction on the number of parents:
# est.estimate(max_indegree=1).edges()

# import pandas as pd
# import numpy as np
# from pgmpy.estimators import ConstraintBasedEstimator
#
# data = pd.DataFrame(np.random.randint(0, 5, size=(2500, 3)), columns=list('XYZ'))
# data['sum'] = data.sum(axis=1)
# # print(data)
# c = ConstraintBasedEstimator(data)
# model = c.estimate()
# print(model.edges())

# import pandas as pd
# from pgmpy.models import BayesianModel
# from pgmpy.estimators import MaximumLikelihoodEstimator
#
# data = pd.DataFrame(data={'A': [0, 0, 1], 'B': [0, 1, 0], 'C': [1, 1, 0]})
# model = BayesianModel([('A', 'C'), ('B', 'C')])
# cpd_A = MaximumLikelihoodEstimator(model, data).estimate_cpd('A')
# print(cpd_A)
#
# cpd_C = MaximumLikelihoodEstimator(model, data).estimate_cpd('C')
# print(cpd_C)

import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator

data = pd.DataFrame(data={'A': [0, 0, 1], 'B': [0, 1, 0], 'C': [1, 1, 0]})
model = BayesianModel([('A', 'C'), ('B', 'C')])
estimator = BayesianEstimator(model, data)
cpd_C = estimator.estimate_cpd('C', prior_type="dirichlet", pseudo_counts=[1, 2])
print(cpd_C)
