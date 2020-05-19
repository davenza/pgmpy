import numpy as np
import pandas as pd
from pgmpy.models import HybridContinuousModel
from pgmpy.factors.continuous import NodeType, CKDE_CPD, LinearGaussianCPD
from pgmpy.estimators import HybridCachedHillClimbing, ValidationLikelihood
from pgmpy.estimators.callbacks import SaveModel, DrawModel
import exp_helper

np.random.seed(0)

N_MODEL = 1000

model = HybridContinuousModel()
model.add_edges_from([('a', 'c'), ('b', 'c'), ('c', 'd'), ('d', 'e')], node_type={'a': NodeType.GAUSSIAN,
                                                                                        'b': NodeType.SPBN,
                                                                                        'c': NodeType.SPBN,
                                                                                        'd': NodeType.GAUSSIAN,
                                                                                        'e': NodeType.SPBN
                                                                                        })

a_cpd = LinearGaussianCPD('a', [0.5], 1, evidence=[])

b_instances = exp_helper.sample_mixture([0.5, 0.5], [-2, 2], [1, 1], N_MODEL)
b_instances_df = pd.DataFrame({'b': b_instances})
b_cpd = CKDE_CPD('b', gaussian_cpds=[], kde_instances=b_instances_df, evidence=[])

c_gaussiancpd_a = LinearGaussianCPD('a', [-2.0, 1, 0], 2, evidence=['c', 'b'])
c_instances = exp_helper.sample_multivariate_mixture([0.7, 0.3], [[-1, -1], [1, 1]],
                                                     [
                                                         [[1, 1],
                                                          [1, 1]],
                                                         [[2, -0.5],
                                                          [-0.5, 0.5]]
                                                     ], N_MODEL)
c_instances_df = pd.DataFrame(c_instances, columns=['c', 'b'])
c_cpd = CKDE_CPD('c', gaussian_cpds=[c_gaussiancpd_a], kde_instances=c_instances_df,
                 evidence=['a', 'b'], evidence_type={'a': NodeType.GAUSSIAN, 'b': NodeType.SPBN})

d_cpd = LinearGaussianCPD('d', [-2.1, -0.6], 1.5, evidence=['c'])

e_gaussian_cpd_d = LinearGaussianCPD('d', [0, 1.2], 0.2, evidence=['e'])
e_instances = exp_helper.sample_mixture([0.2, 0.4, 0.4], [-3, 1, 6], [1, 2, 1], N_MODEL)
e_instances_df = pd.DataFrame({'e': e_instances})
e_cpd = CKDE_CPD('e', gaussian_cpds=[e_gaussian_cpd_d], kde_instances=e_instances_df,
                 evidence=['d'], evidence_type={'d': NodeType.GAUSSIAN})

model.add_cpds(a_cpd, b_cpd, c_cpd, d_cpd, e_cpd)

dataset = model.sample_dataset(10000)
print(dataset)


vl = ValidationLikelihood(dataset)

save_model = SaveModel('iterations/')
draw_model = DrawModel('iterations/')

hc = HybridCachedHillClimbing(dataset, scoring_method=vl)
trained_model = hc.estimate(patience=5, callbacks=[save_model, draw_model])

print("Actual model: ")

total_score = 0
validation_score = 0
for n in model.nodes:
    total_score += vl.local_score(n, model.get_parents(n), model.node_type[n], model.node_type)
    validation_score += vl.validation_local_score(n, model.get_parents(n), model.node_type[n], model.node_type)

print("Score: " + str(total_score))
print("Validation Score: " + str(validation_score))

print()
print("Trained model:")

total_score = 0
validation_score = 0
for n in model.nodes:
    total_score += vl.local_score(n, model.get_parents(n), model.node_type[n], model.node_type)
    validation_score += vl.validation_local_score(n, model.get_parents(n), model.node_type[n], model.node_type)

print("Score: " + str(total_score))
print("Validation Score: " + str(validation_score))