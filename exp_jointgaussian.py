import exp_helper
import numpy as np
from pgmpy.factors.continuous import CKDE_CPD
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.factors.continuous import NodeType

N_TRAIN = 1000
N_TEST = 10

train_data = exp_helper.mixture_data_f(N_TRAIN)
test_data = exp_helper.mixture_data_f(N_TEST)

ckde = MaximumLikelihoodEstimator.ckde_estimate_with_parents('c', ['a', 'b'], {'a': NodeType.GAUSSIAN,
                                                                        'b': NodeType.GAUSSIAN}, train_data)

print("Conditional logpdf:")

truth_logpdf = ckde.logpdf_dataset(test_data)
print(truth_logpdf)

new_logpdf = ckde.cond_logpdf_dataset(test_data)
print(new_logpdf)


new_unilogpdf = ckde.conduni_logpdf_dataset(test_data)
print(new_unilogpdf)

print()
print("Joint logpdf:")
truth_logpdf = ckde.joint_logpdf_dataset(test_data)
print(truth_logpdf)

new_logpdf = ckde.cond_joint_logpdf_dataset(test_data)
print(new_logpdf)

new_unilogpdf = ckde.conduni_joint_logpdf_dataset(test_data)
print(new_unilogpdf)


# print("diff = " + str(truth_logpdf - new_logpdf))

# print("first instance:")
# print(str(test_data.iloc[0,:]))
# print()
# truth_g_logpdf = np.zeros((N_TEST,))
# for i, gaussian_cpd in enumerate(ckde.gaussian_cpds):
#     truth_g_logpdf += gaussian_cpd.logpdf_dataset(test_data)
# print("truth_g_logpdf = " + str(truth_g_logpdf))
#
# cond_g_logpdf = ckde.cond_gaussian_logpdf_dataset(test_data)
# print("cond_g_logpdf" + str(cond_g_logpdf))