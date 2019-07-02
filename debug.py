import numpy as np
import pandas as pd

from pgmpy.models import LinearGaussianBayesianNetwork

# Example data a -> c <- b
a = np.random.normal(0, 1, 5000)
b = np.random.normal(3.2, np.sqrt(1.8), 5000)
c = -1.2 + 2.3*a + 1.5*b + np.random.normal(0, np.sqrt(0.5), 5000)

parents = sorted(['a', 'b'])
node = 'c'
data = pd.DataFrame( {'a': a, 'b': b, 'c': c})

def cpd_cov(data):
    cov = data.cov()

    betas = np.empty((3,))

    sigma_xx_inv = np.linalg.inv(cov.loc[parents, parents])
    # sigma_xx_inv = pd.DataFrame(np.linalg.inv(cov), cov.columns, cov.index)
    # sigma_xx_inv = sigma_xx_inv.loc[parents, parents]
    betas[0] = data[node].mean() - np.dot(cov.loc[node, parents], sigma_xx_inv).dot(data[parents].mean())
    betas[1:] = np.dot(sigma_xx_inv, cov.loc[node, parents])
    sigma = cov.loc[node, node] - np.dot(cov.loc[node, parents], sigma_xx_inv).dot(cov.loc[parents, node])

    return betas, sigma
    # print("cpd_cov():")
    # print("Betas: " + str(betas))
    # print("sigma: " + str(sigma))

def cpd_lstqr(data):

    lindata = np.column_stack((np.ones(data.shape[0]), data[parents].values))
    (betas, res, _, _) = np.linalg.lstsq(lindata, data[node])
    sigma = res / (data.shape[0] -1)

    return betas, sigma
    # print("cpd_lstqr():")
    # print("Betas: " + str(betas))
    # print("sigma: " + str(sigma))


if __name__ == '__main__':
    model = LinearGaussianBayesianNetwork([('a', 'c'), ['b', 'c']])
    model.fit(data)
    for cpd in model.get_cpds():
        print(cpd)