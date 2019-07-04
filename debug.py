import numpy as np
import pandas as pd

from pgmpy.models import LinearGaussianBayesianNetwork
from pgmpy.factors.continuous import LinearGaussianCPD

# Example data a -> c <- b
a = np.random.normal(0, 1, 5000)
b = np.random.normal(3.2, np.sqrt(1.8), 5000)
c = -1.2 + 2.3*a + 1.5*b + np.random.normal(0, np.sqrt(0.5), 5000)

parents = sorted(['a', 'b'])
node = 'c'
data = pd.DataFrame( {'a': a, 'b': b, 'c': c})
data_train = data.iloc[:3000]
data_predict = data.drop('c', axis=1, inplace=False).iloc[3000:]




if __name__ == '__main__':
    model = LinearGaussianBayesianNetwork([('a', 'c'), ('b', 'c')])
    cpd1 = LinearGaussianCPD('a', [1], 4)
    cpd2 = LinearGaussianCPD('b', [-5], 4)
    cpd3 = LinearGaussianCPD('c', [4, -1, 7], 3, ['a', 'b'])
    model.fit(data_train)

    for cpd in model.get_cpds():
        print(cpd)

    print(model.to_joint_gaussian().covariance)
    prediction = model.predict(data_predict)
    print(prediction)
    print((data['c'].iloc[3000:] - prediction['c']).var())