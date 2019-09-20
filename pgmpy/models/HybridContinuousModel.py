from pgmpy.models import BayesianModel

class LinearGaussianBayesianNetwork(BayesianModel):

    def __init__(self, ebunch=None):
        super(LinearGaussianBayesianNetwork, self).__init__()
        if ebunch:
            self.add_edges_from(ebunch)
        self.cpds = []