from pgmpy.estimators.callbacks import Callback

from pgmpy.estimators import ValidationLikelihood, CVPredictiveLikelihood, GaussianValidationLikelihood
from pgmpy.models import HybridContinuousModel
from pgmpy.factors.continuous import NodeType
import networkx as nx
import subprocess

class DrawModel(Callback):

    def __init__(self, folder):
        self.folder = folder

    def call(self, model, operation, scoring_method, iter_no):
        model_copy = model.copy()

        model_copy.graph["overlap"] = "scale"
        model_copy.graph["splines"] = "true"

        total_score = 0

        if isinstance(scoring_method, (CVPredictiveLikelihood, ValidationLikelihood)):
            sc = lambda n, p: scoring_method.local_score(n, p, model.node_type[n], model.node_type)
        else:
            sc = scoring_method

        for node in model.nodes:
            parents = model.get_parents(node)
            total_score += sc(node, parents)

        if isinstance(model, (HybridContinuousModel)):
            for n in model_copy.nodes:
                if model_copy.node_type[n] == NodeType.CKDE:
                    model_copy.nodes[n]['style'] = 'filled'
                    model_copy.nodes[n]['fillcolor'] = 'gray'

        if operation is not None:
            op, source, dest, score = operation

            if op == '+':
                model_copy.edges[source, dest]['color'] = 'green3'
                model_copy.edges[source, dest]['label'] = "{:0.3f}".format(score)
            elif op == '-':
                model_copy.add_edge(source, dest)
                model_copy.edges[source, dest]['color'] = 'firebrick1'
                model_copy.edges[source, dest]['label'] = "{:0.3f}".format(score)
            elif operation == 'flip':
                model_copy.edges[dest, source]['color'] = 'dodgerblue'
                model_copy.edges[dest, source]['label'] = "{:0.3f}".format(score)
            elif operation == 'type':
                model_copy.nodes[source]['style'] = 'filled'
                model_copy.nodes[source]['label'] = "{}\n{:0.3f}".format(source, score)

                if dest == NodeType.CKDE:
                    model_copy.nodes[source]['fillcolor'] = 'darkolivegreen1'
                elif dest == NodeType.GAUSSIAN:
                    model_copy.nodes[source]['fillcolor'] = 'yellow'

        title = "{}\nScore {:0.3f}".format(type(scoring_method).__name__, total_score)
        if isinstance(scoring_method, (ValidationLikelihood, GaussianValidationLikelihood)):
            val_score = 0
            for node in model.nodes:
                parents = model.get_parents(node)
                val_score += scoring_method.validation_local_score(node, parents, model.node_type[node], model.node_type)

            title += "\nValidation Score: {:0.3f}".format(val_score)

        A = nx.nx_agraph.to_agraph(model_copy)
        A.graph_attr.update(label=title, labelloc="t", fontsize='25')
        A.write('{}/{:06d}.dot'.format(self.folder, iter_no))
        A.clear()

        subprocess.run(["dot", "-Tpdf", "{}/{:06d}.dot".format(self.folder, iter_no), "-o",
                        "{}/{:06d}.pdf".format(self.folder, iter_no)])


