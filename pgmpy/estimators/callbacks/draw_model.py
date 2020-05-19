from pgmpy.estimators.callbacks import Callback

from pgmpy.estimators import ValidationLikelihood, CVPredictiveLikelihood, GaussianValidationLikelihood, \
    ValidationConditionalKDE, ValidationSPBNStrict
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

        scoring_isnot_none = scoring_method is not None

        if scoring_isnot_none:
            if isinstance(scoring_method, (CVPredictiveLikelihood, ValidationLikelihood)):
                sc = lambda n, p: scoring_method.local_score(n, p, model.node_type[n], model.node_type)
            elif isinstance(scoring_method, ValidationSPBNStrict):
                sc = lambda n, p: scoring_method.local_score(n, p, model.node_type[n])
            else:
                sc = scoring_method.local_score

            val_sc = None
            if isinstance(scoring_method, ValidationLikelihood):
                val_sc = lambda n, p: scoring_method.validation_local_score(n, p, model.node_type[n], model.node_type)
            elif isinstance(scoring_method, ValidationSPBNStrict):
                val_sc = lambda n, p: scoring_method.validation_local_score(n, p, model.node_type[n])
            elif isinstance(scoring_method, (GaussianValidationLikelihood, ValidationConditionalKDE)):
                val_sc = scoring_method.validation_local_score

        for node in model.nodes:
            parents = model.get_parents(node)
            if scoring_isnot_none:
                total_score += sc(node, parents)

        if isinstance(model, HybridContinuousModel):
            for node in model_copy.nodes:
                if model_copy.node_type[node] == NodeType.SPBN:
                    model_copy.nodes[node]['style'] = 'filled'
                    model_copy.nodes[node]['fillcolor'] = 'gray'
                if model_copy.node_type[node] == NodeType.SPBN_STRICT:
                    model_copy.nodes[node]['style'] = 'filled'
                    model_copy.nodes[node]['fillcolor'] = 'black'
                    model_copy.nodes[node]['fontcolor'] = 'white'


        if operation is not None:
            op, source, dest, score = operation

            if op == '+':
                model_copy.edges[source, dest]['color'] = 'green3'
                model_copy.edges[source, dest]['label'] = "{:0.3f}".format(score)
            elif op == '-':
                model_copy.add_edge(source, dest)
                model_copy.edges[source, dest]['color'] = 'firebrick1'
                model_copy.edges[source, dest]['label'] = "{:0.3f}".format(score)
            elif op == 'flip':
                model_copy.edges[dest, source]['color'] = 'dodgerblue'
                model_copy.edges[dest, source]['label'] = "{:0.3f}".format(score)
            elif op == 'type':
                model_copy.nodes[source]['style'] = 'filled'
                model_copy.nodes[source]['label'] = "{}\n{:0.3f}".format(source, score)

                if dest == NodeType.SPBN:
                    model_copy.nodes[source]['fillcolor'] = 'darkolivegreen1'
                if dest == NodeType.SPBN_STRICT:
                    model_copy.nodes[source]['fillcolor'] = 'lightskyblue'
                    model_copy.nodes[source].pop('fontcolor', None)
                elif dest == NodeType.GAUSSIAN:
                    model_copy.nodes[source]['fillcolor'] = 'yellow'

        title = "{}\nScore {:0.3f}".format(type(scoring_method).__name__, total_score)

        if scoring_isnot_none:
            if isinstance(scoring_method, (ValidationLikelihood, GaussianValidationLikelihood, ValidationConditionalKDE, ValidationSPBNStrict)):
                val_score = 0
                for node in model.nodes:
                    parents = model.get_parents(node)
                    val_score += val_sc(node, parents)

                title += "\nValidation Score: {:0.3f}".format(val_score)

        a = nx.nx_agraph.to_agraph(model_copy)
        if scoring_isnot_none:
            a.graph_attr.update(label=title, labelloc="t", fontsize='25')
        a.write('{}/{:06d}.dot'.format(self.folder, iter_no))
        a.clear()

        subprocess.run(["dot", "-Tpdf", "{}/{:06d}.dot".format(self.folder, iter_no), "-o",
                        "{}/{:06d}.pdf".format(self.folder, iter_no)])
