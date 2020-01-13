from pgmpy.estimators.callbacks import Callback


class SaveModel(Callback):

    def __init__(self, folder):
        self.folder = folder

    def call(self, model, operation, scoring_method, iter_no):
        model.save_model("{}/{:06d}".format(self.folder, iter_no))
