class Callback:

    def call(self, model, operation, scoring_method, iter_no):
        raise NotImplementedError("The callback is implemented by its subclasses.")
