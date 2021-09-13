class BaseMetric(object):

    def __init__(self, metric_names, eval_intermediate=True, eval_validation=True):
        self._names = tuple(metric_names)
        self._eval_intermediate = eval_intermediate
        self._eval_validation = eval_validation

    def eval_intermediate(self):
        return self._eval_intermediate

    def eval_validation(self):
        return self._eval_validation

    def names(self):
        return self._names

    def add(self, predictions, ground_truth):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError


