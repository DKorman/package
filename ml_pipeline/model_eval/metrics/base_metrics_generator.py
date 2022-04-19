from abc import ABC, abstractmethod


class BaseMetricsGenerator(ABC):
    """
        Parent class for project and non-project specific metric calculator subclasses
    """

    def __init__(self, id=None):
        self.id = id

    @abstractmethod
    def generate_metrics(self, y, y_hat):
        pass
