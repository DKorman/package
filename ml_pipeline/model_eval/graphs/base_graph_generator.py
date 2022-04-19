from abc import ABC, abstractmethod


class BaseGraphGenerator(ABC):
    """
        Parent class for project and non-project graph creating subclasses
    """

    def __init__(self):
        pass


    @abstractmethod
    def generate_graphs(self):
        pass

