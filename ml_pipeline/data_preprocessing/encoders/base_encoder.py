from abc import ABC, abstractmethod


class BaseEncoder(ABC):
    """
        Parent class for creating feature encoding subclasses
    """

    def __init__(self, cols):
        """

        :param cols: input columns which will be encoded and replaced with original
        :param encoder: used to store trained encoder object
        """
        self.encoder = None
        self.cols = cols

    @abstractmethod
    def fit_transform(self, df, *kwrgs):
        pass

    @abstractmethod
    def transform(self, df):
        pass
