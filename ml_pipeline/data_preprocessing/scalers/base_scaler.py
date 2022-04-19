from abc import ABC, abstractmethod


class BaseScaler(ABC):
    """
    Parent class for creating feature scaling subclasses
    """

    def __init__(self, cols):
        """

        :param cols: input columns which will be scaled and replaced with original
        :param scaler: used to store trained scaler object
        """

        self.scaler = None
        self.cols = cols

    @abstractmethod
    def fit_transform(self, df, *kwrgs):
        pass

    @abstractmethod
    def transform(self, df):
        pass

