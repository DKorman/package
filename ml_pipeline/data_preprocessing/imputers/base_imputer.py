from abc import ABC, abstractmethod


class BaseImputer(ABC):
    """
    Imputes missing values for specified columns
    """

    def __init__(self, cols):
        """

        :param cols: input columns who's missing values will be imputed
        :param imputations: used to store train-set means for selected columns
        """
        self.imputations = {}
        self.cols = cols

    @abstractmethod
    def fit_transform(self, df, *kwrgs):
        pass

    @abstractmethod
    def transform(self, df):
        pass
