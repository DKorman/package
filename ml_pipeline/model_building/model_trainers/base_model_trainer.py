from abc import ABC, abstractmethod


class BaseModeLTrainer(ABC):
    """
        Parent class for creating wrappers around commonly used algorithms
    """

    def __init__(self, random_state=None):

        # self.model_key = None
        self.model = None
        self.random_state = random_state


    @abstractmethod
    def train_model(self, X_train, y_train, **kwargs):
        pass


    @abstractmethod
    def predict(self, df):
        pass

