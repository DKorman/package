from abc import ABC, abstractmethod


class BaseModeLBuildingStrategy(ABC):
    """
        Parent class for model building strategy subclasses.
        Integrates model base procedures into a specific building schema (build strategy).
        Core components of a particular Model building strategy are the choice of train/test splits

    """

    def __init__(self):

        pass


    @abstractmethod
    def build_model(self,
                    random_state,
                    df,
                    model_target,
                    model_features_regex,
                    model_base,
                    metrics_generators,
                    graph_generators,
                    *args,
                    **kwargs
                    ):

        """This method is the main method used when implementing any model building schemse."""
        pass