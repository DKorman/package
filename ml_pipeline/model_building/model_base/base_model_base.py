from abc import ABC, abstractmethod
import pickle


class BaseModeLBase(ABC):
    """
        Parent class for base model building.
        Base model building contains core ML procedures such as scaling, imputing, model training.
        Each of these procedures is provided as a separate parameter to the model base
        These procedures are then repeated in same order on test/production data as were on train.
    """

    def __init__(
            self,
            trainer=None,
            encoders=None,
            scalers=None,
            imputers=None,
            feature_selectors=None,
            feature_creators=None
    ):
        """

        :param trainer: algorithm class
        :param encoders: list of encoder classes
        :param scalers: list of scaler classes
        :param imputers: list of imputer classes
        :param feature_selectors: feature selector class
        :param feature_creators:  list of feature creator classes
        :param model: used to store trained models
        :param train_features: used to store final feature set
        """
        self.trainer = trainer
        self.encoders = encoders
        self.scalers = scalers
        self.imputers = imputers
        self.feature_selectors = feature_selectors
        self.feature_creators = feature_creators
        self.model = None
        self.train_features = None
        self.aux_features = None

    @abstractmethod
    def build_model(self, X_train, y_train, train_features, **kwargs):
        """
        Executes ML preprocessing procedures (if provided) and builds a model.
        In the background, individual ML procedures store respective objects in their classes.

        :param X_train: pd dataframe - mandatory parameter
        :param y_train: pd dataframe - mandatory parameter
        :param kwargs: other possibly required params specific to chosen algorithm (e.g. parameters)
        :return: optionally, returns transformed train df if required by tools such as SHAP
        """
        pass

    @abstractmethod
    def apply_model(self, predict_data):
        """
        Applies the same steps from the model building process

        :param predict_data: pd dataframe - in practice, either validation, test or production data
        :return: predictions based on trained model. Optionally, also returns transformed prediction dataset
        """
        pass

    @abstractmethod
    def _preprocess_model_input(self, df, use_stored_objects=False):
        """
        Applies ML preprocessing procedures provided to the Model base.
        Applies them in the order as provided when creating Model base instance.

        :param df: pd Dataframe - dataset used for training
        :param use_stored_objects: boolean - whether to fit ML procedures or apply them using stored object
        :return: transformed dataframe used for model building
        """
        pass

    def save(self, path):
        """
        Serializes the model base class (self) to specified directory
        :param path: string or Path object
        :return: None
        """
        pickle.dump(self, open(path, 'wb'))

    def load(self, path):
        """
        Loads the saved model base class
        :param path: string or Path object
        :return: None
        """
        self.__dict__ = pickle.load(open(path, 'rb')).__dict__
