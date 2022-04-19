import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from model_eval import BaseMetricsGenerator

class MeanAbsolutePercentageErrorGenerator(BaseMetricsGenerator):

    def __init__(self):
        super().__init__(id='MAPE')

    def generate_metrics(self, y: pd.DataFrame, y_hat:pd.DataFrame):

        '''
        function used to calculate all relevant regression metrics on actual vs. predicted values

        :param y: actual values
        :param y_hat: predicted values
        :return: list of metrics
        '''

        mape = mean_absolute_percentage_error(y, y_hat)

        return mape

class RootMeanSquaredErrorGenerator(BaseMetricsGenerator):

    def __init__(self):
        super().__init__(id='RMSE')


    def generate_metrics(self, y: pd.DataFrame, y_hat:pd.DataFrame):

        '''
        function used to calculate all relevant regression metrics on actual vs. predicted values

        :param y: actual values
        :param y_hat: predicted values
        :return: list of metrics
        '''


        rmse = mean_squared_error(y, y_hat, squared=False)

        return rmse

class MeanAbsoluteErrorGenerator(BaseMetricsGenerator):

    def __init__(self):
        super().__init__(id='MAE')

    def generate_metrics(self, y: pd.DataFrame, y_hat:pd.DataFrame):

        '''
        function used to calculate all relevant regression metrics on actual vs. predicted values

        :param y: actual values
        :param y_hat: predicted values
        :return: list of metrics
        '''

        mae = mean_absolute_error(y, y_hat)

        return mae


# class BulkYieldErrorGenerator(BaseMetricsGenerator):
#
#     def generate_metrics(self, y_true, y_pred, povrsina_train, povrsina_test, is_test=True):
#         """
#         Calculates error between the actual total yields of all tables vs. predicted yields of all tables
#         Args:
#             y_true: test set true values
#             y_pred: test set predicted values
#             povrsina_train: agriculture table size
#             povrsina_test: agriculture table size
#             is_test:
#
#         Returns: BYPE (bulk yield prediction error)
#
#             """
#         if is_test==True:
#             y_bulk_true = (povrsina_test * y_true).sum()
#             y_bulk_pred = (povrsina_test * y_pred).sum()
#         else:
#             y_bulk_true = (povrsina_train * y_true).sum()
#             y_bulk_pred = (povrsina_train * y_pred).sum()
#
#         return (np.abs(y_bulk_true-y_bulk_pred)/y_bulk_true)
