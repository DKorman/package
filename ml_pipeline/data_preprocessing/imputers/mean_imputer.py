from data_preprocessing import BaseImputer
import numpy as np


class MeanImputer(BaseImputer):
    """
    Used to impute train and test data with means from train data for specified columns
    """

    def fit_transform(self, df):

        for col in self.cols:
            col_mean = df[col].mean()
            df[col] = df[col].replace(np.nan, col_mean)
            self.imputations[col] = col_mean

        return df

    def transform(self, df):

        for col in self.imputations:
            df[col] = self.imputations[col]

        return df

