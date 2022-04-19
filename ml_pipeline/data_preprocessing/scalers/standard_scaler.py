from sklearn.preprocessing import StandardScaler
from data_preprocessing import BaseScaler

import pandas as pd


class CenterAndScaleScaler(BaseScaler):
    """
    Standard scaler from scikit learn used in most situations.
    Scales specified columns and replaces the original with scaled ones
    """

    def fit_transform(self, df):

        # fit and use scaler ond train and test data
        scaler = StandardScaler()

        scaler_df = df[self.cols]

        scaler_df = scaler.fit_transform(scaler_df)

        self.scaler = scaler

        # store trained encoder as instance attribute for later usage
        df = self._replace_original_cols(df, scaler_df, self.cols)

        return df


    def transform(self, df):

        scaler_df = df[self.cols]
        scaler_df = self.scaler.transform(scaler_df)

        df = self._replace_original_cols(df, scaler_df, self.cols)

        return df


    def _replace_original_cols(self,  df, scaler_df, cols):

        # drop old columns and replace them with encoded ones
        df = df.drop(cols, axis=1)

        df = df.reset_index(drop=True)

        df = pd.concat([df, pd.DataFrame(scaler_df, columns=cols)], axis=1, )

        return df

