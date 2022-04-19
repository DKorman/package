from category_encoders.one_hot import OneHotEncoder
from data_preprocessing import BaseEncoder

import pandas as pd


class OHE(BaseEncoder):
    """Applies one hot encoding to selected columns"""

    def fit_transform(self, df):
        # fit and apply  OHE (on training data)
        ohe = OneHotEncoder(cols=self.cols, use_cat_names=True)

        # apply it on a copy of main df
        ohe_df = df[self.cols]
        # print(ohe_df)
        ohe_df = ohe.fit_transform(ohe_df)

        # store trained encoder as instance attribute for later usage
        self.encoder = ohe

        df = self._replace_original_cols(df, ohe_df, self.cols)

        return df

    def transform(self, df):
        ohe_df = df[self.cols]
        ohe_df = self.encoder.transform(ohe_df)

        df = self._replace_original_cols(df, ohe_df, self.cols)

        return df

    def _replace_original_cols(self, df, ohe_df, cols):
        # add perfix to encoded columns
        ohe_df.columns = [f'OHE_{x}' for x in ohe_df.columns]

        # remove nan columns
        ohe_df = ohe_df.drop([x for x in ohe_df.columns if '_nan' in x], axis=1)

        # drop old columns and replace them with encoded ones
        df = df.drop(cols, axis=1)

        df = pd.concat([df, ohe_df], axis=1)

        return df
