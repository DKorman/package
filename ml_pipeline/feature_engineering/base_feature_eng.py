from abc import ABC, abstractmethod


class BaseFeatureEngineering(ABC):
    """
        Parent class for implementing feature engineering relative to train vs. test data
    """

    def __init__(self):
        pass


    @abstractmethod
    def fit_transform(self, df, **kwargs):
        pass

    @abstractmethod
    def transform(self, df, **kwargs):
        pass




# %%

def variety_mean_feature(train_df, test_df, target_var, final):
    tmp = (train_df.groupby(['NazivSorte'])[target_var].mean() / train_df[target_var].mean()).reset_index().rename(
        columns={target_var: 'variety_mean'})

    tmp_train = train_df.merge(tmp, how='left', on='NazivSorte')

    if final == True:
        return tmp_train
    else:
        tmp_test = test_df.merge(tmp, how='left', on='NazivSorte')
        return tmp_train, tmp_test