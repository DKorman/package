import pandas as pd
import functools

from scipy import stats

def t_test_comparison(df_train, df_test, equal_variance=False):
    t_test_temp = pd.DataFrame({'feature': df_train.columns,
                                't-test statistic': stats.ttest_ind(df_train, df_test, equal_var=equal_variance)[0],
                                't-test 2-tailed pvalue': stats.ttest_ind(df_train, df_test, equal_var=equal_variance)[
                                    1]})

    train_describe = df_train.describe().T.drop(['count'], axis=1).reset_index().rename(columns={
        'index': 'feature',
        'mean': 'train_mean',
        'std': 'train_std',
        '25%': 'train_25%',
        '75%': 'train_25%',
        '50%': 'train_median',
        'max': 'train_max',
        'min': 'train_min'})

    test_decribe = df_test.describe().T.drop(['count'], axis=1).reset_index().rename(columns={
        'index': 'feature',
        'mean': 'test_mean',
        'std': 'test_std',
        '25%': 'test_25%',
        '75%': 'test_25%',
        '50%': 'test_median',
        'max': 'test_max',
        'min': 'test_min'})

    data_frames = [t_test_temp, train_describe, test_decribe]
    df_comparison = functools.reduce(lambda left, right: pd.merge(left, right, on=['feature'],
                                                                  how='outer'), data_frames)

    return df_comparison
