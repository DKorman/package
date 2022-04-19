from feature_engineering import BaseFeatureEngineering


class GroupByFeatureMean(BaseFeatureEngineering):

    def __init__(self, group, feature):
        super().__init__()
        self.group = group
        self.feature = feature
        self.group_means = None

    def fit_transform(self, df, target):

        group_means = df.groupby([self.group])[self.feature] \
            .mean() \
            .reset_index() \
            .rename(columns={self.feature: f'mean_{self.feature}_groupedby_{self.group}'})

        df = df.merge(group_means, how='left', on=self.group)

        self.group_means = group_means

        return df

    def transform(self, df):

        df = df.merge(self.group_means, how='left', on=self.group)

        return df