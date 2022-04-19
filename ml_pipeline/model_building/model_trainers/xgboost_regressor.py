from model_building import BaseModeLTrainer
import xgboost as xgb


class XgboostRegressor(BaseModeLTrainer):

    # TODO: add crossvalidation feature to this class

    def train_model(self, X_train, y_train, params):
        # create a trainer instance
        trainer = xgb.XGBRegressor(
            random_state=self.random_state,
            **params
        )

        # apply trainer on train data
        trainer.fit(X_train, y_train)

        # store trainer as instance atribute for later usage
        self.model = trainer

        return trainer

    def predict(self, df):
        return self.model.predict(df)
