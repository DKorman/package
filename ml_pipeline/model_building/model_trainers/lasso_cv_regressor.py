from model_building import BaseModeLTrainer
from sklearn.linear_model import LassoCV

import pandas as pd
import numpy as np

# TODO: Make changes in this class so that it allignes with  Xgboost and functions properly

# class LassoCrossValidation(BaseModeLTrainer):
#
#
#     def train_model(self, X_train, X_test, y_train):
#
#         # create a trainer instance
#         trainer = LassoCV()
#
#         # apply trainer on train data
#         trainer.fit(X_train, y_train)
#
#         if isinstance(X_test, (pd.DataFrame, np.ndarray)):
#
#             # store trainer as class atribute
#             self.model = trainer
#
#             # return predictions
#             return trainer.predict(X_test)
#
#         else:
#
#             # store fitted model as class variable
#             self.model = trainer
#
#     def predict(self, df):
#
#         return self.model.predict(df)
#
