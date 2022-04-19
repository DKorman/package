from model_building import BaseModeLBase
# from utility import extract_cols_from_classes

class OrdinaryModelBase(BaseModeLBase):

    def _preprocess_model_input(self, df, use_stored_objects):

        ### apply encoder if provided ###
        if self.feature_creators != None:
            df = self._execute_preprocess(df, use_stored_objects, self.feature_creators)

        ### apply encoder if provided ###
        if self.encoders != None:
            df = self._execute_preprocess(df, use_stored_objects, self.encoders)

        ### apply imputer if provided ###
        if self.imputers != None:
            df = self._execute_preprocess(df, use_stored_objects, self.imputers)

        ### apply scaler if provided ###
        if self.scalers != None:
            df = self._execute_preprocess(df, use_stored_objects, self.scalers)

        return df

    def _execute_preprocess(self, df, use_stored_objects, preprocess_objects):

        if not use_stored_objects:
            for object in preprocess_objects:
                df = object.fit_transform(df)
        else:
            for object in preprocess_objects:
                df = object.transform(df)

        return df

    def build_model(self, X_train, y_train, train_features, params):

        self.train_features = train_features
        # self.aux_features = extract_cols_from_classes(self.feature_creators, X_train.columns)

        # subset cols for training and FE(if there are any)
        # X_train = X_train[self.train_features + self.aux_features]
        X_train = X_train[self.train_features]

        # apply preprocess objects
        X_train = self._preprocess_model_input(X_train, use_stored_objects=False)

        # train model
        # X_train = X_train.drop(self.aux_features, axis=1)
        self.trainer.train_model(X_train, y_train, params)

        return X_train

    def apply_model(self, predict_data):

        # predict_data = predict_data[self.train_features + self.aux_features]
        predict_data = predict_data[self.train_features]

        predict_data = self._preprocess_model_input(predict_data, use_stored_objects=True)
        # predict_data = predict_data.drop(self.aux_features, axis=1)

        return self.trainer.predict(predict_data), predict_data
