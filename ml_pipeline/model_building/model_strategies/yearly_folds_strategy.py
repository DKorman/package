import logging
import pandas as pd
from sklearn.model_selection import ParameterGrid
import mlflow
import numpy as np
import sys

from model_building import BaseModeLBuildingStrategy

from typing import Dict, List
from pathlib import Path


class YearlyFoldsRegressionStrategy(BaseModeLBuildingStrategy):

    def build_model(self,
                    random_state: int,
                    df: pd.DataFrame,
                    data_subset: Dict[str, int],
                    data_frac: int,
                    model_target: str,
                    model_features_regex: List[str],
                    model_hyperparameters,
                    model_base,
                    metrics_generators,
                    graph_generators,
                    outer_fold_strategy,
                    inner_fold_strategy
                    ):

        # shuffle and subsample df if specified 
        df = df.sample(frac=data_frac, random_state=random_state)

        # subset data
        for key, value in data_subset.items():
            df = df[df[key] == value]

        # create a final list of input features )
        predictor_features = []
        for i in model_features_regex:
            predictor_features.extend(df.filter(regex=fr"\b{i}").columns)

        # prepare for looping process
        fold_ids = []
        outer_fold_results = []
        param_grid = list(ParameterGrid(model_hyperparameters))

        ### perform outer fold splits ###
        df = df.reset_index(drop=True)
        outer_folds = outer_fold_strategy(df=df)

        logger = logging.getLogger(sys._getframe().f_code.co_name)
        for outer_fold in outer_folds:

            # append fold tag
            fold_ids.append(outer_fold[2])
            logging.info('\n')
            logger.info(f'Executing outer CV fold {outer_fold[2]}')

            # define train and test sets
            train = df.loc[outer_fold[0]]
            test = df.loc[outer_fold[1]]
            train = train.sort_values(by='SeasonID')
            # create X and y variables
            X_train = train.drop([model_target], axis=1)
            X_test = test.drop([model_target], axis=1)

            y_train = train[model_target]
            y_test = test[model_target]

            ### perform nested cv gridsearch ###
            X_train = X_train.reset_index(drop=True)
            y_train = y_train.reset_index(drop=True)
            inner_folds = inner_fold_strategy(df=X_train)

            best_inner_fold = []

            for inner_fold in inner_folds:
                logger.info(f'### Executing inner CV fold {inner_fold[2]}')

                # subsetting train sets !
                X_train_inner = X_train.loc[inner_fold[0]]
                X_test_inner = X_train.loc[inner_fold[1]]

                y_train_inner = y_train.loc[inner_fold[0]]
                y_test_inner = y_train.loc[inner_fold[1]]

                tuning_results = []

                counter = 1
                for params in param_grid:
                    _ = model_base.build_model(X_train_inner, y_train_inner, predictor_features, params)

                    predictions, _ = model_base.apply_model(X_test_inner)

                    tuning_results.append(
                        [
                            self._generate_metric_results(
                                y_true=y_test_inner,
                                y_predicted=predictions,
                                metrics_generators=metrics_generators
                            ),
                            params
                        ]
                    )

                    logger.info(f'### ### Executing parameter {counter}/{len(param_grid)}: of inner crossvalidation')
                    counter += 1

                tuning_results.sort()  # TODO: add functionalty to choose normal or reverse sort, and by which metric
                best_inner_fold.append(tuning_results[0])

            #### returns best hyperparameters
            best_params = best_inner_fold[0][1]
            X_train_transformed = model_base.build_model(X_train, y_train, predictor_features, best_params)
            predictions, X_test_transformed = model_base.apply_model(X_test)

            #### calculate fold metrics ###
            outer_fold_results.append(
                [
                    self._generate_metric_results(
                        y_true=y_test,
                        y_predicted=predictions,
                        metrics_generators=metrics_generators
                    ),
                    best_params,
                    y_test.mean()

                ]
            )

            ### generate fold graphs ###
            for generator in graph_generators:
                generator.generate_graphs(
                    X_train=X_train_transformed,
                    X_test=X_test_transformed,
                    y_train=y_train,
                    y_test=y_test,
                    y_predicted=predictions,
                    model=model_base.trainer.model,
                    tag=fold_ids[-1]
                )


        df_metrics = self._postprocess_metrics(
            results = outer_fold_results,
            # values=[x[0] for x in outer_fold_results],
            columns=[x().id for x in metrics_generators],
            fold_names = fold_ids
        )

        logging.info('\n\n')
        logger.info(f'YearlyFoldsRegressionStrategy results are:\n {df_metrics}')

        ### build and save final model
        X_train = df.drop([model_target], axis=1)
        y_train = df[model_target]

        # build final model
        model_base.build_model(
            X_train,
            y_train,
            predictor_features,
            self._extract_optimal_kfold_hyperparameters(outer_fold_results)
        )

        # save model
        model_base.save(Path.cwd() / 'artefacts/model.pkl')
        mlflow.log_artifact(local_path=Path.cwd() / 'artefacts/model.pkl')

    def _postprocess_metrics(self, results, columns, fold_names):

        # aggregate and log results
        df_metrics = pd.DataFrame(
            [x[0] for x in results],
            columns=columns
         )

        df_metrics.insert(loc=0, column='fold/year', value=fold_names)
        df_metrics['target_mean'] = [x[2] for x in results]
        df_metrics.loc[len(df_metrics)] = ['AVG'] + list(df_metrics.mean()[1:].values)
        df_metrics.to_csv(Path.cwd() / 'artefacts/metrics.csv')
        mlflow.log_artifact(local_path=Path.cwd() / 'artefacts/metrics.csv')

        avg_result_first_metric = zip(
            [f'MAPE_{x}' for x in df_metrics.T.iloc[0, ::-1]],
            df_metrics.T.iloc[1, ::-1]
        )
        for fold_result in avg_result_first_metric:
            mlflow.log_metric(fold_result[0], fold_result[1])

        return df_metrics

    def _generate_metric_results(self, y_true, y_predicted, metrics_generators):

        fold_metric_scores = []

        for generator in metrics_generators:
            metric = generator().generate_metrics(
                y=y_true,
                y_hat=y_predicted
            )
            fold_metric_scores.append(metric)

        return fold_metric_scores

    def _extract_optimal_kfold_hyperparameters(self, kfold_results):

        params = []

        for i in kfold_results:
            params.append(i[1])

        params = {
            k: [d.get(k) for d in params]
            for k in set().union(*params)
        }

        for key, value in params.items():
            if type(value[0]) == int:
                params[key] = round(np.mean(value))
            if type(value[0]) == float:
                len_decimals = len(str(value[0]).split('.')[1])
                params[key] = round(np.mean(value), len_decimals + 1)
            if type(value[0]) == str:
                params[key] = max(set(value), key=value.count)

        return params
