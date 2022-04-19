import pandas as pd
import seaborn as sns
import shap
import os
import mlflow
from pathlib import Path
import matplotlib.pyplot as plt

from model_eval import BaseGraphGenerator


class ShapGraphsGenerator(BaseGraphGenerator):

    def __init__(self, explainer):
        self.explainer = explainer
        super().__init__()

    def _select_examples(self, model, X):
        y_pred = [(i, v) for i, v in enumerate(model.predict(X))]
        predictions = pd.DataFrame(y_pred, columns=['id', 'y_pred'])
        predictions = predictions.sort_values(by='y_pred')

        min_ = 0
        max_ = len(predictions) - 1
        q1 = max_ // 4
        q2 = max_ // 2
        q3 = (max_ * 3) // 4

        return set(predictions.iloc[[min_, q1, q2, q3, max_]]['id'])

    def generate_graphs(
            self,
            model,
            X_train,
            X_test,
            tag,
            **kwargs
    ):
        sns.set(style='darkgrid', context='talk', palette='rainbow')

        shap_explainer = self.explainer(model)

        explained_values = shap_explainer(X_train)

        cwd = Path.cwd()

        ### generate summary plots ###
        plt.clf()
        shap.summary_plot(explained_values, plot_type='dot', show=False)
        plt.savefig(
            cwd / f'artefacts/{tag}_dot_summary_plot.jpg',
            bbox_inches="tight"
        )
        mlflow.log_artifact(local_path=cwd / f'artefacts/{tag}_dot_summary_plot.jpg')

        plt.clf()
        shap.summary_plot(explained_values, plot_type='bar', show=False)
        plt.savefig(
            cwd / f'artefacts/{tag}_bar_summary_plot.jpg',
            bbox_inches="tight"
        )
        mlflow.log_artifact(local_path=cwd / f'artefacts/{tag}_bar_summary_plot.jpg')

        plt.clf()
        shap.summary_plot(explained_values, plot_type='violin', show=False)
        plt.savefig(
            cwd / f'artefacts/{tag}_violin_summary_plot.jpg',
            bbox_inches="tight"
        )
        mlflow.log_artifact(local_path=cwd / f'artefacts/{tag}_violin_summary_plot.jpg')

        indices_of_interes = self._select_examples(model, X_train)

        for i in indices_of_interes:
            plt.clf()
            shap.force_plot(shap_explainer.expected_value, explained_values.values[i], X_train.iloc[i],
                            matplotlib=True, show=False)
            plt.savefig(
                cwd / f'artefacts/{tag}_force_plot_ix{i}.jpg',
                bbox_inches="tight"
            )
