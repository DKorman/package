import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import mlflow

from model_eval import BaseGraphGenerator


class ActualVsPredictedGraphGenerator(BaseGraphGenerator):

    def generate_graphs(self,
                        y_test,
                        y_predicted,
                        tag=None,
                        **kwargs
                        ):
        plt.clf()

        min_y = min([min(y_test), min(y_predicted)])
        max_y = max([max(y_test), max(y_predicted)])

        ax = sns.regplot(x=y_test, y=y_predicted, ci=None)
        ax.set(
            xlabel='y actual',
            ylabel='y predicted',
            title='Actual vs. Predicted'
        )

        sns.lineplot(x=[min_y, max_y], y=[min_y, max_y], linestyle='--')

        plt.legend(
            labels=[
                'best fit',
                'identity',
            ]
        )
        plt.grid()
        fig = plt.gcf()

        cwd = Path.cwd()
        file_path = cwd / f'artefacts/{tag}_actual_vs_predicted.jpg'

        fig.savefig(file_path)
        plt.clf()

        mlflow.log_artifact(local_path=file_path)
