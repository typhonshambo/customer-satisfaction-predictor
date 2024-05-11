import pandas as pd
from zenml import step
import logging
from sklearn.base import RegressorMixin
from src.evaluation import MSE, R2Score, RMSE
from typing import Annotated, Tuple

import mlflow
from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def model_evaluation(
    model: RegressorMixin,
    x_test: pd.DataFrame,
    y_test: pd.DataFrame,
    ) -> Tuple[
        Annotated[float, 'MSE'],
        Annotated[float, 'R2 Score'],
        Annotated[float, 'RMSE']
    ]:
    '''
    Evaluates the model on the testing dataset
    args:
        model: Trained model
        x_test: Testing data
        y_test: Testing labels
    
    return:
        MSE: Mean Squared Error
        R2 Score: R2 Score
        RMSE: Root Mean Squared Error
    '''
    try:
        predictions = model.predict(x_test)

        mse_class = MSE()
        r2_class = R2Score()
        rmse_class = RMSE()

        mse = mse_class.evaluate_model(y_test, predictions)
        r2_score = r2_class.evaluate_model(y_test, predictions)
        rmse = rmse_class.evaluate_model(y_test, predictions)

        mlflow.log_metrics({
            'MSE': mse,
            'R2 Score': r2_score,
            'RMSE': rmse
        })

        return mse, r2_score, rmse

    except Exception as e:
        logging.error(f'Error evaluating model: {e}')
        raise e
