import pandas as pd
from zenml import step
import logging
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .model_config import ModelNameConfig

import mlflow
from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(
    x_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelNameConfig

) -> RegressorMixin:
    
    '''
    Train a model

    args:  
        x_train: Training data
        X_test: Testing data
        y_train: Training labels
        y_test: Testing labels
    
    return:
        RegressorMixin: the trained model
    '''
    try: 
        model = None
        if config.model_name == 'LinearRegression':
            try:
                mlflow.sklearn.autolog()
                model = LinearRegressionModel()
                trained_model = model.train(x_train, y_train)
                return trained_model
            except Exception as e:
                logging.error(f'Model {config.model_name} not supported')
                raise e
        
    except Exception as e:
        logging.error(f'Error training model: {e}')
        raise e
    