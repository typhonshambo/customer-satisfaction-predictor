import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression

class ModelDev(ABC):
    '''
    Abstract class for model development
    '''

    @abstractmethod
    def train(self, x_train, y_train) -> None:
        '''
        Train a model
        Args:
            x_train: Training data
            y_train: Training labels
        returns:
            None
        '''
        pass

class LinearRegressionModel(ModelDev):
    '''
    Linear Regression Model
    '''
    def __init__(self):
        self.model = None

    def train(self, x_train, y_train, **kwargs):
        '''
        Train a Linear Regression model
        Args:
            x_train: Training data
            y_train: Training labels
        returns:
            None
        '''
        try:
            logging.info('Training Linear Regression model')
            self.model = LinearRegression(**kwargs)
            trained_model = self.model.fit(x_train, y_train)
            return trained_model

        except Exception as e:
            logging.error(f'Error training model: {e}')
            raise e