import logging
from abc import ABC, abstractmethod
import numpy as np
from  sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
class Evaluation(ABC):
    '''
    Abstract class for model evaluation
    '''
    @abstractmethod
    def evaluate_model(self, y_test:np.ndarray, y_pred:np.ndarray) -> None:
        '''
        Evaluate the model
        Args:
            y_test: Testing labels
            y_pred: Predicted labels
        returns:
            None
        '''
        pass

class MSE(Evaluation):
    '''
    Mean Squared Error
    '''
    def evaluate_model(self, y_test:np.ndarray, y_pred:np.ndarray) -> None:
        '''
        Evaluate the model
        Args:
            y_test: Testing labels
            y_pred: Predicted labels
        returns:
            None
        '''
        try:
            mse = mean_squared_error(y_test, y_pred)
            logging.info(f'Mean Squared Error: {mse}')
            return mse
        except Exception as e:
            logging.error(f'Error calculating MSE: {e}')
            raise e
        
class R2Score(Evaluation):
    '''
    R2 Score
    '''
    def evaluate_model(self, y_test:np.ndarray, y_pred:np.ndarray) -> None:
        '''
        Evaluate the model
        Args:
            y_test: Testing labels
            y_pred: Predicted labels
        returns:
            None
        '''
        try:
            r2 = r2_score(y_test, y_pred)
            logging.info(f'R2 Score: {r2}')
            return r2
        except Exception as e:
            logging.error(f'Error calculating R2 Score: {e}')
            raise e
    
class RMSE(Evaluation):
    '''
    Root Mean Squared Error
    '''
    def evaluate_model(self, y_test:np.ndarray, y_pred:np.ndarray) -> None:
        '''
        Evaluate the model
        Args:
            y_test: Testing labels
            y_pred: Predicted labels
        returns:
            None
        '''
        try:
            rmse = root_mean_squared_error(y_test, y_pred)
            logging.info(f'Root Mean Squared Error: {rmse}')
            return rmse
        except Exception as e:
            logging.error(f'Error calculating RMSE: {e}')
            raise e