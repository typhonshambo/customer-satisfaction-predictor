import logging
from abc import ABC,abstractmethod
from typing import Union
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class dataStrategy(ABC):
    '''
    abstract class for data handling
    '''
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class dataPreProcessing(dataStrategy):
    '''
    class for data preprocessing
    '''
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        '''
        preprocess the data
        '''
        try:
            '''
            droping columns which woun't be used in the model
            '''
            data = data.drop(
                [
                    'order_approved_at',
                    'order_delivered_carrier_date',
                    'order_delivered_customer_date',
                    'order_estimated_delivery_date',
                    'order_purchase_timestamp',

                    'customer_zip_code_prefix', 
                    'order_item_id'
                ],
            axis=1)

            '''
            filling missing values using meadian of data
            '''

            data['product_weight_g'] = data['product_weight_g'].fillna(data['product_weight_g'].median())
            data['product_length_cm'] = data['product_length_cm'].fillna(data['product_length_cm'].median())
            data['product_height_cm'] = data['product_height_cm'].fillna(data['product_height_cm'].median())
            data['product_width_cm'] = data['product_width_cm'].fillna(data ['product_width_cm'].median())
            data['review_comment_message'] = data['review_comment_message'].fillna('No Comment')
            
    
            '''
            selecting only numeric columns for ease implementatons
            Note: this shouldn't be done in real world scenario
            '''
            data = data.select_dtypes(include=[np.number])
            return data
        
        except Exception as e: 
            logging.error(f"Error in data preprocessing: {e}")


class dataDivision(dataStrategy):
    '''
    class for data division into traning and testing datasets
    '''
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        '''
        divide the data into train and test
        '''
        try:
            X = data.drop('review_score', axis=1)
            y = data['review_score']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        
        except Exception as e: 
            logging.error(f"Error in data division: {e}")

class dataCleaning:
    '''
    Class which cleans the data the divides it into training and testing datasets
    '''
    def __init__(self, data: pd.DataFrame, strategy: dataStrategy):
        self.data = data
        self.strategy = strategy
    
    def clean_data(self) -> Union[pd.DataFrame, pd.Series]:
        '''
        clean the data
        '''
        try:
            cleaned_data = self.strategy.handle_data(self.data)
            return cleaned_data
        
        except Exception as e: 
            logging.error(f"Error in data cleaning: {e}")