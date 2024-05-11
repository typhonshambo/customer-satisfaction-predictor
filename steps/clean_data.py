import pandas as pd
from zenml import step
import logging
from src.data_cleaning import dataPreProcessing, dataDivision, dataCleaning
from typing import Annotated, Tuple

@step
def clean_df(df : pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, 'X_train'],
    Annotated[pd.DataFrame, 'X_test'],
    Annotated[pd.Series, 'y_train'],
    Annotated[pd.Series, 'y_test']                      
    ]:
    '''
    Cleans the ingested data
    Args: 
        df: Input data
    Returns: 
        X_train: Training data
        X_test: Testing data
        y_train: Training labels
        y_test: Testing labels
    '''
    try:
        '''
        first step is to preprocess the data  
        then using that preprocessed data to create training and testing datasets
        '''
        data_preprocessing = dataPreProcessing()
        pre_process_class = dataCleaning(df, data_preprocessing)
        pre_processed_data = pre_process_class.clean_data()

        data_division = dataDivision()
        division_class = dataCleaning(pre_processed_data, data_division)
        x_train, x_test, y_train, y_test = division_class.clean_data()
        logging.info("Data cleaning completed")
        return x_train, x_test, y_train, y_test
    except Exception as e: 
        logging.error(f"Error in data cleaning: {e}")