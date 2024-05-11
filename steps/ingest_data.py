import pandas as pd
from zenml import step
import logging

class IngestData:
    '''
    Ingest data from the data_path
    '''
    def __init__(self, data_path:str) -> None:
        self.data_path = data_path

    def get_data(self) -> pd.DataFrame:
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path, delimiter=',')

@step
def ingest_df(data_path:str) -> pd.DataFrame:
    '''
    Ingest data from the file path
    arg: 
        data_path: path to the data

    return:
        pd.DataFrame: the ingested data in the form of a pandas dataframe
    '''
    try:
        data = IngestData(data_path)
        df = data.get_data()
        return df
    except Exception as e:
        logging.error(f"Failed to ingest data: {e}")
        raise e
