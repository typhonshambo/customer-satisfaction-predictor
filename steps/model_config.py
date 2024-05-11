from zenml.steps import base_parameters

class ModelNameConfig(base_parameters.BaseParameters):
    '''
    Configuration for the model to be used
    '''
    model_name: str = 'LinearRegression'
        