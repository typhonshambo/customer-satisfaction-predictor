import pandas as pd
import numpy as np
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLflowModelDeploye
from zenml.integrations.mlflow.services import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output

from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluation import model_evaluation

docker_settings = DockerSettings(required_integrations=[MLFLOW])

class DeploymentTriggerConfig(BaseParameters):
	'''Deployment trigger configuration'''
	min_accuracy: float = 0.92

	
@step
def deploy_trigger(
	accuracy: float,
	config: DeploymentTriggerConfig
) -> bool:
	'''
	Deployment trigger
	args:
		accuracy: accuracy of the model
		config: DeploymentTriggerConfig
	return:
		bool: True if the accuracy is greater than or equal to the minimum accuracy required for deployment
	'''
	return accuracy>=config.min_accuracy


@pipeline(
	enable_cache=True, 
	settings={
		"docker_settings": docker_settings
	}
)
def continous_deployment_pipeline(
	min_accuracy: float = 0.92,
	workers: int = 1,
	timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT
):
	df = ingest_df()
	x_train, x_test, y_train, y_test = clean_df(df)
	model = train_model(x_train, x_test, y_train, y_test)
	mse, r2_score, rmse = model_evaluation(model, x_test, y_test)
	deployment_decision = deploy_trigger(r2_score)