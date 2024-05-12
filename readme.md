# Predicting Customer Satisfaction for E-Commerce Orders (MLOps)

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/zenml)](https://pypi.org/project/zenml/)

## Introduction

Predicting customer satisfaction is crucial for any e-commerce business. This project focuses on predicting the review score for future orders or purchases by leveraging machine learning techniques. We utilize the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data), which contains information on 100,000 orders made at multiple marketplaces in Brazil from 2016 to 2018.

## Problem Statement

Given a customer's historical data, we aim to predict their satisfaction score for the next order or purchase. This involves analyzing various dimensions such as order status, price, payment, freight performance, customer location, product attributes, and customer reviews. The objective is to build a production-ready pipeline using ZenML to predict customer satisfaction scores accurately.

## Approach

We approach the problem with an end-to-end pipeline using ZenML:

- **Data Preprocessing**: Clean the data by dropping unnecessary columns and filling missing values.
- **Data Division**: Divide the data into training and testing datasets.
- **Model Development**: Train a machine learning model to predict customer satisfaction scores.
- **Evaluation**: Evaluate the model's performance using metrics like Mean Squared Error (MSE).
- **Deployment**: Deploy the trained model using MLflow for real-time predictions.

## Key Features

- **Automation**: ZenML automates the machine learning pipeline, from data preprocessing to model deployment.
- **Scalability**: The pipeline is designed to scale according to business needs, ensuring seamless performance.
- **Integration**: Integration with MLflow enables tracking of metrics, parameters, and model versions.
- **Streamlit App**: A Streamlit web application showcases real-time predictions for customer satisfaction.

## Usage

### Python Requirements

Install the required Python packages:

```bash
git clone https://github.com/typhonshambo/customer-satisfaction-predictor
cd customer-satisfaction-predictor
pip install -r requirements.txt
```

### ZenML Setup
```py
zenml integration install mlflow -y
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow --flavor=mlflow
zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set
zenml up
```
### MLFlow Setup Locally
```
mlflow ui --backend-store-uri "put_your_uri_here"
```
`uri` can be obtained from: <br>
[run_pipeline.py](run_pipeline.py)
```py
from pipelines.training_pipeline import train_pipeline
from zenml.client import Client

if __name__ == '__main__':
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    #this will print uri
    ...
``` 


Execute the training pipeline:
```py
python run_pipeline.py
```
Execute the deployment pipeline:
```py
python run_deployment.py --help
```
> this will provide the commands available for run_deployment <br> Avaiable commands : <br> - `--deploy` <br> - `--predict` <br>- `--deploy_and_predict` <br> for example : python `run_deployment.py --deploy`

Execute Streamlit App locally
```py
streamlit run streamlit_app.py
```
### Contributions
I welcome everyone who want to improve this project, there are lots of things which need to be improved in this. Kindly create an issue and i'll assign you with it.

### Credits
Thanks to @ayush714 for his course, from where i learnt to implement this whole project by my own and learnt a lot about MLOps