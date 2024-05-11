from pipelines.training_pipeline import train_pipeline
from zenml. client import Client

if __name__ == '__main__':
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline('data/olist_customers_dataset.csv')
    # mlflow ui --backend-store-uri "file:/Users/shambo/Library/Application Support/zenml/local_stores/69cbbe46-68f9-4102-b54f-84d109af848a/mlruns"