import os
import mlflow

from config import MLFLOW_URI

def init_mlflow(mlflow_uri: str = None):
    tracking_uri = MLFLOW_URI or mlflow_uri or "http://localhost:5000"
    mlflow.set_tracking_uri(tracking_uri)