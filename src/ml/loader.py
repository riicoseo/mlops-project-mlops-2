import mlflow
from mlflow.tracking import MlflowClient

from src.utils.logger import get_logger
from src.ml.config import init_mlflow

logger = get_logger(__name__)

def load_mlflow_model(model_uri: str):
    try:
        init_mlflow()
        logger.info(f"[START] try to load MLflow model: {model_uri}")
        
        model = mlflow.pyfunc.load_model(model_uri)

        run_id = model.metadata.run_id

        client = MlflowClient()
        run = client.get_run(run_id)
        run_name = run.data.tags.get("mlflow.runName")
        model_timestamp = run.data.tags.get("model_timestamp")

        model.run_id = run_id
        model.run_name = run_name
        model.model_timestamp = model_timestamp

        logger.info(f"[INFO] Loaded MLflow model")
        logger.info(f"       Run ID: {run_id}")
        logger.info(f"       Run Name: {run_name}")
        logger.info(f"       Model Timestamp: {model_timestamp}")
        

        return model

    except Exception as e:
        logger.error(f"[ERROR] failed to load mlflow model : {e}")
        raise    