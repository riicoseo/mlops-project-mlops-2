import mlflow

from src.utils.logger import get_logger
from src.ml.config import init_mlflow

logger = get_logger(__name__)

def load_mlflow_model(model_uri: str):
    try:
        init_mlflow()
        logger.info(f"[Start] try to load MLflow model: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        return model

    except Exception as e:
        logger.error(f"[ERROR] failed to load mlflow model")
        raise    