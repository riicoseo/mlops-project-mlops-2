from fastapi import APIRouter
import src.api
from src.ml.loader import load_mlflow_model
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

@router.post("/reload")
def reload_model():
    logger.info("[START] reloading mlflow model")
    src.api.mlflow_model = load_mlflow_model(model_uri="models:/best_model/Production")
    logger.info("[END] mlflow model reloaded")
    return {"status": "success", "message": "Model reloaded successfully"}
