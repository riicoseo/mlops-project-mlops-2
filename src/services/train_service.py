from src.ml.trainer import train_and_log_model
from src.utils.logger import get_logger

logger = get_logger(__name__)

def run_training_job(model_name: str, **kwargs):
    try:
        logger.info(f"[START] run traing job : model_name={model_name}")
        train_and_log_model(model_name, **kwargs)
        logger.info(f"[END] SUCCESS training : model_name={model_name}")
    except Exception as e:
        logger.error(f"[ERROR] fail to training model")
        logger.error(f"[ERROR] {e}")        
