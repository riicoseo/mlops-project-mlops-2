from src.ml.trainer import train_and_log_model
from src.utils.logger import get_logger

logger = get_logger(__name__)

def run_training_job(experiment_name: str, training_params: dict):
    try:
        logger.info(f"[START] run traing job : experiment={experiment_name}")
        train_and_log_model(experiment_name, training_params)
        logger.info(f"[END] SUCCESS training : experiment={experiment_name}")
    except Exception as e:
        logger.error(f"[ERROR] fail to training model")
        logger.error(f"[ERROR] {e}")        