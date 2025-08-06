import os
import shutil

from fastapi import APIRouter, HTTPException
from src.ml.loader import reload_model, get_model_info
from src.utils.logger import get_logger
from data_prepare.main import run_popular_movie_crawler
from src.utils.utils import project_path
from src.main import run_train
 
logger = get_logger(__name__)
router = APIRouter(prefix="/airflow")

@router.post("/daily/retrain")
def daily_retrain():

    try:
        # step1.
        run_popular_movie_crawler()

        # step2.
        cache_path = os.path.join(project_path(),"src","dataset","cache")
        if os.path.exists(cache_path):
            shutil.rmtree(cache_path)
            print(f"cache 폴더 삭제 완료: {cache_path}")
        else:
            print(f"cache 폴더가 존재하지 않습니다: {cache_path}")


        # step3.
        run_train("lightgbm")
        run_train("randomforest")
        run_train("xgboost") 

        
        return {
            "status": "success", 
            "message": "Success to daily retrain",
        }
        
    except Exception as e:
        logger.error(f"[ERROR] : {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"airflow daily retrain 오류가 발생했습니다: {str(e)}"
        )
