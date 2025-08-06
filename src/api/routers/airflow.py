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

@router.post("/crawling")
def airflow_crawling():

    try:
        # step1. crawler task
        run_popular_movie_crawler()

        # PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        # popular_json_path = os.path.join(PROJECT_ROOT, "data_prepare", "result", "popular.json")

        # original_path =project_path() + "/result/popular.json"
        # shutil.move(original_path, popular_json_path)
        
        popular_json_path = os.path.join(project_path(), "data_prepare", "result", "popular.json")  
        

        if not os.path.exists(popular_json_path):
            raise FileNotFoundError(f"popular.json이 생성되지 않았습니다!! : {popular_json_path}")

    
        return {
            "status": "success", 
            "message": "Success to airflow_crawling",
        }
        
    except Exception as e:
        logger.error(f"[ERROR] : {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"airflow_crawling 오류가 발생했습니다: {str(e)}"
        )



@router.post("/train")
def airflow_train():

    try:
        # step2. rm cache file
        cache_path = os.path.join(project_path(),"src","dataset","cache")
        if os.path.exists(cache_path):
            shutil.rmtree(cache_path)
            print(f"cache 폴더 삭제 완료: {cache_path}")
        else:
            print(f"cache 폴더가 존재하지 않습니다: {cache_path}")

        # step3. preprocess and model train
        run_train("lightgbm")
        run_train("randomforest")
        run_train("xgboost") 

        return {
            "status": "success", 
            "message": "Success to airflow_train",
        }
        
    except Exception as e:
        logger.error(f"[ERROR] : {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"airflow_train 오류가 발생했습니다: {str(e)}"
        )
