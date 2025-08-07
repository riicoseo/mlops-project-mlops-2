import mlflow
from mlflow.tracking import MlflowClient

from src.utils.logger import get_logger
from src.ml.config import init_mlflow
from src.api import state

logger = get_logger(__name__)

def load_mlflow_model(model_uri: str):
    """MLflow에서 등록된 모델 로드 (팀원 업데이트 버전)"""
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
        # 팀원 코드에서는 예외를 다시 발생시키므로 동일하게 처리
        raise

def get_model():
    
    """
    팀원 업데이트 버전에 맞춘 모델 로더
    - 로컬 폴백 기능 제거 (팀원 의도에 맞춤)
    - MLflow 모델만 사용
    """
    if state.mlflow_model is None:
        try:
            logger.info("MLflow 모델 로드 시도...")
            state.mlflow_model = load_mlflow_model("models:/best_model/Production")
            logger.info("✅ MLflow 모델 로드 성공!")
        except Exception as e:
            logger.error(f"❌ MLflow 모델 로드 실패: {e}")
            logger.error("모델이 MLflow에 등록되어 있고 'best_model' 별칭으로 Production 단계에 있는지 확인하세요.")
            # 팀원 코드와 동일하게 None을 반환하지 않고 예외 발생
            raise
    
    return state.mlflow_model

def reload_model():
    """모델 재로드 (캐시 초기화)"""
    state.mlflow_model = None
    logger.info("모델 캐시 초기화 완료")
    return get_model()

def get_model_info():
    """현재 로드된 모델 정보 반환"""
    try:
        model = state.mlflow_model
        if model is None:
            return {"status": "no_model", "type": None}
        
        # MLflow 모델 정보
        return {
            "status": "loaded",
            "type": "mlflow",
            "run_id": getattr(model, 'run_id', None),
            "run_name": getattr(model, 'run_name', None),
            "model_timestamp": getattr(model, 'model_timestamp', None),
            "version": "updated_by_teammate"
        }
    except Exception as e:
        return {
            "status": "error",
            "type": None,
            "error": str(e)
        }