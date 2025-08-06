from fastapi import APIRouter, HTTPException
from src.ml.loader import reload_model, get_model_info
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/reload")

@router.post("/model")
def reload_model_endpoint():
    """
    모델 재로드 (MLflow 모델 또는 로컬 모델)
    캐시를 초기화하고 최신 모델을 다시 로드합니다.
    """
    try:
        logger.info("[START] 모델 재로드 시작")
        
        model = reload_model()
        
        if model is None:
            raise HTTPException(
                status_code=500,
                detail="모든 모델 로드 시도가 실패했습니다."
            )
        
        model_info = get_model_info()
        logger.info(f"[END] 모델 재로드 완료: {model_info}")
        
        return {
            "status": "success", 
            "message": "모델이 성공적으로 재로드되었습니다.",
            "model_info": model_info
        }
        
    except Exception as e:
        logger.error(f"모델 재로드 실패: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"모델 재로드 중 오류가 발생했습니다: {str(e)}"
        )

@router.get("/status")
def get_model_status():
    """현재 로드된 모델 상태 조회"""
    try:
        model_info = get_model_info()
        return {
            "status": "success",
            "model_info": model_info
        }
    except Exception as e:
        logger.error(f"모델 상태 조회 실패: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "model_info": {"status": "error", "type": None}
        }