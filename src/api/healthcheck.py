import os
from fastapi import APIRouter
from src.ml.loader import get_model



project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
frontend_path = os.path.join(project_root, "frontend")

router = APIRouter()

@router.get("/health")
async def health_check():
    """서버 상태 확인"""
    try:
        model = get_model()
        model_loaded = model is not None
    except:
        model_loaded = False
        
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "service": "영화 평점 예측 서비스",
        "frontend_available": os.path.exists(frontend_path)
    }