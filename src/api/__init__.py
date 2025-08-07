from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from contextlib import asynccontextmanager
import os
import sys

# 프로젝트 루트 경로 추가
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.api.middleware import register_middleware
from src.api.routers import train, predict, reload, airflow, pages
from src.ml.loader import get_model
from src.utils.logger import get_logger
from src.api import state

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app):
    logger.info("서버 시작 Step")
    logger.info("[START] loading mlflow model")

    # MLflow 모델 로딩 (에러 처리 추가)
    try:
        model = get_model()
        state.mlflow_model = model
        logger.info(f"[END] mlflow model loaded successfully")
    except Exception as e:
        logger.warning(f"MLflow 모델 로딩 실패: {e}")
        logger.info("모델 없이 서버를 시작합니다.")
        state.mlflow_model = None

    yield

    logger.info("서버 종료")


# FastAPI 앱 생성
app = FastAPI(
    title="영화 평점 예측 서비스",
    description="MLOps 영화 평점 예측 API 및 웹 서비스",
    version="1.0.0",
    lifespan=lifespan
)

# 미들웨어 등록
register_middleware(app)

# API 라우터 등록
app.include_router(train.router)
app.include_router(predict.router)
app.include_router(reload.router)
app.include_router(airflow.router)
app.include_router(pages.router)

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
frontend_path = os.path.join(project_root, "frontend")
