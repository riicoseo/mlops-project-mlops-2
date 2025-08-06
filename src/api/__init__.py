from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import os
import sys

# 프로젝트 루트 경로 추가
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.api.middleware import register_middleware
from src.api.routers import train, predict, reload
from src.ml.loader import load_mlflow_model
from src.utils.logger import get_logger

logger = get_logger(__name__)

mlflow_model = None

@asynccontextmanager
async def lifespan(app):
    global mlflow_model
    logger.info("서버 시작 Step")
    logger.info("[START] loading mlflow model")

    # MLflow 모델 로딩 (에러 처리 추가)
    try:
        mlflow_model = load_mlflow_model(model_uri="models:/best_model/Production")
        logger.info(f"[END] mlflow model loaded successfully")
    except Exception as e:
        logger.warning(f"MLflow 모델 로딩 실패: {e}")
        logger.info("기본 모델로 폴백합니다.")
        mlflow_model = None

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

# 프로젝트 루트 경로 계산
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
frontend_path = os.path.join(project_root, "frontend")

# 정적 파일 서빙 (CSS, JS, 이미지 등)
app.mount("/static", StaticFiles(directory=os.path.join(frontend_path, "assets")), name="static")

# 웹페이지 라우트 추가
@app.get("/")
async def root():
    """메인 페이지 - 로그인 페이지로 리다이렉트"""
    return FileResponse(os.path.join(frontend_path, "login.html"))

@app.get("/login")
@app.get("/login.html")
async def login_page():
    """로그인 페이지"""
    return FileResponse(os.path.join(frontend_path, "login.html"))

@app.get("/survey")
@app.get("/survey.html")
async def survey_page():
    """영화 정보 입력 설문 페이지"""
    return FileResponse(os.path.join(frontend_path, "survey.html"))

@app.get("/result")
@app.get("/result.html")
async def result_page():
    """예측 결과 페이지"""
    return FileResponse(os.path.join(frontend_path, "result.html"))

# 헬스체크 엔드포인트
@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    return {
        "status": "healthy",
        "model_loaded": mlflow_model is not None,
        "service": "영화 평점 예측 서비스"
    }

# 전역 변수로 mlflow_model을 앱에서 접근 가능하게 만들기
def get_mlflow_model():
    """MLflow 모델 인스턴스 반환"""
    return mlflow_model