
from fastapi import FastAPI
from contextlib import asynccontextmanager

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

    mlflow_model = load_mlflow_model(model_uri="models:/best_model/Production")

    logger.info(f"[END] mlflow model loaded")

    yield

    logger.info("서버 종료")


app = FastAPI(lifespan=lifespan)

register_middleware(app)

app.include_router(train.router)
app.include_router(predict.router)
app.include_router(reload.router)


