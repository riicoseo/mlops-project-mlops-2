import os
import sys
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))))
)

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional, List
from src.ml.loader import load_mlflow_model
import pandas as pd 
import numpy as np

from src.ml.loader import get_model

router = APIRouter(prefix="/predict")

class PredictRequest(BaseModel):
    adult: Optional[int] = Field(None, description="성인영화 여부 (0/1)")
    video: Optional[int] = Field(None, description="비디오 여부 (0/1)")
    original_language: Optional[str] = Field(None, description="원어(코드)")
    overview: Optional[str] = Field(None, description="줄거리")
    genre_ids: Optional[List[str]] = Field(None, description="장르 id 리스트")

class PredictResponse(BaseModel):
    pred: float

@router.post("/json", response_model=PredictResponse)
async def predict_json(req: PredictRequest):
    # 입력값 결손 시 기본값 보정
    features = {
        "adult": req.adult if req.adult is not None else 0,
        "video": req.video if req.video is not None else 0,
        "original_language": req.original_language if req.original_language is not None else "en",
        "overview": req.overview if req.overview is not None else "",
        "genres": req.genre_ids if req.genre_ids is not None else [],
    }
    features = pd.DataFrame([features])

    model = get_model()
    pred = model.predict(features)
    return PredictResponse(pred=pred)
