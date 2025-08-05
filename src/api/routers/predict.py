from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional, List
from src.ml.loader import load_mlflow_model

router = APIRouter(prefix="/predict")

class PredictRequest(BaseModel):
    adult: Optional[int] = Field(None, description="성인영화 여부 (0/1)")
    genre_ids: Optional[List[int]] = Field(None, description="장르 id 리스트")
    original_language: Optional[str] = Field(None, description="원어(코드)")
    overview: Optional[str] = Field(None, description="줄거리")
    release_date: Optional[str] = Field(None, description="개봉일 (YYYY-MM-DD)")
    video: Optional[int] = Field(None, description="비디오 여부 (0/1)")

class PredictResponse(BaseModel):
    pred: float

@router.post("/json", response_model=PredictResponse)
async def predict_json(req: PredictRequest):
    # 입력값 결손 시 기본값 보정
    features = {
        "adult": req.adult if req.adult is not None else 0,
        "genre_ids": req.genre_ids if req.genre_ids is not None else [],
        "original_language": req.original_language if req.original_language is not None else "en",
        "overview": req.overview if req.overview is not None else "",
        "release_date": req.release_date if req.release_date is not None else "2000-01-01",
        "video": req.video if req.video is not None else 0
    }

    # (실제 예측로직: 예시는 pred=5.0 고정)
    # 실제로는 features dict을 전처리 후 모델에 넣어서 예측
    pred = 5.0

    return PredictResponse(pred=pred)
