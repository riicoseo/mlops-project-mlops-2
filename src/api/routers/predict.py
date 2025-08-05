from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/predict")

@router.post("/id/{movie_id}")
async def predict(movie_id:int):
    pred = 5.0
    return {"pred": pred}



class PredictRequest(BaseModel):
    movie_id : int

class PredictResponse(BaseModel):
    movie_id : int
    pred : float

@router.post("/json", response_model=PredictResponse)
async def predict_json(req:PredictRequest):
    movie_id = req.movie_id
    pred = 5.0
    return PredictResponse(movie_id=movie_id, pred = pred )
    # return {"movie_id": movie_id, "pred":pred}
