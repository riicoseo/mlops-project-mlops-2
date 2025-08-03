from fastapi import APIRouter

router = APIRouter(prefix="/predict")

@router.post("/{movie_id}")
async def predict(movie_id:int):
    pred = 5.0
    return {"pred": pred}