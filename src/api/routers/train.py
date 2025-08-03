from fastapi import APIRouter

router = APIRouter(prefix="/train")

@router.post("/")
async def train_model():
    return {"status": "training started"}