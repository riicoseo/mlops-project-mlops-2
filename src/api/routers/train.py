from fastapi import APIRouter, BackgroundTasks, status
from datetime import datetime, timezone, timedelta
from pydantic import BaseModel

from src.services.train_service import run_training_job
# from src.main import run_train

router = APIRouter(prefix="/train")

class TrainRequest(BaseModel):
    model_name: str = "lightgbm"
    training_params: dict ={}    

    
@router.post("/")
async def train_model(request: TrainRequest, background_tasks: BackgroundTasks):

    background_tasks.add_task(
        run_training_job,
        model_name=request.model_name,
        **request.training_params
    )

    KST = timezone(timedelta(hours=9))
    timestamp = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")

    return {
        "status": status.HTTP_202_ACCEPTED,
        "success": True,
        "message": "start model training at background task.",
        "timestamp": timestamp
    }    

