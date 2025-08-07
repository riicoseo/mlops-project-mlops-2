from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request

from src.utils.logger import get_logger

logger = get_logger(__name__)

def register_middleware(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )

    @app.middleware("http")
    async def log_req_res(request: Request, call_next):
        logger.info(f"[Request] {request.method} {request.url}")
        response = await call_next(request)
        logger.info(f"[Response] {request.method} {request.url} , [Status_code] {response.status_code}")
        return response
    
    
