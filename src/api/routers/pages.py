import os

from fastapi import APIRouter
from fastapi.responses import FileResponse, HTMLResponse

from src.services.train_service import run_training_job
# from src.main import run_train

router = APIRouter(prefix="/pages")
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
frontend_path = os.path.join(project_root, "frontend")

def get_simple_html(title: str, message: str) -> str:
    return f"""<html><head><title>{title}</title></head><body><h1>{title}</h1><p>{message}</p></body></html>"""

@router.get("/")
async def root():
    path = os.path.join(frontend_path, "login.html")
    return FileResponse(path) if os.path.exists(path) else HTMLResponse(get_simple_html("홈", "서비스입니다."))

@router.get("/login")
@router.get("/login.html")
async def login_page():
    path = os.path.join(frontend_path, "login.html")
    return FileResponse(path) if os.path.exists(path) else HTMLResponse(get_simple_html("로그인", "로그인 페이지입니다."))

@router.get("/survey")
@router.get("/survey.html")
async def survey_page():
    path = os.path.join(frontend_path, "survey.html")
    return FileResponse(path) if os.path.exists(path) else HTMLResponse(get_simple_html("설문", "영화 정보를 입력하세요."))

@router.get("/easytest")
@router.get("/easytest.html")
async def survey_page():
    path = os.path.join(frontend_path, "easytest.html")
    return FileResponse(path) if os.path.exists(path) else HTMLResponse(get_simple_html("Easy Test", "모델 선택 후, 간단한 하이퍼 파라미터를 셋팅해주세요."))