from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from contextlib import asynccontextmanager
import os
import sys

# 프로젝트 루트 경로 추가
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.api.middleware import register_middleware
from src.api.routers import train, predict, reload
from src.ml.loader import get_model
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
        mlflow_model = get_model()
        logger.info(f"[END] mlflow model loaded successfully")
    except Exception as e:
        logger.warning(f"MLflow 모델 로딩 실패: {e}")
        logger.info("모델 없이 서버를 시작합니다.")
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

logger.info(f"프로젝트 루트: {project_root}")
logger.info(f"프론트엔드 경로: {frontend_path}")
logger.info(f"프론트엔드 경로 존재 여부: {os.path.exists(frontend_path)}")

# 정적 파일 서빙 (에러 처리 추가)
if os.path.exists(os.path.join(frontend_path, "assets")):
    app.mount("/static", StaticFiles(directory=os.path.join(frontend_path, "assets")), name="static")
    logger.info("✅ 정적 파일 경로 마운트 완료")
else:
    logger.warning("⚠️ frontend/assets 디렉토리가 존재하지 않습니다.")

# 간단한 HTML 응답 함수들
def get_simple_html(title, message):
    return f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <title>{title}</title>
        <style>
            body {{ font-family: 'Malgun Gothic', sans-serif; 
                   background: #f5f6fa; 
                   display: flex; 
                   justify-content: center; 
                   align-items: center; 
                   height: 100vh; 
                   margin: 0; }}
            .container {{ background: #fff; 
                         border-radius: 12px; 
                         padding: 40px 30px; 
                         box-shadow: 0 2px 12px rgba(0,0,0,0.07); 
                         text-align: center; }}
            h1 {{ color: #27408b; margin-bottom: 24px; }}
            p {{ color: #666; line-height: 1.6; }}
            .btn {{ display: inline-block; 
                   background: #27408b; 
                   color: #fff; 
                   padding: 12px 24px; 
                   border: none; 
                   border-radius: 6px; 
                   text-decoration: none; 
                   margin: 10px; }}
            .btn:hover {{ background: #4169e1; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{title}</h1>
            <p>{message}</p>
            <a href="/docs" class="btn">API 문서 보기</a>
            <a href="/predict/sample" class="btn">샘플 예측 테스트</a>
        </div>
    </body>
    </html>
    """

# 웹페이지 라우트 추가
@app.get("/")
async def root():
    """메인 페이지"""
    login_html_path = os.path.join(frontend_path, "login.html")
    
    if os.path.exists(login_html_path):
        return FileResponse(login_html_path)
    else:
        # 파일이 없으면 간단한 HTML 응답
        html_content = get_simple_html(
            "영화 평점 예측 서비스",
            "MLOps 프로젝트의 영화 평점 예측 서비스입니다.<br>API 문서에서 예측 기능을 테스트해보세요!"
        )
        return HTMLResponse(content=html_content)

@app.get("/login")
@app.get("/login.html")
async def login_page():
    """로그인 페이지"""
    login_html_path = os.path.join(frontend_path, "login.html")
    
    if os.path.exists(login_html_path):
        return FileResponse(login_html_path)
    else:
        html_content = """
        <!DOCTYPE html>
        <html lang="ko">
        <head>
            <meta charset="UTF-8">
            <title>로그인 - 영화 평점 예측</title>
            <style>
                body { font-family: 'Malgun Gothic', sans-serif; background: #f5f6fa; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0;}
                .login-container { background: #fff; border-radius: 12px; padding: 40px 30px; box-shadow: 0 2px 12px rgba(0,0,0,0.07); min-width: 300px;}
                h2 { margin-bottom: 24px; color: #27408b; text-align: center;}
                label { display: block; margin-bottom: 8px; }
                input[type="text"], input[type="password"] { width: 100%; padding: 10px; margin-bottom: 20px; border-radius: 6px; border: 1px solid #ccc; box-sizing: border-box;}
                button { width: 100%; background: #27408b; color: #fff; padding: 12px; border: none; border-radius: 6px; font-size: 16px; cursor: pointer; }
                button:hover { background: #4169e1;}
            </style>
        </head>
        <body>
            <div class="login-container">
                <h2>로그인</h2>
                <form id="login-form">
                    <label for="userid">아이디</label>
                    <input type="text" id="userid" required>
                    <label for="userpw">비밀번호</label>
                    <input type="password" id="userpw" required>
                    <button type="submit">로그인</button>
                </form>
            </div>
            <script>
                document.getElementById('login-form').addEventListener('submit', function(e){
                    e.preventDefault();
                    window.location.href = '/survey';
                });
            </script>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)

@app.get("/survey")
@app.get("/survey.html")
async def survey_page():
    """영화 정보 입력 설문 페이지"""
    survey_html_path = os.path.join(frontend_path, "survey.html")
    
    if os.path.exists(survey_html_path):
        return FileResponse(survey_html_path)
    else:
        html_content = """
        <!DOCTYPE html>
        <html lang="ko">
        <head>
            <meta charset="UTF-8">
            <title>영화 정보 입력 - 평점 예측</title>
            <style>
                body { font-family: 'Malgun Gothic', sans-serif; background: #f5f6fa; display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; padding: 20px; box-sizing: border-box;}
                .survey-container { background: #fff; border-radius: 12px; padding: 40px 30px; box-shadow: 0 2px 12px rgba(0,0,0,0.07); max-width: 500px; width: 100%;}
                h2 { margin-bottom: 24px; color: #27408b; text-align: center;}
                label { display: block; margin-bottom: 8px; margin-top: 16px; font-weight: bold;}
                input, select, textarea { width: 100%; padding: 8px; border-radius: 6px; border: 1px solid #ccc; margin-bottom: 10px; box-sizing: border-box;}
                .checkbox-group { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px;}
                .checkbox-group label { margin: 0; font-weight: normal; display: flex; align-items: center;}
                .checkbox-group input { width: auto; margin-right: 5px;}
                button { width: 100%; background: #27408b; color: #fff; padding: 12px; border: none; border-radius: 6px; font-size: 16px; cursor: pointer; margin-top: 10px;}
                button:hover { background: #4169e1;}
                .result { margin-top: 20px; padding: 15px; background: #e8f5e8; border-radius: 6px; text-align: center;}
            </style>
        </head>
        <body>
            <div class="survey-container">
                <h2>영화 정보 입력</h2>
                <form id="survey-form">
                    <label>성인영화 여부</label>
                    <select name="adult">
                        <option value="">선택안함</option>
                        <option value="0">아니오</option>
                        <option value="1">예</option>
                    </select>
                    
                    <label>장르 (복수 선택 가능)</label>
                    <div class="checkbox-group">
                        <label><input type="checkbox" name="genre_ids" value="28"> 액션</label>
                        <label><input type="checkbox" name="genre_ids" value="35"> 코미디</label>
                        <label><input type="checkbox" name="genre_ids" value="18"> 드라마</label>
                        <label><input type="checkbox" name="genre_ids" value="16"> 애니메이션</label>
                        <label><input type="checkbox" name="genre_ids" value="14"> 판타지</label>
                        <label><input type="checkbox" name="genre_ids" value="53"> 스릴러</label>
                    </div>
                    
                    <label>원어(언어)</label>
                    <select name="original_language">
                        <option value="">선택안함</option>
                        <option value="ko">한국어</option>
                        <option value="en">영어</option>
                        <option value="ja">일본어</option>
                        <option value="zh">중국어</option>
                    </select>
                    
                    <label>줄거리</label>
                    <textarea name="overview" rows="3" placeholder="영화 줄거리를 간단히 입력해주세요"></textarea>
                    
                    <label>비디오 여부</label>
                    <select name="video">
                        <option value="">선택안함</option>
                        <option value="0">아니오</option>
                        <option value="1">예</option>
                    </select>
                    
                    <button type="submit">예측하기</button>
                </form>
                
                <div id="result" class="result" style="display: none;">
                    <h3>예측 결과</h3>
                    <p id="pred-value"></p>
                </div>
            </div>
            
            <script>
                document.getElementById('survey-form').addEventListener('submit', async function(e){
                    e.preventDefault();
                    
                    const form = e.target;
                    const adult = form.adult.value === "" ? null : Number(form.adult.value);
                    const original_language = form.original_language.value === "" ? null : form.original_language.value;
                    const overview = form.overview.value.trim() === "" ? null : form.overview.value.trim();
                    const video = form.video.value === "" ? null : Number(form.video.value);
                    
                    const genreNodes = form.querySelectorAll('input[name="genre_ids"]:checked');
                    const genre_ids = Array.from(genreNodes).map(x=>Number(x.value));

                    const payload = {
                        adult, 
                        genre_ids: genre_ids.length ? genre_ids : null, 
                        original_language, 
                        overview, 
                        video
                    };

                    try {
                        const res = await fetch('/predict/json', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify(payload)
                        });
                        
                        if (!res.ok) throw new Error('예측 요청 실패');
                        
                        const data = await res.json();
                        document.getElementById('pred-value').innerText = `예상 평점: ${data.pred}점`;
                        document.getElementById('result').style.display = 'block';
                        
                    } catch (err) {
                        alert('예측 실패: ' + err.message);
                    }
                });
            </script>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)

@app.get("/result")
@app.get("/result.html")
async def result_page():
    """예측 결과 페이지"""
    return HTMLResponse(content=get_simple_html(
        "예측 결과",
        "예측이 완료되었습니다!<br>다시 예측하려면 설문 페이지로 이동하세요."
    ))

# 헬스체크 엔드포인트
@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    try:
        model = get_model()
        model_loaded = model is not None
    except:
        model_loaded = False
        
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "service": "영화 평점 예측 서비스",
        "frontend_available": os.path.exists(frontend_path)
    }

# 전역 변수로 mlflow_model을 앱에서 접근 가능하게 만들기
def get_mlflow_model():
    """MLflow 모델 인스턴스 반환"""
    return mlflow_model