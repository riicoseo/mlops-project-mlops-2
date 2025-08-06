import os
import sys
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))))
)

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import pandas as pd 
import numpy as np
import json

# 팀원이 업데이트한 모듈들 import
from src.ml.loader import get_model
from src.dataset.movie_rating import get_genre_decode
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/predict")

class PredictRequest(BaseModel):
    """영화 정보 예측 요청 모델 - 팀원 최신 코드와 호환"""
    adult: Optional[int] = Field(None, description="성인영화 여부 (0: 아니오, 1: 예)")
    video: Optional[int] = Field(None, description="비디오 여부 (0: 아니오, 1: 예)") 
    original_language: Optional[str] = Field(None, description="원어 코드 (ko, en, ja, zh 등)")
    genre_ids: Optional[List[int]] = Field(None, description="장르 ID 리스트")
    overview: Optional[str] = Field(None, description="영화 줄거리")
    release_date: Optional[str] = Field(None, description="개봉일 (YYYY-MM-DD)")

class PredictResponse(BaseModel):
    """예측 결과 응답 모델"""
    pred: float = Field(..., description="예측된 평점 (0-10)")
    status: str = Field("success", description="요청 처리 상태")
    message: Optional[str] = Field(None, description="추가 메시지")
    input_info: Optional[dict] = Field(None, description="처리된 입력 정보")

def convert_genre_ids_to_names(genre_ids: List[int]) -> List[str]:
    """
    팀원의 장르 디코딩 로직을 사용하여 장르 ID를 이름으로 변환
    """
    try:
        # 팀원이 만든 장르 디코딩 함수 사용
        genre_decode = get_genre_decode()
        
        # ID -> 이름 매핑이 반대로 되어있을 수 있으므로 확인
        if isinstance(list(genre_decode.keys())[0], str):
            # 이름 -> ID 매핑인 경우, 역변환
            id_to_name = {v: k for k, v in genre_decode.items()}
            genre_names = [id_to_name.get(gid, f"장르_{gid}") for gid in genre_ids]
        else:
            # ID -> 이름 매핑인 경우
            genre_names = [genre_decode.get(gid, f"장르_{gid}") for gid in genre_ids]
            
        return genre_names
        
    except Exception as e:
        logger.warning(f"장르 디코딩 실패: {e}")
        # 폴백: 기본 장르 매핑 사용
        fallback_mapping = {
            28: "액션", 35: "코미디", 18: "드라마", 16: "애니메이션",
            14: "판타지", 53: "스릴러", 10751: "가족", 27: "공포",
            10402: "음악", 9648: "미스터리", 10749: "로맨스", 878: "SF",
            10770: "TV영화", 37: "서부", 10752: "전쟁", 99: "다큐멘터리"
        }
        return [fallback_mapping.get(gid, f"장르_{gid}") for gid in genre_ids]

def prepare_model_input_v2(req: PredictRequest) -> pd.DataFrame:
    """
    팀원의 최신 MovieRatingModel에 맞춰 입력 데이터 준비
    - is_english와 overview_clean은 모델 내부에서 처리하므로 제외
    - original_language는 그대로 전달 (모델에서 is_english로 변환)
    """
    
    # 기본값 설정
    adult_val = req.adult if req.adult is not None else 0
    video_val = req.video if req.video is not None else 0
    overview_val = req.overview if req.overview and req.overview.strip() else "영화 줄거리 정보 없음"
    original_language_val = req.original_language if req.original_language else "en"
    
    # 장르 처리 - 팀원의 최신 MovieRatingModel.predict() 형식에 맞춤
    if req.genre_ids and len(req.genre_ids) > 0:
        # 장르 ID -> 장르 이름 변환
        genre_names = convert_genre_ids_to_names(req.genre_ids)
        # JSON 문자열 형태로 저장 (팀원 코드에서 ast.literal_eval로 파싱)
        genres_json = json.dumps(genre_names, ensure_ascii=False)
    else:
        genres_json = '["기타"]'  # 기본 장르
    
    # 팀원의 최신 MovieRatingModel이 기대하는 정확한 컬럼명과 형식
    model_input = {
        "overview": overview_val,                    # 줄거리 (모델에서 clean_korean_text 적용)
        "genres": genres_json,                       # JSON 문자열로 된 장르 리스트
        "adult": float(adult_val),                   # 성인영화 여부
        "video": float(video_val),                   # 비디오 여부  
        "original_language": original_language_val   # 원어 (모델에서 is_english로 변환)
    }
    
    logger.info(f"모델 입력 데이터 준비 완료 (v2): {model_input}")
    return pd.DataFrame([model_input])

@router.post("/json", response_model=PredictResponse)
async def predict_json(req: PredictRequest):
    """
    팀원의 최신 전처리 파이프라인을 사용한 영화 평점 예측
    
    - **adult**: 성인영화 여부 (0 또는 1)
    - **video**: 비디오 여부 (0 또는 1)  
    - **original_language**: 원어 코드 (ko, en, ja, zh 등)
    - **genre_ids**: 장르 ID 배열 [28, 35, 18 등]
    - **overview**: 영화 줄거리 텍스트
    - **release_date**: 개봉일 (현재 사용하지 않음)
    """
    
    try:
        logger.info(f"예측 요청 받음: {req}")
        
        # 1. 입력 데이터를 팀원의 최신 형식으로 전처리
        model_input = prepare_model_input_v2(req)
        
        # 2. 팀원이 업데이트한 MLflow 모델 로드
        model = get_model()
        if model is None:
            raise HTTPException(
                status_code=500, 
                detail="모델이 로드되지 않았습니다. MLflow 서버 및 모델 등록 상태를 확인하세요."
            )
        
        # 3. 팀원의 최신 MovieRatingModel.predict() 메서드 호출
        logger.info("팀원의 최신 전처리 파이프라인으로 예측 시작...")
        
        # 팀원의 최신 코드는 context를 첫 번째 인자로 받지 않음
        prediction_result = model.predict(model_input)
        
        # 4. 예측 결과 처리
        if isinstance(prediction_result, (list, np.ndarray)):
            pred_value = float(prediction_result[0])
        else:
            pred_value = float(prediction_result)
        
        # 5. 예측값 범위 조정 (0-10 범위로 클리핑)
        pred_value = max(0.0, min(10.0, pred_value))
        
        logger.info(f"예측 완료! 결과: {pred_value:.2f}")
        
        # 6. 입력 정보 요약 (디버깅용)
        input_summary = {
            "processed_overview": model_input.iloc[0]["overview"][:50] + "..." if len(model_input.iloc[0]["overview"]) > 50 else model_input.iloc[0]["overview"],
            "processed_genres": model_input.iloc[0]["genres"],
            "processed_adult": model_input.iloc[0]["adult"],
            "processed_video": model_input.iloc[0]["video"],
            "processed_original_language": model_input.iloc[0]["original_language"]
        }
        
        return PredictResponse(
            pred=round(pred_value, 2),
            status="success",
            message=f"팀원의 최신 전처리 파이프라인으로 예측 완료: {pred_value:.2f}점",
            input_info=input_summary
        )
        
    except Exception as e:
        logger.error(f"예측 중 오류 발생: {str(e)}")
        logger.error(f"오류 상세: {type(e).__name__}")
        
        raise HTTPException(
            status_code=500, 
            detail=f"예측 처리 중 오류가 발생했습니다: {str(e)}"
        )

@router.get("/health")
async def predict_health():
    """예측 서비스 상태 확인"""
    try:
        model = get_model()
        model_status = "loaded" if model is not None else "not_loaded"
        
        # 장르 디코딩 정보도 확인
        try:
            genre_decode = get_genre_decode()
            genre_count = len(genre_decode) if genre_decode else 0
        except:
            genre_count = 0
        
        return {
            "status": "healthy",
            "model_status": model_status,
            "genre_decode_count": genre_count,
            "service": "predict",
            "pipeline": "팀원 최신 전처리 파이프라인 연동",
            "version": "v2 - updated for latest MovieRatingModel"
        }
    except Exception as e:
        return {
            "status": "error", 
            "error": str(e),
            "service": "predict"
        }

@router.get("/sample")
async def predict_sample():
    """팀원 최신 파이프라인으로 샘플 데이터 예측 테스트"""
    sample_request = PredictRequest(
        adult=0,
        video=0,
        original_language="en",
        genre_ids=[28, 53],  # 액션, 스릴러
        overview="액션과 스릴러가 가득한 흥미진진한 영화입니다. 주인공이 위험한 상황에서 벗어나기 위해 노력하는 이야기입니다.",
        release_date="2024-01-01"
    )
    
    return await predict_json(sample_request)

@router.get("/genre-info")
async def get_genre_info():
    """사용 가능한 장르 정보 조회"""
    try:
        genre_decode = get_genre_decode()
        return {
            "genre_decode": genre_decode,
            "available_genres": list(genre_decode.keys()) if genre_decode else [],
            "total_count": len(genre_decode) if genre_decode else 0
        }
    except Exception as e:
        return {
            "error": str(e),
            "genre_decode": {},
            "available_genres": [],
            "total_count": 0
        }