import os
import sys
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))))
)

# Pandas 호환성 문제 해결
import warnings
warnings.filterwarnings('ignore')
from src.api import state


try:
    import pandas as pd
    
    # pandas 버전 호환성 패치
    try:
        # 구버전 pandas에서 missing된 모듈들을 임시로 생성
        if not hasattr(pd.core.indexes, 'numeric'):
            import types
            pd.core.indexes.numeric = types.ModuleType('numeric')
            
        # 필요한 인덱스 클래스들을 numeric 모듈에 추가
        if hasattr(pd, 'Int64Index'):
            pd.core.indexes.numeric.Int64Index = pd.Int64Index
        if hasattr(pd, 'Float64Index'):
            pd.core.indexes.numeric.Float64Index = pd.Float64Index
        if hasattr(pd, 'UInt64Index'):
            pd.core.indexes.numeric.UInt64Index = pd.UInt64Index
            
    except Exception as e:
        print(f"Pandas 호환성 패치 실패 (무시됨): {e}")
        pass
        
    print(f"✅ Pandas 로딩 성공 (버전: {pd.__version__})")
    
except ImportError as e:
    print(f"❌ Pandas import 실패: {e}")
    raise

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import numpy as np
import json

# 팀원이 업데이트한 모듈들 import (try-catch로 안전하게)
try:
    # from src.ml.loader import get_model
    from src.dataset.movie_rating import get_genre_decode
    from src.utils.logger import get_logger
    print("✅ 모든 모듈 import 성공")
except Exception as e:
    print(f"❌ 모듈 import 실패: {e}")
    raise

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
    - pandas 호환성 문제를 고려하여 안전하게 DataFrame 생성
    """
    
    # 기본값 설정
    adult_val = req.adult if req.adult is not None else 0
    video_val = req.video if req.video is not None else 0
    overview_val = req.overview if req.overview and req.overview.strip() else "영화 줄거리 정보 없음"
    original_language_val = req.original_language if req.original_language else "en"
    
    # 장르 처리
    if req.genre_ids and len(req.genre_ids) > 0:
        genre_names = convert_genre_ids_to_names(req.genre_ids)
        genres_json = json.dumps(genre_names, ensure_ascii=False)
    else:
        genres_json = '["기타"]'
    
    # DataFrame 생성 (pandas 호환성을 고려한 안전한 방법)
    model_input_dict = {
        "overview": [overview_val],                    
        "genres": [genres_json],                       
        "adult": [float(adult_val)],                   
        "video": [float(video_val)],                   
        "original_language": [original_language_val]   
    }
    
    try:
        # pandas DataFrame 생성
        model_input = pd.DataFrame(model_input_dict)
        logger.info(f"DataFrame 생성 성공: {model_input.shape}")
        return model_input
        
    except Exception as e:
        logger.error(f"DataFrame 생성 실패: {e}")
        raise HTTPException(status_code=500, detail=f"데이터 처리 중 오류: {str(e)}")

@router.post("/json", response_model=PredictResponse)
async def predict_json(req: PredictRequest):
    """
    팀원의 최신 전처리 파이프라인을 사용한 영화 평점 예측
    """
    
    try:
        logger.info(f"예측 요청 받음: {req}")
        
        # 1. 입력 데이터 전처리
        model_input = prepare_model_input_v2(req)
        logger.info(f"입력 데이터 준비 완료: {model_input.dtypes}")
        
        # 2. 모델 로드
        model = state.mlflow_model
        if model is None:
            raise HTTPException(
                status_code=500, 
                detail="모델이 로드되지 않았습니다. MLflow 서버 및 모델 등록 상태를 확인하세요."
            )
        
        # 3. 예측 수행
        logger.info("예측 시작...")
        prediction_result = model.predict(model_input)
        logger.info(f"예측 결과 (원본): {prediction_result}")
        
        # 4. 결과 처리
        if isinstance(prediction_result, (list, np.ndarray)):
            pred_value = float(prediction_result[0])
        else:
            pred_value = float(prediction_result)
        
        # 5. 값 범위 조정
        pred_value = max(0.0, min(10.0, pred_value))
        
        logger.info(f"예측 완료! 최종 결과: {pred_value:.2f}")
        
        # 6. 응답 생성
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
            message=f"예측 완료: {pred_value:.2f}점",
            input_info=input_summary
        )
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"예측 중 오류 발생: {error_msg}")
        logger.error(f"오류 타입: {type(e).__name__}")
        
        raise HTTPException(
            status_code=500, 
            detail=f"예측 처리 중 오류: {error_msg}"
        )

@router.get("/health")
async def predict_health():
    """예측 서비스 상태 확인 - pandas 호환성 포함"""
    try:
        # pandas 버전 확인
        pandas_version = pd.__version__
        
        # 모델 상태 확인
        model = state.mlflow_model
        model_status = "loaded" if model is not None else "not_loaded"
        
        # 장르 디코딩 확인
        try:
            genre_decode = get_genre_decode()
            genre_count = len(genre_decode) if genre_decode else 0
        except:
            genre_count = 0
        
        return {
            "status": "healthy",
            "model_status": model_status,
            "pandas_version": pandas_version,
            "genre_decode_count": genre_count,
            "service": "predict",
            "pipeline": "팀원 최신 전처리 파이프라인 연동",
            "version": "v2 - pandas compatibility fixed"
        }
        
    except Exception as e:
        return {
            "status": "error", 
            "error": str(e),
            "service": "predict"
        }

@router.get("/sample")
async def predict_sample():
    """샘플 데이터로 예측 테스트"""
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
    """장르 정보 조회"""
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