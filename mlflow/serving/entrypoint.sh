#!/bin/bash
set -e

echo "[INFO] MLflow에서 Production 모델 conda.yaml 다운로드 시도..."
mlflow artifacts download \
    --artifact-uri "models:/best_model/Production" \
    --dst-path "/app" || echo "[WARN] 모델 artifacts 다운로드 실패"

# conda.yaml이 있으면 변환
if [ -f "/app/conda.yaml" ]; then
    echo "[INFO] conda.yaml 발견, requirements.txt 변환 시작..."
    python /app/convert_conda_to_requirements.py
else
    echo "[WARN] conda.yaml 없음, 기본 requirements.txt 사용"
fi

# requirements.txt가 있으면 설치
if [ -f "/app/requirements.txt" ]; then
    echo "[INFO] requirements.txt 기반 pip install..."
    pip install -r /app/requirements.txt
else
    echo "[WARN] requirements.txt 없음, mlflow만 설치된 상태로 진행"
fi


mlflow models serve \
    -m "${MODEL_URI}" \
    --no-conda \
    --host 0.0.0.0 \
    --port 5001
