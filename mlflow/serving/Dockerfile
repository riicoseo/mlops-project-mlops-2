FROM python:3.10.13-slim
WORKDIR /app

# 필수 패키지 설치 (git, libgomp1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
RUN pip install mlflow==3.1.4 pyyaml

# 스크립트 복사
COPY convert_conda_to_requirements.py /app/convert_conda_to_requirements.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x entrypoint.sh

EXPOSE 5001
ENTRYPOINT ["sh", "/app/entrypoint.sh"]
