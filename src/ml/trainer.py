import mlflow
import io
import os
import joblib
import mlflow.lightgbm
import lightgbm as lgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime, timezone, timedelta
from lightgbm import early_stopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from mlflow.models import infer_signature

from src.ml.config import init_mlflow
from src.data.preprocess import load_data, preprocess
from src.utils.logger import get_logger

logger = get_logger(__name__)

def train_and_log_model(name:str, **kwargs):
    init_mlflow(experiment_name = name)

    # 현재는 일단 고정 경로로 처리 > 추후 변수 처리화 필요! 
    raw_df = pd.read_csv("/home/ubuntu/workspace/yoon/mlops-project-mlops-2/data/raw/discover_movies.csv", encoding='utf-8-sig',  engine="python")
    df = preprocess(raw_df)

    X = df.drop(columns=["vote_average"])
    y = df["vote_average"]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. 기본 파라미터 + 요청에서 받은 파라미터 덮어쓰기
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "seed": 42,
        "verbose": -1
    }
    params.update(training_params)

    # 3. LightGBM Dataset
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_valid, y_valid)
    
    KST = timezone(timedelta(hours=9))
    timestamp = datetime.now(KST).strftime("%Y%m%d_%H:%M:%S")
    run_name = f"lgbm_training_{timestamp}"

    with mlflow.start_run(run_name=run_name) as run:
        evals_result = {}
        model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_valid],
            num_boost_round=4000,
            callbacks=[
                early_stopping(stopping_rounds=100),
                lgb.record_evaluation(evals_result)
            ]
        )

        # 4. 검증 RMSE 계산
        preds = model.predict(X_valid)
        rmse = np.sqrt(mean_squared_error(y_valid, preds))
        logger.info(f"Validation RMSE: {rmse}")

        # 5. 학습 곡선 저장 & MLflow 업로드
        plt.figure()
        lgb.plot_metric(evals_result, metric='rmse')
        plt.title("Training Curve (RMSE)")
        curve_path = f"plots/training_curve_{run_name}.png"
        plt.savefig(curve_path)
        plt.close()
        mlflow.log_artifact(curve_path)

        # 6. MLflow 메트릭 & 파라미터 로깅
        mlflow.log_params(params)
        mlflow.log_metric("rmse", rmse)
        mlflow.set_tag("model_timestamp", timestamp)

        # 7. 모델 저장
        
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.lightgbm.log_model(
            model,
            name="lgb_model",
            input_example=X_train.iloc[:5],
            signature=signature
        )

        # 8-1. 로컬 저장
        os.makedirs("src/models", exist_ok=True)
        model.save_model("src/models/lgb_model.txt")
        joblib.dump(X_train.columns.tolist(), "src/models/feature_list.pkl")

        # 8-1. mlflow artifact 에 저장
        mlflow.log_artifact("src/models/lgb_model.txt")
        mlflow.log_artifact("src/models/feature_list.pkl")

        # 9. 전처리된 데이터 버퍼에 저장 후 mlflow 에 저장 (재현성)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        mlflow.log_text(csv_buffer.getvalue(), "processed_train.csv")

        logger.info(f"[{run_name}] RMSE: {rmse:.4f}")





def train_and_log_model_2(model_name, name:str, training_params:dict):
    init_mlflow(experiment_name = name)

    # 현재는 일단 고정 경로로 처리 > 추후 변수 처리화 필요! 
    raw_df = pd.read_csv("/home/ubuntu/workspace/yoon/mlops-project-mlops-2/data/raw/discover_movies.csv", encoding='utf-8-sig',  engine="python")
    df = preprocess(raw_df)

    X = df.drop(columns=["vote_average"])
    y = df["vote_average"]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. 기본 파라미터 + 요청에서 받은 파라미터 덮어쓰기
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "seed": 42,
        "verbose": -1
    }
    params.update(training_params)

    # 3. LightGBM Dataset
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_valid, y_valid)
    
    KST = timezone(timedelta(hours=9))
    timestamp = datetime.now(KST).strftime("%Y%m%d_%H:%M:%S")
    run_name = f"lgbm_training_{timestamp}"

    with mlflow.start_run(run_name=run_name) as run:
        evals_result = {}
        model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_valid],
            num_boost_round=4000,
            callbacks=[
                early_stopping(stopping_rounds=100),
                lgb.record_evaluation(evals_result)
            ]
        )

        # 4. 검증 RMSE 계산
        preds = model.predict(X_valid)
        rmse = np.sqrt(mean_squared_error(y_valid, preds))
        logger.info(f"Validation RMSE: {rmse}")

        # 5. 학습 곡선 저장 & MLflow 업로드
        plt.figure()
        lgb.plot_metric(evals_result, metric='rmse')
        plt.title("Training Curve (RMSE)")
        curve_path = f"plots/training_curve_{run_name}.png"
        plt.savefig(curve_path)
        plt.close()
        mlflow.log_artifact(curve_path)

        # 6. MLflow 메트릭 & 파라미터 로깅
        mlflow.log_params(params)
        mlflow.log_metric("rmse", rmse)
        mlflow.set_tag("model_timestamp", timestamp)

        # 7. 모델 저장
        
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.lightgbm.log_model(
            model,
            name="lgb_model",
            input_example=X_train.iloc[:5],
            signature=signature
        )

        # 8-1. 로컬 저장
        os.makedirs("src/models", exist_ok=True)
        model.save_model("src/models/lgb_model.txt")
        joblib.dump(X_train.columns.tolist(), "src/models/feature_list.pkl")

        # 8-1. mlflow artifact 에 저장
        mlflow.log_artifact("src/models/lgb_model.txt")
        mlflow.log_artifact("src/models/feature_list.pkl")

        # 9. 전처리된 데이터 버퍼에 저장 후 mlflow 에 저장 (재현성)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        mlflow.log_text(csv_buffer.getvalue(), "processed_train.csv")

        logger.info(f"[{run_name}] RMSE: {rmse:.4f}")