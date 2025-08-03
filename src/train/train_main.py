import mlflow
import io
import mlflow.lightgbm
import pandas as pd
from datetime import datetime
import lightgbm as lgb
from lightgbm import early_stopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from src.data.preprocess import load_data, preprocess, save_processed_data
import os
import joblib

def train_main():

    mlflow.set_tracking_uri("http://14.38.177.115:5000/")
    mlflow.set_experiment("movie_rating")

    # 1. 데이터 로드 & 전처리
    raw_df = load_data()
    df = preprocess(raw_df)
    # save_processed_data(processed_df)

    X = df.drop(columns=["vote_average"])
    y = df["vote_average"]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. 하이퍼파라미터
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "seed": 42, 
        "verbose": 50
    }

    # 3. 학습
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_valid, y_valid)

    run_name = f"lgbm_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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

        # 평가
        preds = model.predict(X_valid)
        rmse = np.sqrt(mean_squared_error(y_valid, preds))

        # 7. MLflow: 학습 곡선 시각화 → artifacts 저장
        plt.figure()
        lgb.plot_metric(evals_result, metric='rmse')
        plt.title("Training Curve (RMSE)")
        curve_path = "training_curve.png"
        plt.savefig(curve_path)
        plt.close()
        mlflow.log_artifact(curve_path)

        # 9. MLflow: 파라미터 / 메트릭 기록
        mlflow.log_params(params)
        mlflow.log_metric("rmse", rmse)

  
        #  MLflow:  모델 저장
        from mlflow.models import infer_signature
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.lightgbm.log_model(
            model,
            name="lgb_model",
            input_example= X_train.iloc[:5] ,
            signature=signature
        )

        # 로컬에 모델 저장
        os.makedirs("src/models", exist_ok=True)
        model.save_model("src/models/lgb_model.txt")
        joblib.dump(X_train.columns.tolist(), "src/models/feature_list.pkl")
        mlflow.log_artifact("src/models/lgb_model.txt")
        mlflow.log_artifact("src/models/feature_list.pkl")

        # 11. 전처리된 데이터를 버퍼로 저장 → 재현성 확보
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        mlflow.log_text(csv_buffer.getvalue(), "processed_train.csv")

        print(f"RMSE: {rmse:.4f}")


if __name__ == "__main__":
    train_main()

