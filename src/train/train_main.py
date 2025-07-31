import mlflow
import mlflow.lightgbm
import pandas as pd
import lightgbm as lgb
from lightgbm import early_stopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from data.preprocess import load_data, preprocess, save_processed_data

def train_main():
    mlflow.set_tracking_uri("http://43.200.183.125:5000/")
    mlflow.set_experiment("movie_rating_prediction")

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

    with mlflow.start_run():
        model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_valid],
            num_boost_round=3000,
            callbacks=[early_stopping(stopping_rounds=100)]
        )

        # 평가
        preds = model.predict(X_valid)
        rmse = np.sqrt(mean_squared_error(y_valid, preds))

        # MLflow 로깅 & 모델 저장

        input_example = X_train.iloc[:5]  # 5개 행 예시
        from mlflow.models import infer_signature
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.log_params(params)
        mlflow.log_metric("rmse", rmse)
        mlflow.lightgbm.log_model(
            model,
            name="lgb_model",
            input_example=input_example,
            signature=signature
        )

        # 로컬에 모델 저장
        import os
        MODEL_DIR = os.path.join("src", "models")
        os.makedirs(MODEL_DIR, exist_ok=True)
        MODEL_PATH = os.path.join(MODEL_DIR, "lgb_model.txt")
        model.save_model(MODEL_PATH)
        
        features = X_train.columns.tolist()
        import joblib
        joblib.dump(features, "src/models/feature_list.pkl")

        print(f"RMSE: {rmse:.4f}")


if __name__ == "__main__":
    train_main()
