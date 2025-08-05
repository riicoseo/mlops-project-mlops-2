import mlflow
import io
import os
import joblib
import lightgbm as lgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime, timezone, timedelta
from lightgbm import early_stopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.dataset.movie_rating import get_datasets, MovieRatingDataset, GenreEmbeddingModule
from src.evaluate.evaluate import evaluate
from src.ml.config import init_mlflow
from src.utils.logger import get_logger
from src.utils.utils import init_seed, model_dir, project_path
from src.utils.enums import ModelType

logger = get_logger(__name__)

def filter_custom_params(model, user_defined_params: dict):
    all_params = model.get_params()
    filtered = {}

    for key, value in user_defined_params.items():
        if key in all_params and all_params[key] != value:
            print(f"⚠️ 경고: '{key}' 파라미터가 모델 내부 값과 다릅니다.")
        filtered[key] = value

    return filtered

def model_save(model, all_params, model_params, tf_idf, embedding_module, genre2idx,
               timestamp, rmse, update_checkpoint=True):
    model_path = os.path.join(project_path(), 'models') 
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    file_name = type(model).__name__

    dst = os.path.join(model_path, f"{file_name}_{timestamp}.pkl") # mlops/models/modelclass_T현재시간.pkl

    save_data = {
        "model": model,
        "model_params": model_params,
        "tf_idf": tf_idf,
        "embedding_module": embedding_module,
        "genre2idx": genre2idx,
        "rmse": rmse,
        "timestamp": timestamp
    }

    joblib.dump(save_data, dst)
    print(f"✅ 모델 저장 완료: {dst}")

    if update_checkpoint:
        checkpoint_path = os.path.join(model_path, "checkpoint.pkl")
        best_rmse = None

        if os.path.exists(checkpoint_path):
            prev = joblib.load(checkpoint_path)
            best_rmse = prev.get("rmse", float("inf"))

        current_valid_rmse = rmse.get("valid_rmse", float("inf"))

        if best_rmse is None or current_valid_rmse < best_rmse:
            joblib.dump({"path": model_path, "rmse": current_valid_rmse}, checkpoint_path)
            print(f"🎯 Checkpoint 갱신됨 (Valid RMSE: {current_valid_rmse:.4f})")
        else:
            print(f"ℹ️ Checkpoint 유지됨 (기존 Valid RMSE: {best_rmse:.4f})")

    return dst


def train_and_log_model(model_name, **kwargs):
    init_mlflow(experiment_name = "movie_rating_final")

    if isinstance(model_name, str):
        model_type = ModelType.validation(model_name)
    elif isinstance(model_name, ModelType):
        model_type = model_name
    else:
        raise TypeError("model_name은 str 또는 ModelType Enum 이어야 합니다.")

    model_class = {
        ModelType.RANDOMFOREST: RandomForestRegressor,
        ModelType.XGBOOST: XGBRegressor,
        ModelType.LIGHTGBM: LGBMRegressor
    }[model_type]

    # load dataset
    train_dataset, valid_dataset, test_dataest = get_datasets()

    valid_keys = model_class().get_params().keys()

    # 잘못 기입한 키 탐색
    invalid_keys = [k for k in kwargs if k not in valid_keys]
    if invalid_keys:
        raise ValueError(f"❌ 잘못된 하이퍼파라미터: {invalid_keys}\n"
                        f"✅ 사용 가능한 파라미터: {list(valid_keys)}")


    user_params = {k: v for k, v in kwargs.items() if k in valid_keys}

    model = model_class(**user_params, random_state = 42)
    custom_params = filter_custom_params(model, user_params)

    # model training and predict
    X_train, y_train = train_dataset.features, train_dataset.target
    X_val, y_val = valid_dataset.features, valid_dataset.target
    X_test, y_test = test_dataest.features, test_dataest.target

    KST = timezone(timedelta(hours=9))
    timestamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_name}_training_{timestamp}"

    with mlflow.start_run(run_name=run_name) as run:
        model.fit(X_train, y_train)

        train_preds = evaluate(model, X_train)
        valid_preds = evaluate(model, X_val)
        test_preds = evaluate(model, X_test)

        train_rmse = mean_squared_error(y_train, train_preds, squared=False)
        valid_rmse = mean_squared_error(y_val, valid_preds, squared=False)
        test_rmse = mean_squared_error(y_test, test_preds, squared= False)

        print(f"✅ [{model_type.value.upper()}] Train RMSE: {train_rmse:.4f}")
        print(f"✅ [{model_type.value.upper()}] Valid RMSE: {valid_rmse:.4f}")
        print(f"✅ [{model_type.value.upper()}] Test  RMSE: {test_rmse:.4f}")

        all_params = model.get_params()

        rmse_metrics = {
                "train_rmse": train_rmse,
                "valid_rmse": valid_rmse,
                "test_rmse": test_rmse
            }

        dst = model_save(
            model = model,
            all_params = all_params,
            model_params = custom_params,
            tf_idf = train_dataset.tf_idf,
            embedding_module=train_dataset.embedding_module,
            genre2idx=train_dataset.genre2idx,
            timestamp = timestamp,
            rmse = {
                "train_rmse": train_rmse,
                "valid_rmse": valid_rmse,
                "test_rmse": test_rmse
            }
        )

        # 6. MLflow 메트릭 & 파라미터 로깅
        mlflow.log_params(custom_params)
        mlflow.log_metric("rmse", valid_rmse)
        mlflow.set_tag("model_timestamp", timestamp)

        # 7. 모델 저장
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            name=type(model).__name__,
            input_example=X_train.iloc[:5],
            signature=signature
        )

        # 8-1. mlflow artifact 에 저장
        mlflow.log_artifact(dst)

        logger.info(f"[{run_name}][{model_type.value.upper()}] RMSE: {valid_rmse:.4f}")




