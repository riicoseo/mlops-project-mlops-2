import os
import sys
import datetime
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


import numpy as np
import pandas as pd 
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import fire

from src.dataset.movie_rating import get_datasets, MovieRatingDataset, GenreEmbeddingModule
from src.evaluate.evaluate import evaluate
from src.utils.utils import init_seed, model_dir, project_path
from src.utils.enums import ModelType

import os
import datetime
import joblib


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



def run_train(model_name, **kwargs):
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

    
    model_save(
        model = model,
        all_params = model.get_params(),
        model_params = custom_params,
        tf_idf = train_dataset.tf_idf,
        embedding_module=train_dataset.embedding_module,
        genre2idx=train_dataset.genre2idx,
        rmse = {
            "train_rmse": train_rmse,
            "valid_rmse": valid_rmse,
            "test_rmse": test_rmse
        }
    )
    print(f"✅ [{model_type.value.upper()}] Valid RMSE: {valid_rmse:.4f}")





if __name__ == "__main__":
    # fire.Fire({
    #     "train": run_train,
    # }
    # )
    run_train()