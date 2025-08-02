# predict.py
import pandas as pd
import numpy as np
import os

import lightgbm as lgb
import joblib

from sklearn.metrics import mean_squared_error

from src.data.preprocess import preprocess

MODEL_PATH = os.path.join("src", "models", "lgb_model.txt")
RAW_TEST_PATH = os.path.join("data","raw", "discover_movies_test_set.csv")
FEATURE_LIST_PATH = os.path.join("src", "models", "feature_list.pkl")

def predict_main():
    print("Loading model...")
    model = lgb.Booster(model_file=MODEL_PATH)
    train_features = joblib.load(FEATURE_LIST_PATH)

    print("Loading test data...")
    df = pd.read_csv(RAW_TEST_PATH)
    df_processed = preprocess(df)

    X_test = df_processed.drop(columns=["vote_average"])  # target 제거
    X_test = X_test[train_features]
    
    print("Predicting...")
    preds = model.predict(X_test, num_iteration=model.best_iteration)

    # rmse test 해보기
    rmse =np.sqrt(mean_squared_error(df['vote_average'], preds))
    print(f"RMSE: {rmse:.4f}")

    result = pd.DataFrame({
        "id": df["id"],
        "predicted_vote_average": preds
    })

    result.to_csv("predictions.csv", index=False)
    print("Predictions saved to predictions.csv")

if __name__ == "__main__":
    predict_main()
