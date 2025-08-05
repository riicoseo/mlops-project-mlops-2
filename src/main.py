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
import joblib

from src.dataset.movie_rating import get_datasets, MovieRatingDataset, GenreEmbeddingModule
from src.evaluate.evaluate import evaluate
from src.utils.utils import init_seed, model_dir, project_path
from src.utils.enums import ModelType
from src.ml import trainer
from src.ml import loader


def run_train(model_name, **kwargs):
    trainer.train_and_log_model(model_name, **kwargs)



if __name__ == "__main__":
    fire.Fire({
        "train": run_train,
    }
    )