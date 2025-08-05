import joblib
import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch


from src.utils.utils import project_path
from src.dataset import movie_rating



class MovieRatingModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
        self.tf_idf = None
        self.embedding_module = None
        self.genre2idx = None
        self.genre_decode = movie_rating.get_genre_decode()

    def load_context(self, context):
        bundle = joblib.load(context.artifacts["artifacts_bundle"])
        self.tf_idf = bundle["tf_idf"]
        self.embedding_module = bundle["embedding_module"]
        self.genre2idx = bundle["genre2idx"]

    def predict(self, context, model_input):
        # 1. overview 전처리
        overview_clean = movie_rating.clean_korean_text(model_input["overview"])
        overview_vec = self.tf_idf.transform([overview_clean])

        # 2. genres
        genre_names = model_input["genres"] 
        genre_ids = [self.genre_decode.get(name,0) for name in genre_names]
        genre_idx_list = [self.genre2idx.get(gid, 0) for gid in genre_ids]
        genre_tensor = self.embedding_module([genre_idx_list])
        genre_vec = genre_tensor.cpu().detach().numpy()        
        
        # 3. 기타 변수들 전처리
        meta_features = np.array([[model_input["adult"], model_input["video"], model_input["is_english"]]])

        X = np.hstack([meta_features, overview_vec.toarray(), genre_vec]) 

        return self.model.predict(X).clip(0,10)
