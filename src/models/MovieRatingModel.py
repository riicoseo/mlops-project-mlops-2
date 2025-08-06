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
import torch.nn as nn
import ast

from src.dataset.movie_rating import GenreEmbeddingModule
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
        self.genre2idx = bundle["genre2idx"]
        self.tf_idf = bundle["tfidf_vectorizer"]
        embedding_state = bundle["embedding_state_dict"]

        # 수정: GenreEmbeddingModule 사용
        self.embedding_module = GenreEmbeddingModule(set(self.genre2idx.keys()))
        self.embedding_module.load_state_dict(embedding_state)
        self.embedding_module.eval()

        self.genre_decode = movie_rating.get_genre_decode()


    def predict(self, context, model_input):
        results = []
        model_input['is_english'] = (model_input['original_language'] == 'en').astype(int)
        model_input['overview_clean'] = model_input['overview'].fillna("").apply(movie_rating.MovieRatingDataset.clean_korean_text)
        for _, row in model_input.iterrows():
            # overview 처리
            overview_vec = self.tf_idf.transform([row['overview_clean']])

            # genres 처리
            raw_genres = row["genres"]
            if isinstance(raw_genres, str):
                genre_names = ast.literal_eval(raw_genres)
            else:
                genre_names = raw_genres
            genre_ids = [self.genre_decode.get(name, 0) for name in genre_names]
            genre_idx_list = [self.genre2idx.get(str(gid), 0) for gid in genre_ids]

            with torch.no_grad():
                self.embedding_module.eval()
                genre_tensor = self.embedding_module([genre_idx_list])
                genre_vec = genre_tensor.cpu().numpy().reshape(1, -1)

            meta_features = np.array([[row["adult"], row["video"], row["is_english"]]])
            X = np.hstack([meta_features, overview_vec.toarray(), genre_vec])
            y_pred = self.model.predict(X).clip(0, 10)
            results.append(y_pred[0])

        return np.array(results)
