import os
import sys
import re
from collections import defaultdict
import json
import joblib

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
)

import pandas as pd
import numpy as np
from konlpy.tag import Okt
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from src.utils.utils import project_path, save_artifacts_bundle, load_artifacts_bundle, default_to_unk


class GenreEmbeddingModule(nn.Module):
    def __init__(self, genre_id_set, emb_dim=32):
        super().__init__()

        # 장르 인덱싱 + UNK
        genre_id_set = [str(g) for g in genre_id_set]
        genre2idx = {g: idx + 1 for idx, g in enumerate(sorted(genre_id_set))}  # 1부터 시작
        genre2idx['UNK'] = 0  # 0번은 패딩/UNK 용
        self.genre2idx = defaultdict(default_to_unk, genre2idx)  # default to UNK

        self.embedding = nn.Embedding(num_embeddings=len(genre2idx), embedding_dim=emb_dim, padding_idx=0)

    def get_genre2idx(self):
        return dict(self.genre2idx)

    def forward(self, genre_ids_batch):
        """
        genre_ids_batch: List[List[int]]
        Returns: Tensor [batch_size, emb_dim]
        """
        # 인덱스 매핑
        mapped_ids = [[self.genre2idx[g] for g in row] for row in genre_ids_batch]
        mapped_tensors = [torch.tensor(row, dtype=torch.long) for row in mapped_ids]

        # 패딩 적용
        padded = rnn_utils.pad_sequence(mapped_tensors, batch_first=True)  # [batch, max_len]
        device = self.embedding.weight.device
        padded = padded.to(device)

        # 임베딩
        emb = self.embedding(padded)  # [batch, max_len, emb_dim]

        # 마스크를 이용한 평균
        mask = (padded != 0).unsqueeze(-1)        # [batch, max_len, 1]
        masked = emb * mask                       # [batch, max_len, emb_dim]
        summed = masked.sum(dim=1)                # [batch, emb_dim]
        count = mask.sum(dim=1).clamp(min=1)      # [batch, 1]
        mean_emb = summed / count                 # [batch, emb_dim]

        return mean_emb


class MovieRatingDataset:
    def __init__(self, df, tf_idf = None, embedding_module = None):
        self.df = df
        self.features = None
        self.target = None
        self.tf_idf = tf_idf
        self.embedding_module = embedding_module
        self.okt = Okt()
        self._preprocessing()

    def genre_embedding(self, emb_dim:int = 32):
        genre_set = set(g for row in self.df['genre_ids'] for g in row)

    # ✅ 모델 정의 및 GPU로 이동
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embedding_module = GenreEmbeddingModule(genre_set, emb_dim=emb_dim).to(device)

        return embedding_module

    def tensor_to_df(self, tensor_or_array, prefix, index):
        if isinstance(tensor_or_array, torch.Tensor):
            data = tensor_or_array.cpu().detach().numpy()
        else:
            data = tensor_or_array
        return pd.DataFrame(data, columns=[f"{prefix}_{i}" for i in range(data.shape[1])], index=index)


    # tfidf_df
    @staticmethod
    def clean_korean_text(text):
        text = re.sub(r'[^가-힣\s]', '', str(text))
        return text.strip()
    

    def okt_tokenizer(self, text):
        return self.okt.nouns(text)


    def overview_tf_idf(self, max_features:int = 300):
        vectorizer = TfidfVectorizer(tokenizer=self.okt_tokenizer, max_features=max_features)
        vectorizer.fit(self.df['overview_clean'])
        return vectorizer


    def _preprocessing(self):
        self.df['overview_clean'] = self.df['overview'].fillna("").apply(self.clean_korean_text)

        # genre embedding
        if self.embedding_module:
            genre_vecs = self.embedding_module(self.df['genre_ids'].tolist())
            genre_emb_df = self.tensor_to_df(genre_vecs, "emb", self.df.index)
        else:
            self.embedding_module = self.genre_embedding()
            genre_vecs = self.embedding_module(self.df['genre_ids'].tolist())
            genre_emb_df = self.tensor_to_df(genre_vecs, "emb", self.df.index)
            
        # overview tf-idf
        if self.tf_idf:
            X_tfidf = self.tf_idf.transform(self.df['overview_clean'])
            tfidf_df = self.tensor_to_df(X_tfidf.toarray(), "tfidf", self.df.index)
        else:
            self.tf_idf = self.overview_tf_idf()
            X_tfidf = self.tf_idf.transform(self.df['overview_clean'])
            tfidf_df = self.tensor_to_df(X_tfidf.toarray(), "tfidf", self.df.index)
            
        self.df['adult'] = self.df['adult'].astype('int')
        self.df['video'] = self.df['video'].astype("int")

        drop_features = ["backdrop_path", "id", "genre_ids", "original_title",
                          "title", "vote_count", "poster_path", "release_date",
                          "overview", "popularity",'overview_clean', 'original_language', 'vote_average']
        self.df['is_english'] = (self.df["original_language"] == 'en').astype(int)

        self.target = self.df['vote_average']
        self.features = pd.concat([self.df, tfidf_df, genre_emb_df], axis = 1).drop(columns = drop_features, axis = 1)
        

    @property
    def genre2idx(self):
        if self.embedding_module:
            return self.embedding_module.get_genre2idx()
        return {}
        
    @property
    def features_dim(self):
        return self.features.shape[1]


    def __len__(self):
        return len(self.target)


    def __getitem__(self, idx):
        return self.features.iloc[idx].values, self.target.iloc[idx]


    def __getstate__(self):
        state = self.__dict__.copy()
        del state['okt']
        return state


    def __setstate__(self, state):
        self.__dict__.update(state)
        self.okt = Okt() 

def read_dataset():
    movie_rating_path = os.path.join(project_path(),"data_prepare","result")
    with open(movie_rating_path +"/popular.json","r", encoding= 'utf-8')as f:
        data = json.load(f)
    df = pd.DataFrame(data['movies'])
    return df


def split_dataset(df):
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42)
    return train_df, val_df, test_df


def get_genre_decode():
    movie_rating_path = os.path.join(project_path(),"data_prepare","result")
    with open(movie_rating_path +"/popular.json","r", encoding= 'utf-8')as f:
        data = json.load(f)
    return data['genre_decode']


def get_datasets(path="cache", use_cache=True):
    path = os.path.join(project_path(), 'src','dataset', path)
    os.makedirs(path, exist_ok=True)

    train_cache = os.path.join(path, "train_dataset.pkl")
    val_cache = os.path.join(path, "val_dataset.pkl")
    test_cache = os.path.join(path, "test_dataset.pkl")
    bundle_path = os.path.join(path, "artifacts_bundle.pkl")

    # 캐시 로드
    if use_cache and all(os.path.exists(p) for p in [train_cache, val_cache, test_cache, bundle_path]):
        print("✅ 캐시 및 아티팩트 불러오는 중...")
        tfidf_vectorizer, genre2idx, embedding_module = load_artifacts_bundle(GenreEmbeddingModule, bundle_path)

        train_dataset = joblib.load(train_cache)
        val_dataset = joblib.load(val_cache)
        test_dataset = joblib.load(test_cache)

        # 아티팩트 연결
        train_dataset.tf_idf = tfidf_vectorizer
        train_dataset.embedding_module = embedding_module
        val_dataset.tf_idf = tfidf_vectorizer
        val_dataset.embedding_module = embedding_module
        test_dataset.tf_idf = tfidf_vectorizer
        test_dataset.embedding_module = embedding_module

        return train_dataset, val_dataset, test_dataset

    # 전처리 수행
    print("🚀 캐시 없음 → 전처리 실행 중...")
    df = read_dataset()
    train_df, val_df, test_df = split_dataset(df)

    train_dataset = MovieRatingDataset(train_df)
    val_dataset = MovieRatingDataset(val_df, tf_idf=train_dataset.tf_idf, embedding_module=train_dataset.embedding_module)
    test_dataset = MovieRatingDataset(test_df, tf_idf=train_dataset.tf_idf, embedding_module=train_dataset.embedding_module)

    # 캐시 저장
    joblib.dump(train_dataset, train_cache)
    joblib.dump(val_dataset, val_cache)
    joblib.dump(test_dataset, test_cache)

    # 아티팩트 저장
    save_artifacts_bundle(train_dataset.tf_idf, train_dataset.genre2idx, train_dataset.embedding_module.cpu(), path=bundle_path)
    print("💾 전처리 및 아티팩트 캐시 저장 완료!")

    return train_dataset, val_dataset, test_dataset




if __name__ == "__main__":
    print('test 중 입니다.')
    # train, valid, test = get_datasets()
    # print("train set 첫번째 행 : ", train.features.columns)
    # print("valid set 첫번째 행 : ", valid[0])
    # print("test set 첫번째 행 : ", test[0])
    print([i for i, j in get_genre_decode().items()])
