import pandas as pd
import numpy as np
import os
import ast
import csv
import sys


csv.field_size_limit(sys.maxsize)

# 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "discover_movies_test_set.csv")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "movies_processed_test_set.csv")

def load_data():
    print(f"base dir : {BASE_DIR}")
    df = pd.read_csv(RAW_DATA_PATH, encoding='utf-8-sig',  engine="python")
    print(f"원본 데이터 크기: {df.shape}")
    return df

def basic_eda(df):
    print("\n=== 데이터 정보 ===")
    print(df.info())
    print("\n=== 결측치 현황 ===")
    print(df.isnull().sum())
    print("\n=== 기초 통계 ===")
    print(df.describe())

    if "id" in df.columns:
        dup_count = df.duplicated(subset=["id"]).sum()
        print(f"\n=== ID 기준 중복 행 개수: {dup_count} ===")
        if dup_count > 0:
            print(df[df.duplicated(subset=["id"], keep=False)].sort_values("id"))


import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler

def preprocess(df):
    """영화 데이터 전처리"""
    df = df.copy()

    # 0. ID 기준 중복 제거
    if "id" in df.columns:
        before_rows = len(df)
        df = df.drop_duplicates(subset=["id"], keep="first")
        after_rows = len(df)
        print(f"[INFO] ID 기준 중복 제거: {before_rows - after_rows}건 제거됨")

    # 1. 결측치 처리
    df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')
    df["popularity"] = df["popularity"].fillna(0)

    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce")
    df["vote_average"] = df["vote_average"].fillna(df["vote_average"].mean())
    
    df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce")
    df["vote_count"] = df["vote_count"].fillna(0)

    # 2. 로그 변환
    df['popularity'] = np.log1p(df['popularity'])
    df['vote_count'] = np.log1p(df['vote_count'])

    # 3. 장르 처리 → 모든 장르를 MultiLabel One-Hot Encoding
    def extract_all_genres(genre_str):
        try:
            genres_list = ast.literal_eval(genre_str)
            if isinstance(genres_list, list):
                return genres_list
        except:
            return ["Unknown"]
        return ["Unknown"]

    df["genres_list"] = df["genres"].apply(extract_all_genres)

    mlb = MultiLabelBinarizer()
    genre_ohe = pd.DataFrame(
        mlb.fit_transform(df["genres_list"]),
        columns=[f"genre_{g}" for g in mlb.classes_],
        index=df.index
    )

    df = pd.concat([df, genre_ohe], axis=1)
    df.drop(columns=["genres", "genres_list"], inplace=True)

    # 4. 출시 연도 파생 변수
    df["release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year

    # 5. 감독/배우 정보 → Top N 여부 (빈도 기반)
    def safe_eval_list(x):
        try:
            return ast.literal_eval(x)
        except:
            return []

    df["directors_list"] = df["directors"].apply(safe_eval_list)
    df["cast_list"] = df["cast"].apply(safe_eval_list)

    # 감독 출연 횟수 Top 20
    top_directors = pd.Series([d for sub in df["directors_list"] for d in sub]).value_counts().head(20).index.tolist()
    df["has_top_director"] = df["directors_list"].apply(lambda lst: int(any(d in top_directors for d in lst)))

    # 배우 출연 횟수 Top 50
    top_actors = pd.Series([a for sub in df["cast_list"] for a in sub]).value_counts().head(50).index.tolist()
    df["has_top_actor"] = df["cast_list"].apply(lambda lst: int(any(a in top_actors for a in lst)))

    # 6-1. 영어 여부 (Binary)
    df["is_english"] = (df["original_language"] == "en").astype(int)

    # 6-2. Target Encoding (평균 평점)
    lang_mean_map = df.groupby("original_language")["vote_average"].mean().to_dict()
    df["lang_target_enc"] = df["original_language"].map(lang_mean_map)


    # 6. 불필요한 컬럼 제거
    drop_cols = ["id", "title", "original_title", "overview", "genres", "genre_list",
                 "release_date", "directors", "cast", "genre_ids", "original_language", "directors_list", "cast_list"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")

    return df


def save_processed_data(df):
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"전처리 데이터 저장 완료 → {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    df = load_data()
    basic_eda(df)
    processed_df = preprocess(df)
    # basic_eda(processed_df)
    save_processed_data(processed_df)
