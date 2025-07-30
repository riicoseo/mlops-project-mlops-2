import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import wandb

wandb.init(project="beer-recommendation", job_type="training")

def run_training():
    df = pd.read_csv("data/beer_clean.csv")
    df_pivot = df.pivot_table(index="아이디", columns="맥주", values="평점").fillna(0)
    similarity = cosine_similarity(df_pivot.T)
    beer_sim = pd.DataFrame(similarity, index=df_pivot.columns, columns=df_pivot.columns)
    joblib.dump(beer_sim, "models/beer_similarity.joblib")
    wandb.save("models/beer_similarity.joblib")
