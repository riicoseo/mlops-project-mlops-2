import os
import random
import joblib
import hashlib

import numpy as np


def init_seed(seed:int = 0):
    np.random.seed(seed)
    random.seed(seed)

# /dev/mlops/
def project_path():
    return os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)
        ), "..", ".."
    )

# /dev/mlops/models/{model_name}
def model_dir(model_name):
    return os.path.join(
        project_path(),
        "models",
        model_name
    )



def save_artifacts_bundle(tfidf_vectorizer, genre2idx, embedding_module, path="cache/artifacts_bundle.pkl"):
    os.makedirs(os.path.join(project_path(), "src", "dataset", os.path.dirname(path)), exist_ok = True)

    path = os.path.join(project_path(), "src", "dataset", path)


    artifacts = {
        "genre2idx": genre2idx,
        "tfidf_vectorizer": tfidf_vectorizer,
        "embedding_state_dict": embedding_module.state_dict(),
    }

    joblib.dump(artifacts, path)
    print(f"✅ Artifacts bundled and saved to: {path}")


def load_artifacts_bundle(embedding_module_class, path = "cache/artifacts_bundle.pkl", emb_dim=32):

    path = os.path.join(project_path(), "src", "dataset", path)
    
    artifacts = joblib.load(path)

    tfidf_vectorizer = artifacts["tfidf_vectorizer"]
    genre2idx_raw = artifacts["genre2idx"]

    genre2idx = {str(k): v for k, v in genre2idx_raw.items()}

    emb_module = embedding_module_class(set(genre2idx.keys()), emb_dim=emb_dim)
    emb_module.load_state_dict(artifacts["embedding_state_dict"])
    return tfidf_vectorizer, genre2idx, emb_module

def default_to_unk():
    return 0