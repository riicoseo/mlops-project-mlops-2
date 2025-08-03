import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.getenv("PYTHONPATH")
if not BASE_DIR:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 데이터 경로
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw")
# 모델 저장 경로
MODEL_DIR = os.path.join(BASE_DIR, "src", "models")


API_KEY = os.getenv("TMDB_API_KEY")
if not API_KEY:
    raise ValueError("TMDB_API_KEY is not set in the environment variables.")


MLFLOW_URI = os.getenv("MLFLOW_URI")

if __name__ == "__main__":
    print("BASE_DIR:", BASE_DIR)
    print("RAW_DATA_PATH:", RAW_DATA_PATH)
    print("MODEL_DIR:", MODEL_DIR)