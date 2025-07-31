import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("TMDB_API_KEY")
RAW_DATA_PATH = os.getenv("RAW_DATA_PATH", "data/raw")

if not API_KEY:
    raise ValueError("TMDB_API_KEY is not set in the environment variables.")