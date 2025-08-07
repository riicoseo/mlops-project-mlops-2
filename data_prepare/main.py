import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import pandas as pd
from dotenv import load_dotenv

from src.utils.utils import project_path
from data_prepare.crawler import TMDBCrawler

load_dotenv()

def run_popular_movie_crawler():
    tmdb_crawler = TMDBCrawler()
    movie = tmdb_crawler.get_bulk_popular_movies(start_page=1, end_page=500)
    genre_name_to_id = tmdb_crawler.get_genre_name_to_id()
    # tmdb_crawler.save_movies_to_json_file(movie, genre_name_to_id, "./result", 'popular')
    popular_json_path = os.path.join(project_path(), "data_prepare", "result")  
    tmdb_crawler.save_movies_to_json_file(movie, genre_name_to_id, popular_json_path, 'popular')

if __name__ == "__main__":
    run_popular_movie_crawler()
    print()