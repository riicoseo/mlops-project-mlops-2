import pandas as pd
from dotenv import load_dotenv

# from preprocessing import TMDBPreProcessor
from crawler import TMDBCrawler

load_dotenv()

def run_popular_movie_crawler():
    tmdb_crawler = TMDBCrawler()
    result = tmdb_crawler.get_bulk_popular_movies(start_page=1, end_page=500)
    tmdb_crawler.save_movies_to_json_file(result, "./result", 'popular')

if __name__ == "__main__":
    run_popular_movie_crawler()