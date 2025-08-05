import pandas as pd
from dotenv import load_dotenv

# from preprocessing import TMDBPreProcessor
from crawler import TMDBCrawler

load_dotenv()

def run_popular_movie_crawler():
    tmdb_crawler = TMDBCrawler()
    movie = tmdb_crawler.get_bulk_popular_movies(start_page=1, end_page=500)
    genre_name_to_id = tmdb_crawler.get_genre_name_to_id()
    tmdb_crawler.save_movies_to_json_file(movie, genre_name_to_id, "./result", 'popular')

if __name__ == "__main__":
    run_popular_movie_crawler()