from tmdbv3api import TMDb, Movie, Genre, Discover
import pandas as pd
import os

from config import API_KEY, RAW_DATA_PATH
from utils.utils import as_dict

def collect_popular_movies(pages=5):
    tmdb = TMDb()
    tmdb.api_key = API_KEY

    movie = Movie()
    genre = Genre()

    genre_map = {g.id : g.name for g in genre.movie_list()}

    all_movies = []
    for page in range(1, pages+1):
        results = movie.popular(page=page)
        for m in results:
            credits = movie.credits(m.id)
            crew_list = credits['crew']["_json"]
            cast_list = credits['cast']["_json"]
            directors = [c['name'] for c in crew_list if c['job'] == 'Director']
            cast = [c['name'] for c in cast_list[:5]]  # 상위 5명 배우

            all_movies.append({
                "id": m.id,
                "title": m.title,
                "original_title": m.original_title,
                "overview": m.overview,
                "genre_ids": m.genre_ids,
                "genres": [genre_map[gid] for gid in m.genre_ids if gid in genre_map],
                "original_language": m.original_language,
                "popularity": m.popularity,
                "vote_average": m.vote_average,
                "vote_count": m.vote_count,
                "release_date": m.release_date,
                "directors": directors,
                "cast": cast
            })

    df = pd.DataFrame(all_movies)
    os.makedirs(RAW_DATA_PATH, exist_ok=True)
    output_path = os.path.join(RAW_DATA_PATH, "popular_movies.csv")
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Saved {len(df)} movies to popular_movies.csv")




def collect_movies_with_discover(pages=5, start_year=2000, end_year=2025):
    tmdb = TMDb()
    tmdb.api_key = API_KEY

    discover = Discover()
    genre = Genre()
    movie = Movie()

    genre_map = {g.id: g.name for g in genre.movie_list()}

    all_movies = []

    for year in range(start_year, end_year + 1):
        print(f"=== {year}년 영화 수집 시작 ===")
        for page in range(1, pages + 1):
            results = discover.discover_movies({
                "primary_release_year": year,
                "sort_by": "popularity.desc",  # 인기순 정렬
                "page": page
            })

            for m in results:
                # 크레딧 정보 가져오기
                credits = movie.credits(m.id)

                # crew / cast 처리
                try:
                    crew_list = credits['crew']["_json"]
                    cast_list = credits['cast']["_json"]
                except KeyError:
                    crew_list = credits.get('crew', [])
                    cast_list = credits.get('cast', [])

                if not isinstance(crew_list, list):
                    crew_list = []

                if not isinstance(cast_list, list):
                    cast_list = []        

                try:
                    directors = [c['name'] for c in crew_list if c.get('job') == 'Director']
                    cast = [c['name'] for c in cast_list[:5]]  # 상위 5명 배우
                except Exception:
                    continue 

                all_movies.append({
                    "id": m.id,
                    "title": m.title,
                    "original_title": m.original_title,
                    "overview": m.overview,
                    "genre_ids": m.genre_ids,
                    "genres": [genre_map[gid] for gid in m.genre_ids if gid in genre_map],
                    "original_language": m.original_language,
                    "popularity": m.popularity,
                    "vote_average": m.vote_average,
                    "vote_count": m.vote_count,
                    "release_date": m.release_date,
                    "directors": directors,
                    "cast": cast
                })

    # DataFrame 저장
    df = pd.DataFrame(all_movies)
    os.makedirs(RAW_DATA_PATH, exist_ok=True)
    output_path = os.path.join(RAW_DATA_PATH, "discover_movies.csv")
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Saved {len(df)} movies to discover_movies.csv")

def collect_movies_daily(pages=5, start_year=2000, end_year=2025):
    tmdb = TMDb()
    tmdb.api_key = API_KEY

    discover = Discover()
    genre = Genre()
    movie = Movie()

    genre_map = {g.id: g.name for g in genre.movie_list()}

    all_movies = []

    for year in range(start_year, end_year + 1):
        print(f"=== {year}년 영화 수집 시작 ===")
        for page in range(1, pages + 1):
            results = discover.discover_movies({
                "primary_release_year": year,
                "sort_by": "popularity.desc",  # 인기순 정렬
                "page": page
            })

            for m in results:
                # 크레딧 정보 가져오기
                credits = movie.credits(m.id)

                # crew / cast 처리
                try:
                    crew_list = credits['crew']["_json"]
                    cast_list = credits['cast']["_json"]
                except KeyError:
                    crew_list = credits.get('crew', [])
                    cast_list = credits.get('cast', [])

                if not isinstance(crew_list, list):
                    crew_list = []

                if not isinstance(cast_list, list):
                    cast_list = []        

                try:
                    directors = [c['name'] for c in crew_list if c.get('job') == 'Director']
                    cast = [c['name'] for c in cast_list[:5]]  # 상위 5명 배우
                except Exception:
                    continue 

                all_movies.append({
                    "id": m.id,
                    "title": m.title,
                    "original_title": m.original_title,
                    "overview": m.overview,
                    "genre_ids": m.genre_ids,
                    "genres": [genre_map[gid] for gid in m.genre_ids if gid in genre_map],
                    "original_language": m.original_language,
                    "popularity": m.popularity,
                    "vote_average": m.vote_average,
                    "vote_count": m.vote_count,
                    "release_date": m.release_date,
                    "directors": directors,
                    "cast": cast
                })

    # DataFrame 저장
    df = pd.DataFrame(all_movies)
    os.makedirs(RAW_DATA_PATH, exist_ok=True)
    output_path = os.path.join(RAW_DATA_PATH, "discover_movies_test_set.csv")

    if os.path.exists(output_path):
        old_df = pd.read_csv(output_path)
        combined_df = pd.concat([old_df, df], ignore_index=True)

        # 중복 제거 (id 기준)
        combined_df.drop_duplicates(subset=["id"], keep="last", inplace=True)
    else:
        combined_df = df



    combined_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Saved {len(df)} new movies, total {len(combined_df)} movies to discover_movies_test_set.csv")


if __name__ == "__main__":
    # collect_popular_movies(pages=2)
    # collect_movies_with_discover(pages=50, start_year=1995, end_year=2025)

    # for test set
    # collect_movies_with_discover(pages=50, start_year=1993, end_year=1994)

    # for daily data set
    collect_movies_daily(pages=50, start_year=1993, end_year=1994)
