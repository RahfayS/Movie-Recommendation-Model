import streamlit as st
import requests
from dotenv import load_dotenv
import os
from recommender import search_titles, recommend_movie_genre, get_similar_movies, movies_df, ratings_df, avg_ratings

# --- Load TMDB API ---
load_dotenv()
TMDB_API = os.getenv('TMDB_API')
if not TMDB_API:
    st.error("TMDB API key not found. Set the TMDB_API environment variable.")
    st.stop()

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommender App")

# --- Helper to fetch poster and IMDb link ---
def fetch_poster_with_imdb(title, api_key):
    search_url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": api_key, "query": title}
    resp = requests.get(search_url, params=params)
    if resp.status_code != 200:
        return None, None
    results = resp.json().get("results")
    if not results:
        return None, None

    movie = results[0]
    poster_path = movie.get("poster_path")
    poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None

    # Get IMDb link
    imdb_url = None
    tmdb_id = movie.get("id")
    if tmdb_id:
        details_url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
        details_resp = requests.get(details_url, params={"api_key": api_key})
        if details_resp.status_code == 200:
            imdb_id = details_resp.json().get("imdb_id")
            if imdb_id:
                imdb_url = f"https://www.imdb.com/title/{imdb_id}/"

    return poster_url, imdb_url

# --- Helper to filter valid TMDB movies ---
def get_valid_recs(movie_df, api_key):
    valid_titles = []
    for title in movie_df['clean_title']:
        poster_url, _ = fetch_poster_with_imdb(title, api_key)
        if poster_url:
            valid_titles.append(title)
    return valid_titles

# --- Helper to display posters in a grid with clickable IMDb links ---
def display_movie_grid(movie_list, api_key, num_cols=5):
    rows = (len(movie_list) // num_cols) + 1
    for i in range(rows):
        cols = st.columns(num_cols)
        for j, col in enumerate(cols):
            idx = i * num_cols + j
            if idx < len(movie_list):
                movie = movie_list[idx]
                poster_url, imdb_url = fetch_poster_with_imdb(movie, api_key)
                with col:
                    if poster_url and imdb_url:
                        st.markdown(f"[![{movie}]({poster_url})]({imdb_url})")
                    elif poster_url:
                        st.image(poster_url, width=150)
                    st.caption(movie.title())

# --- Sidebar controls ---
num_recs = st.sidebar.slider("Number of recommendations", 1, 10, 5)

# --- User input ---
movie_query = st.text_input("Enter a movie title:")

if movie_query:
    # Search
    results_df, movie_idx = search_titles(movies_df, movie_query)
    with st.expander('Results DataFrame'):
        st.write("🔍 Search results:", results_df[['clean_title','similarity_score']])

    if not results_df.empty:
        # First movie poster with IMDb link
        first_title = results_df['clean_title'].values[0]
        poster_url, imdb_url = fetch_poster_with_imdb(first_title, TMDB_API)
        if poster_url and imdb_url:
            st.markdown(f"[![{first_title}]({poster_url})]({imdb_url})")
        elif poster_url:
            st.image(poster_url, width=200)
        else:
            st.info("Poster not found on TMDB.")

        # Genre-based recommendations
        genre_recs = recommend_movie_genre(movies_df, movie_idx, n=num_recs*2)
        genre_titles = get_valid_recs(genre_recs, TMDB_API)[:num_recs]
        st.subheader("🎭 Genre-based recommendations")
        if genre_titles:
            display_movie_grid(genre_titles, TMDB_API, num_cols=5)
        else:
            st.info("No genre-based recommendations found on TMDB.")

        # User-based recommendations
        user_recs = get_similar_movies(ratings_df, movie_idx, avg_ratings, n=num_recs*2)
        user_titles = get_valid_recs(user_recs, TMDB_API)[:num_recs]
        st.subheader("👥 User-based recommendations")
        if user_titles:
            display_movie_grid(user_titles, TMDB_API, num_cols=5)
        else:
            st.info("No user-based recommendations found on TMDB.")
