# 🎬 Movie Recommender System

A Python-based movie recommendation system with a Streamlit web app interface. This project allows users to search for movies, get recommendations based on genres and similar users, and view movie posters fetched dynamically from TMDB.

---

## 🔹 Features

1. **Movie Search**
   - Search for a movie by title using a TF-IDF based text similarity.
   - Handles typos and partial matches.

2. **Genre-based Recommendations**
   - Uses K-Nearest Neighbors (KNN) to find movies with similar genres.

3. **User-based Recommendations**
   - Finds movies liked by users who rated the selected movie highly.
   - Uses a weighted average to account for popularity and sparsity.

4. **Poster Display**
   - Fetches movie posters dynamically from [TMDB API](https://www.themoviedb.org/).

5. **Interactive Streamlit App**
   - Users can select the number of recommendations to display.
   - Movie posters are displayed in a clean grid layout.

---

## 🔹 Dataset

- **Movies:** `movies.csv` – contains movie titles, genres, and IDs.
- **Ratings:** `ratings.csv` – contains user ratings for movies.
- Source: [MovieLens Dataset](https://grouplens.org/datasets/movielens/)

---

### How it Works

![Demo](test_video/output.gif)