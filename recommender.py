# Import Libraries
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ---------------------------
# Load datasets
# ---------------------------
ratings_df = pd.read_csv('datasets/ratings.csv').drop('timestamp', axis=1)
movies_df = pd.read_csv('datasets/movies.csv').drop_duplicates(subset=['title']).dropna()

# ---------------------------
# Preprocess movies_df
# ---------------------------
def extract_year(df):
    '''Extract Year from movie titles'''
    df['year'] = df['title'].str.extract(r'\((\d{4})\)')

extract_year(movies_df)
movies_df = movies_df.dropna()
movies_df['year'] = movies_df['year'].astype(int) # Convert the year df to int types

def normalize_title(title):
    '''Remove all items in the title thats not a letter or a space'''
    return re.sub('[^a-zA-Z\s]','',title).lower()

movies_df['clean_title'] = movies_df['title'].apply(normalize_title)

# Drop original title
movies_df = movies_df.drop('title', axis=1)

# Split genres into lists
movies_df['genres'] = movies_df['genres'].apply(lambda x: x.split('|'))

# ---------------------------
# Average ratings for weighted score
# ---------------------------
avg_ratings = ratings_df.groupby('movieId').agg(
    number_rating=('rating','count'),
    average_rating=('rating','mean')
).reset_index()


# ---------------------------
# TF-IDF vector for search
# ---------------------------
vectorizer = TfidfVectorizer(ngram_range=(1,2))
tf_idf = vectorizer.fit_transform(movies_df['clean_title'])

def search_titles(df, title):
    """Search for a movie and return top 5 closest matches."""
    title_norm = normalize_title(title)
    query_vector = vectorizer.transform([title_norm])
    similarity = cosine_similarity(query_vector, tf_idf).flatten()
    top5_idxs = np.argsort(similarity)[-5:][::-1]
    results_df = df.iloc[top5_idxs].copy()
    results_df['similarity_score'] = similarity[top5_idxs]
    most_similar_idx = int(results_df['movieId'].values[0])
    return results_df, most_similar_idx


# ---------------------------
# Genre-based recommendations
# ---------------------------
mlb = MultiLabelBinarizer()
genre_vectors = mlb.fit_transform(movies_df['genres'])
genre_df = pd.DataFrame(genre_vectors, columns=mlb.classes_)

def recommend_movie_genre(movies_df, most_similar_idx, n=10):
    """
    Return a DataFrame of recommended movies by genre.
    """
    # Use the genre one-hot vectors
    X = mlb.transform(movies_df['genres'])
    
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(X)
    
    query_idx = movies_df.index[movies_df['movieId']==most_similar_idx][0]
    distances, idxs = knn.kneighbors([X[query_idx]], n_neighbors=n+1)
    
    # Skip itself
    neighbor_idxs = idxs[0][1:]
    
    rec_df = movies_df.iloc[neighbor_idxs][['movieId','clean_title','genres']]
    return rec_df


# ---------------------------
# User-based recommendations
# ---------------------------
def get_weighted_avg_rating(avg_ratings, movie_id):
    '''Get weighted average for rating to determine similar users'''
    C = avg_ratings['average_rating'].mean()
    m = 10
    movie_row = avg_ratings.loc[avg_ratings['movieId']==movie_id].iloc[0]
    v = movie_row['number_rating']
    R = movie_row['average_rating']
    return (v*R + m*C)/(v + m)



def get_similar_movies(ratings_df, most_similar_idx, avg_ratings, thresh=0.001, n=10):
    """
    Return user-based recommended movies as a DataFrame with movieId and clean_title.
    """
    weighted_avg = get_weighted_avg_rating(avg_ratings, most_similar_idx)
    similar_users = ratings_df[
        (ratings_df['movieId']==most_similar_idx) & (ratings_df['rating'] >= weighted_avg)
    ]['userId'].unique()
    
    user_recs = ratings_df[
        (ratings_df['userId'].isin(similar_users)) & (ratings_df['rating'] > weighted_avg)
    ]['movieId']
    user_recs = user_recs.value_counts(normalize=True)
    user_recs = user_recs[user_recs > thresh]
    
    # For comparison with all users
    all_users = ratings_df[
        (ratings_df['movieId'].isin(user_recs.index)) & (ratings_df['rating'] >= weighted_avg)
    ]
    all_users_recs = all_users['movieId'].value_counts()/len(all_users['userId'].unique())
    
    difference = pd.concat([user_recs, all_users_recs], axis=1, join='inner')
    difference.columns = ['similar','all']
    difference['score'] = difference['similar']/difference['all']
    top_ids = difference.sort_values('score', ascending=False).head(n).index.tolist()
    
    # Return DataFrame with movieId and clean_title
    rec_df = movies_df[movies_df['movieId'].isin(top_ids)][['movieId','clean_title']]
    return rec_df
