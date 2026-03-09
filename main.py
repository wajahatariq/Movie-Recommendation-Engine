import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import requests

# 1. Page Configuration & Professional Styling
st.set_page_config(page_title="Recommendation Engine", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .stApp { background-color: #F4F6F9; }
    h1, h2, h3, p, span, label { color: #1A365D !important; font-family: 'Segoe UI', Tahoma, sans-serif; }
    [data-testid="stSidebar"] { background-color: #E2E8F0 !important; border-right: 1px solid #CBD5E0; }
    [data-testid="stSidebar"] .stSelectbox label, [data-testid="stSidebar"] .stSlider label {
        color: #1A365D !important; font-weight: 700 !important;
    }
    .stButton>button { background-color: #1A365D; color: white !important; border-radius: 6px; padding: 10px 24px; font-weight: bold; width: 100%; }
    .stButton>button:hover { background-color: #2B6CB0; color: white !important; }
    .movie-card { background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 15px; border-top: 4px solid #1A365D; }
    .movie-title { font-size: 16px; font-weight: bold; color: #1A365D; margin-bottom: 5px; }
    .movie-rating { font-size: 14px; color: #E53E3E; font-weight: bold; }
    .movie-link { font-size: 12px; color: #3182CE; text-decoration: none; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

# 2. Secure API Connection
def fetch_movie_data(tmdb_id):
    try:
        api_key = st.secrets["TMDB_API_KEY"]
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={api_key}&language=en-US"
        response = requests.get(url, timeout=5)
        data = response.json()
        poster_path = data.get('poster_path', '')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
        return "https://via.placeholder.com/500x750?text=No+Poster"
    except Exception:
        return "https://via.placeholder.com/500x750?text=Poster+Unavailable"

# 3. On-The-Fly Model Training (Cached for Speed)
@st.cache_resource(show_spinner="Initializing Machine Learning Models in the Cloud...")
def train_models_on_the_fly():
    # Load raw CSVs directly
    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')
    tags = pd.read_csv('tags.csv')
    links = pd.read_csv('links.csv')

    # Process Tags and Links
    tags['tag'] = tags['tag'].astype(str)
    movie_tags = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
    
    movies = movies.merge(movie_tags, on='movieId', how='left').fillna('')
    movies = movies.merge(links[['movieId', 'tmdbId', 'imdbId']], on='movieId', how='left')

    # Calculate Average Ratings
    avg_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
    avg_ratings.rename(columns={'rating': 'avg_rating'}, inplace=True)
    movies = movies.merge(avg_ratings, on='movieId', how='left').fillna(0)

    # Feature Engineering
    movies['genres'] = movies['genres'].str.replace('|', ' ')
    movies['combined_features'] = movies['genres'] + " " + movies['tag']

    # Filter for Popular Movies to prevent cold-starts
    df = pd.merge(ratings, movies, on='movieId')
    movie_counts = df['title'].value_counts()
    popular_movies = movie_counts[movie_counts >= 10].index
    df_filtered = df[df['title'].isin(popular_movies)]

    # Matrix Construction
    user_item_matrix = df_filtered.pivot_table(index='userId', columns='title', values='rating').fillna(0)
    item_user_matrix = user_item_matrix.T

    # Train Content-Based Model
    movies_unique = df_filtered[['title', 'combined_features', 'tmdbId', 'imdbId', 'avg_rating', 'genres']].drop_duplicates().reset_index(drop=True)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_unique['combined_features'])
    content_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Train Collaborative Model
    knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_model.fit(item_user_matrix.values)
    movie_titles_index = item_user_matrix.index.tolist()

    # Extract Unique Genres
    all_genres = set()
    for genre_string in movies_unique['genres']:
        for g in genre_string.split():
            if g not in ['(no', 'genres', 'listed)']:
                all_genres.add(g)
    genre_list = sorted(list(all_genres))

    return knn_model, item_user_matrix, movie_titles_index, movies_unique, content_similarity, genre_list

# Initialize the engine
knn_model, item_user_matrix, movie_titles_index, movies_unique, content_similarity, genre_list = train_models_on_the_fly()

# 4. User Interface Build
with st.sidebar:
    st.header("Search Filters")
    selected_tag = st.selectbox("Preferred Genre:", ["Any"] + genre_list)
    min_rating = st.slider("Minimum Rating (0-5):", 0.0, 5.0, 3.0, 0.5)
    
    st.markdown("---")
    st.header("Reference Movie")
    selected_movie = st.selectbox("Select Target:", movie_titles_index)
    generate = st.button("Generate Portfolio")

st.title("Movie Intelligence Dashboard")
st.markdown("Mathematical recommendations based on collaborative filtering and metadata analysis.")
st.markdown("---")

if generate:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Content-Based Results")
        try:
            idx = movies_unique[movies_unique['title'] == selected_movie].index[0]
            sim_scores = sorted(list(enumerate(content_similarity[idx])), key=lambda x: x[1], reverse=True)
            
            count = 0
            for i in [x[0] for x in sim_scores[1:]]:
                if count >= 5: break
                
                movie_data = movies_unique.iloc[i]
                if (movie_data['avg_rating'] >= min_rating) and (selected_tag == "Any" or selected_tag in movie_data['genres']):
                    poster_url = fetch_movie_data(movie_data['tmdbId'])
                    imdb_url = f"https://www.imdb.com/title/tt{str(int(movie_data['imdbId'])).zfill(7)}/" if pd.notna(movie_data['imdbId']) else "#"
                    
                    st.markdown(f'''
                    <div class="movie-card">
                        <div style="display: flex;">
                            <img src="{poster_url}" width="70" style="border-radius: 4px; margin-right: 15px;">
                            <div>
                                <div class="movie-title">{movie_data['title']}</div>
                                <div class="movie-rating">User Rating: {movie_data['avg_rating']:.1f}/5.0</div>
                                <a class="movie-link" href="{imdb_url}" target="_blank">View Details</a>
                            </div>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                    count += 1
            if count == 0: st.warning("No matches for these filters.")
        except Exception: st.error("Content processing error.")

    with col2:
        st.subheader("Collaborative Results")
        try:
            movie_vector = item_user_matrix.loc[selected_movie].values.reshape(1, -1)
            distances, indices = knn_model.kneighbors(movie_vector, n_neighbors=40)
            
            count = 0
            for i in range(1, len(distances.flatten())):
                if count >= 5: break
                rec_title = movie_titles_index[indices.flatten()[i]]
                
                try:
                    match_data = movies_unique[movies_unique['title'] == rec_title].iloc[0]
                    if (match_data['avg_rating'] >= min_rating) and (selected_tag == "Any" or selected_tag in match_data['genres']):
                        poster_url = fetch_movie_data(match_data['tmdbId'])
                        imdb_url = f"https://www.imdb.com/title/tt{str(int(match_data['imdbId'])).zfill(7)}/" if pd.notna(match_data['imdbId']) else "#"
                        
                        st.markdown(f'''
                        <div class="movie-card" style="border-top-color: #38A169;">
                            <div style="display: flex;">
                                <img src="{poster_url}" width="70" style="border-radius: 4px; margin-right: 15px;">
                                <div>
                                    <div class="movie-title">{rec_title}</div>
                                    <div class="movie-rating" style="color: #38A169;">User Rating: {match_data['avg_rating']:.1f}/5.0</div>
                                    <a class="movie-link" href="{imdb_url}" target="_blank">View Details</a>
                                </div>
                            </div>
                        </div>
                        ''', unsafe_allow_html=True)
                        count += 1
                except Exception: continue
            if count == 0: st.warning("No matches for these filters.")
        except Exception: st.error("Collaborative processing error.")
