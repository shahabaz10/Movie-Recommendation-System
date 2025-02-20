import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load datasets
movies_df = pd.read_csv("data/movies.csv")

# Load Collaborative Filtering Model & Data
with open("models/collab_model.pkl", "rb") as file:
    collab_model = pickle.load(file)

with open("models/movie_user_matrix.pkl", "rb") as file:
    movie_user_df = pickle.load(file)

# Load Content-Based Filtering Model & Data
with open("models/content_model.pkl", "rb") as file:
    content_model = pickle.load(file)

with open("models/columns.pkl", "rb") as file:
    genre_columns = pickle.load(file)

# Streamlit UI Setup
st.title("🎬 Movie Recommendation System")
st.write("Choose a recommendation method and enter a movie to get suggestions!")

# User selects the recommendation method
option = st.radio("Select Recommendation Method:", ["Collaborative Filtering", "Content-Based Filtering"])
movie_name = st.selectbox("Choose a Movie:", movies_df["title"].values)
num_recommendations = st.slider("Number of Recommendations:", min_value=1, max_value=10, value=5)

# Collaborative Filtering Recommendation Function
def recommend_collaborative(movie_name, num_recommendations):
    try:
        movieId = movies_df.loc[movies_df["title"] == movie_name, "movieId"].values[0]
        distances, neighbors = collab_model.kneighbors([movie_user_df.loc[movieId]], n_neighbors=num_recommendations+1)
        recommended_movies = [movies_df.loc[movies_df["movieId"] == movie_user_df.iloc[i].name, "title"].values[0] for i in neighbors[0] if i != movieId]
        return recommended_movies
    except:
        return ["Movie not found in user-rating data."]


# Content-Based Filtering Recommendation Function
def recommend_content(movie_name, num_recommendations):
    # Load the genre feature matrix
    movie_genres_df = pd.read_pickle("models/movie_genres_df.pkl")
    try:
        # Ensure the movie exists in the dataset
        if movie_name not in movies_df["title"].values:
            return ["Movie not found in dataset."]
        
        # Find the movie index
        movie_index = movies_df[movies_df["title"] == movie_name].index[0]
        
        # Check if the index exists in movie_genres_df
        if movie_index >= len(movie_genres_df):
            return ["Movie not found in genre data."]
        
        # Find similar movies using the content-based model
        distances, neighbors = content_model.kneighbors([movie_genres_df.iloc[movie_index]], n_neighbors=num_recommendations+1)
        
        recommended_movies = [movies_df.iloc[i]["title"] for i in neighbors[0] if i != movie_index]
        
        return recommended_movies
    
    except Exception as e:
        return [f"Error: {str(e)}"]


# Run the Selected Recommendation Method
if st.button("Get Recommendations"):
    if option == "Collaborative Filtering":
        recommendations = recommend_collaborative(movie_name, num_recommendations)
    else:
        recommendations = recommend_content(movie_name, num_recommendations)

    st.write("### Recommended Movies:")
    for movie in recommendations:
        st.write(f"- {movie}")
