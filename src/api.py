import torch
import pandas as pd
import numpy as np
from flask import Flask, jsonify
from flask_cors import CORS  # Import CORS
from model import NCF
from dataset import load_and_preprocess_data

# Load and preprocess data to get the mappings (user/movie ID to index)
file_path = 'data/ml-100k/u.data'
ratings_df = pd.read_csv(file_path, sep='\t', names=['user id', 'item id', 'rating', 'timestamp'])

user_ids = ratings_df['user id'].unique().tolist()
user2idx = {user_id: idx for idx, user_id in enumerate(user_ids)}

movie_ids = ratings_df['item id'].unique().tolist()
movie2idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
idx2movie = {idx: movie_id for movie_id, idx in movie2idx.items()}

num_users = len(user2idx)
num_movies = len(movie2idx)

# Load the trained NCF model from the 'models' directory
model = NCF(num_users, num_movies, embedding_dim=16)
model.load_state_dict(torch.load("models/ncf_recommender.pth"))
model.eval() # Set the model to evaluation mode

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

@app.route('/recommend/<int:user_id>', methods=['GET'])
def recommend(user_id):
    # Check if the requested user ID exists in our dataset
    if user_id not in user2idx:
        return jsonify({"error": "User not found"}), 404

    user_idx = user2idx[user_id]
    
    # Get all movies the user has not yet rated
    all_movie_indices = np.arange(num_movies)
    rated_movies = ratings_df[ratings_df['user id'] == user_id]['item id'].values
    rated_movie_indices = [movie2idx[mid] for mid in rated_movies if mid in movie2idx]
    unrated_movie_indices = np.setdiff1d(all_movie_indices, rated_movie_indices)

    # Use the model to predict ratings for unrated movies
    with torch.no_grad():
        user_tensor = torch.tensor([user_idx] * len(unrated_movie_indices), dtype=torch.long)
        movie_tensor = torch.tensor(unrated_movie_indices, dtype=torch.long)
        predictions = model(user_tensor, movie_tensor).numpy()

    # Get the top 10 recommended movies based on the highest predicted ratings
    top_n_indices = np.argsort(predictions)[-10:][::-1]
    recommended_movies = [idx2movie[idx] for idx in unrated_movie_indices[top_n_indices]]

    return jsonify({"user_id": user_id, "recommendations": recommended_movies})

if __name__ == '__main__':
    app.run(debug=True)