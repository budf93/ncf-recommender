import pandas as pd
import torch
from torch.utils.data import Dataset

class MovieLensDataset(Dataset):
    def __init__(self, users, movies, ratings):
        self.users = users
        self.movies = movies
        self.ratings = ratings

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]

def load_and_preprocess_data(file_path):
    # Define column names for the u.data file
    column_names = ['user id', 'item id', 'rating', 'timestamp']

    # Read the tab-separated u.data file
    ratings_df = pd.read_csv(
        file_path,
        sep='\t',
        names=column_names,
        engine='python'
    )

    # Map user and movie IDs to a new set of continuous indices
    user_ids = ratings_df['user id'].unique().tolist()
    user2idx = {user_id: idx for idx, user_id in enumerate(user_ids)}

    movie_ids = ratings_df['item id'].unique().tolist()
    movie2idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}

    ratings_df['user_idx'] = ratings_df['user id'].map(user2idx)
    ratings_df['movie_idx'] = ratings_df['item id'].map(movie2idx)

    # Convert preprocessed data into PyTorch tensors
    user_tensors = torch.tensor(ratings_df['user_idx'].values, dtype=torch.long)
    movie_tensors = torch.tensor(ratings_df['movie_idx'].values, dtype=torch.long)
    rating_tensors = torch.tensor(ratings_df['rating'].values, dtype=torch.float32)

    # Create a PyTorch Dataset object
    dataset = MovieLensDataset(user_tensors, movie_tensors, rating_tensors)

    return dataset, user2idx, movie2idx