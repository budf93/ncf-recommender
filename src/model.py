import torch
import torch.nn as nn

class NCF(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim):
        super(NCF, self).__init__()
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_dim = embedding_dim

        # Generalized Matrix Factorization (GMF) components
        self.user_gmf_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_gmf_embedding = nn.Embedding(num_movies, embedding_dim)

        # Multi-Layer Perceptron (MLP) components
        self.user_mlp_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_mlp_embedding = nn.Embedding(num_movies, embedding_dim)

        self.mlp_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU()
        )

        # Final prediction layer
        self.final_layer = nn.Linear(embedding_dim + embedding_dim // 2, 1)

    def forward(self, user_indices, movie_indices):
        # GMF path
        user_gmf = self.user_gmf_embedding(user_indices)
        movie_gmf = self.movie_gmf_embedding(movie_indices)
        gmf_output = user_gmf * movie_gmf

        # MLP path
        user_mlp = self.user_mlp_embedding(user_indices)
        movie_mlp = self.movie_mlp_embedding(movie_indices)
        mlp_input = torch.cat([user_mlp, movie_mlp], dim=-1)
        mlp_output = self.mlp_layers(mlp_input)

        # Combine GMF and MLP outputs
        combined_output = torch.cat([gmf_output, mlp_output], dim=-1)
        prediction = self.final_layer(combined_output)

        return prediction.squeeze()