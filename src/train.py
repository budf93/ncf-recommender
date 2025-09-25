import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import load_and_preprocess_data
from model import NCF

# Load and preprocess data using the function from dataset.py
file_path = 'data/ml-100k/u.data'
dataset, user2idx, movie2idx = load_and_preprocess_data(file_path)

num_users = len(user2idx)
num_movies = len(movie2idx)

# Split data into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

# Initialize model, loss function, and optimizer
model = NCF(num_users, num_movies, embedding_dim=16)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for users, movies, ratings in train_loader:
        optimizer.zero_grad()
        predictions = model(users, movies)
        loss = loss_fn(predictions, ratings)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation step
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for users, movies, ratings in val_loader:
            predictions = model(users, movies)
            loss = loss_fn(predictions, ratings)
            val_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

# Save the trained model to the models folder
torch.save(model.state_dict(), "models/ncf_recommender.pth")
print("Model trained and saved to models/ncf_recommender.pth")