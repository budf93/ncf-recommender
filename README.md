# Movie Recommender System with Neural Collaborative Filtering

This project is a simple movie recommender system that uses a Neural Collaborative Filtering (NCF) model to provide personalized movie recommendations. The system is built with **PyTorch** for deep learning and **Flask** for serving the recommendations through a web API. It's designed for anyone interested in a practical example of a machine learning project, from model development to deployment.

-----

## 🚀 Getting Started

### Prerequisites

You'll need a working Python environment (3.7+) with `pip`.

### 📦 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your_username/ncf-recommender-project.git
    cd ncf-recommender-project
    ```
2.  **Set up the virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use: venv\Scripts\activate
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The `requirements.txt` file is automatically generated from the libraries you installed.*
4.  **Download the dataset**:
    Download the **MovieLens 100k dataset** from [this link](https://www.google.com/search?q=https://files.grouplens.org/datasets/movielens/ml-100k.zip). Extract the zip file and place the `u.data` file into the `data/ml-100k/` directory.

-----

## ⚙️ How to Use

### Step 1: Train the Model

The training script processes the raw data, builds the NCF model, and trains it. This process creates a file named `ncf_recommender.pth` in the `models/` directory, which contains the model's trained weights.

From the root directory, run the training script:

```bash
python src/train.py
```

This might take a few minutes to complete depending on your hardware.

### Step 2: Run the API

Once the model is trained, you can start the Flask API. The API loads the saved model and provides a simple endpoint for getting recommendations.

From the root directory, run the API script:

```bash
python src/api.py
```

The API will be available at `http://127.0.0.1:5000`.

-----

## 🖥️ API Endpoint

The API has one main endpoint for retrieving movie recommendations.

### `GET /recommend/<user_id>`

  - **Description**: Returns a list of the top 10 recommended movie IDs for a given user.
  - **URL**: `http://127.0.0.1:5000/recommend/<user_id>`
  - **Parameters**:
      - `<user_id>`: An integer representing the user's ID (e.g., `1`, `2`, etc.).

#### **Example Request**

To get recommendations for **User ID 1**, navigate to the following URL in your browser:

`http://127.0.0.1:5000/recommend/1`

#### **Example Response**

```json
{
  "user_id": 1,
  "recommendations": [
    346,
    286,
    269,
    182,
    306,
    318,
    273,
    222,
    312,
    294
  ]
}
```

-----

## 📂 Project Structure

  - `data/`: Contains the raw `u.data` file.
  - `models/`: Stores the trained `ncf_recommender.pth` model file.
  - `src/`: Holds all the Python source code.
      - `train.py`: The script for data processing, model training, and saving.
      - `api.py`: The Flask application for serving recommendations.
      - `dataset.py`: Defines the PyTorch `Dataset` for data handling.
      - `model.py`: Defines the NCF model architecture.
  - `venv/`: The Python virtual environment.
  - `README.md`: This documentation file.

-----

## ✨ Technologies Used

  - **Python**: The main programming language.
  - **PyTorch**: For building and training the deep learning model.
  - **Pandas**: For data handling and preprocessing.
  - **Flask**: A micro web framework for creating the API.