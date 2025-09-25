# Movie Recommender System with Neural Collaborative Filtering

This project is a simple movie recommender system built using a Neural Collaborative Filtering (NCF) model with **PyTorch** and served via a **Flask** API. It demonstrates the complete lifecycle of a machine learning project, from data exploration and model development to API deployment.

<br>

-----

## Project Structure

The project is organized into a clean and logical directory structure for clarity and maintainability.

```
ncf_recommender_project/
├── data/
│   └── ml-100k/
│       └── u.data
├── models/
│   └── ncf_recommender.pth
├── src/
│   ├── api.py
│   ├── dataset.py
│   ├── model.py
│   └── train.py
├── venv/
└── README.md
```

  - **`data/`**: Contains the raw MovieLens 100k dataset.
  - **`models/`**: Stores the trained PyTorch model file (`.pth`).
  - **`src/`**: Houses all the Python source code for the project.
  - **`venv/`**: The Python virtual environment for managing dependencies.
  - **`README.md`**: This project documentation file.

<br>

-----

## Setup and Installation

### Prerequisites

  - Python 3.7+
  - `pip` (Python package installer)

### Installation Steps

1.  **Clone the Repository**: If this project were in a repository, you would clone it. Since we are building from scratch, you'll simply create the directory as per the tutorial.

2.  **Create and Activate a Virtual Environment**: It is highly recommended to use a virtual environment to manage project dependencies.

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use: venv\Scripts\activate
    ```

3.  **Install Dependencies**: Install the required libraries using `pip`.

    ```bash
    pip install pandas numpy torch flask
    ```

4.  **Download the Dataset**: Download the **MovieLens 100k dataset** and place the `u.data` file inside the `data/ml-100k/` directory.

<br>

-----

## How to Run the Project

Follow these steps to train the model and start the API.

### 1\. Train the NCF Model

Navigate to the `src` directory and run the `train.py` script. This script loads the data, trains the NCF model, and saves the trained model to the `models/` folder.

```bash
cd src
python train.py
```

Upon successful training, you'll see a confirmation message, and a `ncf_recommender.pth` file will be created in the `models/` directory.

### 2\. Start the Recommendation API

In the same `src` directory, run the `api.py` script to launch the Flask server. This API loads the trained model and is ready to serve recommendations.

```bash
python api.py
```

The API will start running locally, typically on `http://127.0.0.1:5000`.

<br>

-----

## API Endpoints

The API has a single endpoint for getting movie recommendations for a specific user.

### `GET /recommend/<user_id>`

  - **Description**: Retrieves a list of the top 10 recommended movie IDs for a given user.
  - **URL**: `http://127.0.0.1:5000/recommend/<user_id>`
  - **Parameters**:
      - `<user_id>`: The unique ID of the user (e.g., `1`, `2`, `3`).

### Example

To get recommendations for user `1`, open your web browser or use a tool like `curl` to access the following URL:

`http://127.0.0.1:5000/recommend/1`

**Example Response:**

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

<br>

-----

## Core Technologies

  - **Python**: The main programming language.
  - **PyTorch**: A powerful deep learning framework used for building and training the NCF model.
  - **Pandas**: A data analysis and manipulation library used for handling the dataset.
  - **Flask**: A micro web framework for creating the recommendation API.

<br>

-----