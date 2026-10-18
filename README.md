# Spotify Music Popularity Predictor 🎵

A machine learning project that predicts an estimated Spotify popularity score (0–100) using 13 audio features. The project uses a **Random Forest Regressor** and a **Streamlit web application** to make predictions through a simple, interactive interface.

## Features

- Predicts estimated song popularity from audio features.
- Interactive web interface built with Streamlit.
- Pre-trained Random Forest model included.
- Exploratory data analysis using a Jupyter Notebook.
- Option to retrain the model using the included dataset.

## Project Structure

```text
spotify_music_popularity_predictor/
├── app.py
├── data/
│   └── spotify_data.csv
├── models/
│   └── music_popularity_model.pkl
├── notebooks/
│   └── 01_eda.ipynb
├── scripts/
│   └── create_model.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Machine Learning Model

**Algorithm:** Random Forest Regressor

**Target variable:** Spotify popularity score

**Input:** 13 audio features

The model uses the following features:

- Duration
- Explicit content
- Danceability
- Energy
- Musical key
- Loudness
- Mode
- Speechiness
- Acousticness
- Instrumentalness
- Liveness
- Valence
- Tempo

The model returns an estimated popularity score on a 0–100 scale. Predictions are estimates and are not guarantees of a song's actual or future popularity.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Jupyter Notebook
- Matplotlib and Seaborn

## Installation and Setup

### Prerequisites

- Python installed on your computer
- Git (optional, for cloning the repository)

### Run on Windows

Open PowerShell or the VS Code terminal in the project directory.

1. Create a virtual environment:

   ```powershell
   py -m venv .venv
   ```

2. Activate the environment:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

3. Install the dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Start the application:

   ```powershell
   python -m streamlit run app.py
   ```

Streamlit will provide a local URL in the terminal. Open that URL in your browser to use the application.

If PowerShell blocks environment activation, run the app directly using the virtual environment's Python executable:

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

## Retraining the Model (Optional)

The included pre-trained model can be used without retraining.

To train and save a new model using the included dataset, run:

```powershell
python scripts/create_model.py
```

The script saves the trained model to:

```text
models/music_popularity_model.pkl
```

## Limitations

- The dataset is relatively small, so predictions may not generalize well to all songs.
- Predicted popularity may differ from the actual Spotify popularity score.
- Model performance should be evaluated using suitable regression metrics before making claims about accuracy.

## Purpose

This project was developed as a student machine learning project to explore data analysis, regression modelling, model integration, and deployment through Streamlit.

## Developer

**Rakesh Kumar**
