import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_and_save_model():
    """
    Trains a RandomForestRegressor model on spotify_data.csv
    and saves it to music_popularity_model.pkl.
    """
    project_dir = Path(__file__).resolve().parents[1]
    data_file = project_dir / "data" / "spotify_data.csv"
    model_dir = project_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_file)

    # Rename / map columns to model feature order
    feature_mapping = {
        "Duration (ms)": "duration_ms",
        "Explicit": "explicit",
        "Danceability": "danceability",
        "Energy": "energy",
        "Key": "key",
        "Loudness": "loudness",
        "Mode": "mode",
        "Speechiness": "speechiness",
        "Acousticness": "acousticness",
        "Instrumentalness": "instrumentalness",
        "Liveness": "liveness",
        "Valence": "valence",
        "Tempo": "tempo"
    }

    feature_columns = [
        "duration_ms", "explicit", "danceability", "energy", "key",
        "loudness", "mode", "speechiness", "acousticness", "instrumentalness",
        "liveness", "valence", "tempo"
    ]

    # Convert explicit column boolean to int
    X = df.rename(columns=feature_mapping)[feature_columns].copy()
    X["explicit"] = X["explicit"].astype(int)

    y = df["Popularity"]

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"Model trained on {len(df)} records!")
    print(f"Metrics -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")

    model_file = model_dir / "music_popularity_model.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(model, f)

    print(f"Saved model to {model_file}")

if __name__ == "__main__":
    train_and_save_model()
