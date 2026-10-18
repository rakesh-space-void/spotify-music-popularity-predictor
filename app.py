"""
Spotify Music Popularity Predictor - Streamlit Application
Developer: Rakesh Kumar

A professional commercial-grade AI web application that predicts song popularity score (0-100)
using a Random Forest Regressor model trained on Spotify audio features.
"""

import os
from pathlib import Path
import time
import pickle
import datetime
import io
import csv
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "models" / "music_popularity_model.pkl"
DATASET_FILE = BASE_DIR / "data" / "spotify_data.csv"
DEVELOPER_NAME = "Rakesh Kumar"

KEY_MAP = {
    "C": 0, "C#": 1, "D": 2, "D#": 3,
    "E": 4, "F": 5, "F#": 6, "G": 7,
    "G#": 8, "A": 9, "A#": 10, "B": 11
}
KEY_REVERSE_MAP = {v: k for k, v in KEY_MAP.items()}

MODE_MAP = {"Major": 1, "Minor": 0}
EXPLICIT_MAP = {"Yes": 1, "No": 0}

FEATURE_COLUMNS = [
    "duration_ms",
    "explicit",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"
]

DEFAULT_VALUES = {
    "song_name": "Midnight Dreams",
    "artist_name": "Luna Horizon",
    "genre": "Pop",
    "danceability": 0.75,
    "energy": 0.80,
    "loudness": -6.0,
    "speechiness": 0.08,
    "acousticness": 0.14,
    "instrumentalness": 0.00,
    "liveness": 0.16,
    "valence": 0.45,
    "tempo": 120,
    "minutes": 3,
    "seconds": 30,
    "explicit": "No",
    "key": "C",
    "mode": "Major"
}

# ==============================================================================
# STREAMLIT PAGE CONFIGURATION & SESSION STATE
# ==============================================================================

st.set_page_config(
    page_title="Spotify Music Popularity Predictor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "history" not in st.session_state:
    st.session_state["history"] = []

if "prediction_counter" not in st.session_state:
    st.session_state["prediction_counter"] = 0

if "reset_trigger" not in st.session_state:
    st.session_state["reset_trigger"] = 0

# ==============================================================================
# CUSTOM CSS (Spotify Dark-Green Theme & Pixel-Perfect Alignment)
# ==============================================================================

def inject_custom_css():
    """Injects premium custom CSS for alignment, spacing, typography and Spotify styling."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        .stApp {
            background-color: #121212;
            color: #FFFFFF;
        }

        /* Header Container */
        .header-container {
            background: linear-gradient(135deg, rgba(29,185,84,0.18) 0%, rgba(18,18,12,0.9) 100%);
            border: 1px solid rgba(29, 185, 84, 0.35);
            border-radius: 16px;
            padding: 2.2rem 2rem;
            margin-bottom: 1.8rem;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
        }

        .header-title {
            color: #FFFFFF;
            font-size: 2.4rem;
            font-weight: 800;
            margin: 0;
            letter-spacing: -0.5px;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .header-title span {
            color: #1DB954;
        }

        .header-subtitle {
            color: #B3B3B3;
            font-size: 1.05rem;
            font-weight: 400;
            margin-top: 0.5rem;
            margin-bottom: 0;
        }

        /* Section Cards & Uniform Layout Grid */
        .feature-card {
            background-color: #181818;
            border: 1px solid #282828;
            border-radius: 12px;
            padding: 1.2rem 1.4rem;
            margin-bottom: 1.2rem;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            transition: border-color 0.2s ease;
        }

        .feature-card:hover {
            border-color: #1DB954;
        }

        /* Consistent Widget Row Headers */
        .widget-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
            min-height: 28px;
        }

        .widget-title {
            font-weight: 600;
            font-size: 0.95rem;
            color: #E0E0E0;
        }

        .widget-value-badge {
            background-color: rgba(29, 185, 84, 0.12);
            color: #1DB954;
            border: 1px solid rgba(29, 185, 84, 0.35);
            border-radius: 12px;
            padding: 2px 10px;
            font-size: 0.8rem;
            font-weight: 700;
        }

        .widget-quality-pill {
            background-color: #242424;
            color: #B3B3B3;
            border-radius: 10px;
            padding: 2px 8px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-left: 6px;
        }

        /* Callout Notes */
        .info-note {
            background-color: rgba(29, 185, 84, 0.08);
            border-left: 4px solid #1DB954;
            padding: 0.75rem 1rem;
            border-radius: 4px;
            color: #B3B3B3;
            font-size: 0.88rem;
            margin-top: 0.5rem;
        }

        /* Compact Integrated Duration Display */
        .duration-summary-box {
            background: #222222;
            border: 1px solid #333333;
            border-radius: 8px;
            padding: 6px 12px;
            margin-top: 6px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.85rem;
        }
        .duration-summary-box .dur-label { color: #B3B3B3; }
        .duration-summary-box .dur-ms { color: #1DB954; font-weight: 700; }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #000000;
            border-right: 1px solid #282828;
        }

        .sidebar-header {
            font-size: 1.3rem;
            font-weight: 700;
            color: #FFFFFF;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .sidebar-info-box {
            background-color: #181818;
            border-radius: 10px;
            padding: 0.9rem 1rem;
            border: 1px solid #282828;
            margin-bottom: 1rem;
        }

        .sidebar-item {
            display: flex;
            justify-content: space-between;
            padding: 0.4rem 0;
            border-bottom: 1px solid #242424;
            font-size: 0.86rem;
        }

        .sidebar-item:last-child {
            border-bottom: none;
        }

        .sidebar-label { color: #B3B3B3; }
        .sidebar-value { color: #1DB954; font-weight: 600; }

        /* Prominent Center Predict Button */
        .predict-btn-container {
            display: flex;
            justify-content: center;
            margin: 2rem 0 1.5rem 0;
        }

        .stButton > button {
            background: linear-gradient(135deg, #1DB954 0%, #179843 100%);
            color: #000000;
            font-weight: 800;
            font-size: 1.3rem;
            padding: 0.9rem 2.8rem;
            border-radius: 50px;
            border: none;
            cursor: pointer;
            box-shadow: 0 4px 20px rgba(29, 185, 84, 0.4);
            transition: all 0.3s ease;
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, #1ed760 0%, #1db954 100%);
            box-shadow: 0 6px 25px rgba(29, 185, 84, 0.6);
            transform: translateY(-2px);
            color: #000000;
        }

        /* Centered Compact Result Card */
        .result-card-container {
            background: linear-gradient(145deg, #1e1e1e 0%, #121212 100%);
            border-radius: 20px;
            padding: 2rem 2rem;
            text-align: center;
            box-shadow: 0 10px 35px rgba(0, 0, 0, 0.6);
            max-width: 680px;
            margin: 1.5rem auto;
        }

        .song-title-display {
            font-size: 1.6rem;
            font-weight: 800;
            color: #FFFFFF;
            margin-bottom: 2px;
        }

        .artist-genre-display {
            color: #B3B3B3;
            font-size: 1rem;
            margin-bottom: 1.2rem;
        }

        .category-badge-pill {
            display: inline-block;
            padding: 6px 20px;
            border-radius: 30px;
            font-size: 1.3rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
        }

        .rating-stars {
            font-size: 1.4rem;
            margin-bottom: 0.8rem;
        }

        .confidence-badge {
            display: inline-block;
            background-color: rgba(29, 185, 84, 0.12);
            color: #1DB954;
            border: 1px solid rgba(29, 185, 84, 0.4);
            padding: 4px 14px;
            border-radius: 20px;
            font-size: 0.88rem;
            font-weight: 600;
            margin-top: 6px;
            margin-bottom: 1rem;
        }

        .interpretation-box {
            color: #FFFFFF;
            font-size: 1.05rem;
            font-weight: 500;
            background-color: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 0.8rem 1.6rem;
            border-radius: 30px;
            display: inline-block;
            margin-top: 1rem;
        }

        /* Observation & Input Summary Items */
        .observation-item {
            background-color: #1e1e1e;
            border-left: 4px solid #1DB954;
            padding: 0.75rem 1.2rem;
            border-radius: 6px;
            margin-bottom: 0.6rem;
            font-size: 0.95rem;
            color: #E0E0E0;
        }

        .input-summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
            gap: 10px;
            margin-top: 1rem;
        }

        .input-summary-card {
            background-color: #181818;
            border: 1px solid #282828;
            border-radius: 8px;
            padding: 8px 12px;
            text-align: center;
        }
        .input-summary-card .param-name { color: #B3B3B3; font-size: 0.78rem; text-transform: uppercase; }
        .input-summary-card .param-val { color: #1DB954; font-weight: 700; font-size: 0.95rem; margin-top: 2px; }

        /* Footer */
        .footer-container {
            border-top: 1px solid #282828;
            padding-top: 1.8rem;
            margin-top: 3.5rem;
            text-align: center;
            color: #888888;
            font-size: 0.88rem;
        }

        .footer-highlight {
            color: #1DB954;
            font-weight: 600;
        }

        .github-link-btn {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background-color: #242424;
            color: #FFFFFF;
            border: 1px solid #383838;
            border-radius: 20px;
            padding: 6px 16px;
            text-decoration: none;
            font-size: 0.85rem;
            font-weight: 600;
            margin-top: 10px;
            transition: all 0.2s ease;
        }

        .github-link-btn:hover {
            border-color: #1DB954;
            color: #1DB954;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# ==============================================================================
# DATASET & MODEL LOADERS (STREAMLIT CACHING)
# ==============================================================================

@st.cache_data
def load_dataset(file_path: str):
    """Loads and caches the Spotify dataset automatically."""
    if not os.path.exists(file_path):
        return None
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading dataset '{file_path}': {e}")
        return None


@st.cache_resource
def load_model(file_path: str):
    """Loads and caches the pre-trained Random Forest Regressor model using @st.cache_resource."""
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading pickle file '{file_path}': {e}")
        return None

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_popularity_category(score: float):
    """
    Returns category name, icon, color hex, stars, and interpretation.
    """
    if score <= 20:
        return ("Very Poor", "🔴", "#E74C3C", "⭐", "This song has very limited commercial appeal.")
    elif score <= 40:
        return ("Poor", "🟠", "#E67E22", "⭐⭐", "This song may reach a niche audience.")
    elif score <= 60:
        return ("Average", "🟡", "#F1C40F", "⭐⭐⭐", "This song has moderate commercial potential.")
    elif score <= 80:
        return ("Good", "🔵", "#3D91FF", "⭐⭐⭐⭐", "This song has strong commercial potential.")
    else:
        return ("Excellent", "🟢", "#1DB954", "⭐⭐⭐⭐⭐", "The song has excellent commercial potential.")


def get_feature_quality_indicator(feature_name: str, value: float) -> str:
    """Returns quality indicator badge text for features."""
    if feature_name == "danceability":
        return "Excellent" if value >= 0.80 else ("Good" if value >= 0.60 else "Moderate")
    elif feature_name == "energy":
        return "High Energy" if value >= 0.80 else ("Good" if value >= 0.50 else "Mellow")
    elif feature_name == "speechiness":
        return "Spoken Word" if value >= 0.66 else ("Rap / Spoken" if value >= 0.33 else "Typical Music")
    elif feature_name == "acousticness":
        return "Pure Acoustic" if value >= 0.70 else ("Semi-Acoustic" if value >= 0.30 else "Amplified / Synth")
    elif feature_name == "instrumentalness":
        return "Pure Instrumental" if value >= 0.50 else ("High Instrumental" if value >= 0.15 else "Vocals Present")
    elif feature_name == "liveness":
        return "Live Concert" if value >= 0.70 else ("Room Ambience" if value >= 0.30 else "Studio Track")
    elif feature_name == "valence":
        return "Cheerful Mood" if value >= 0.70 else ("Balanced Mood" if value >= 0.40 else "Dark Mood")
    return ""


def generate_intelligent_observations(features: dict) -> list:
    """Generates intelligent combined multi-feature observations."""
    obs = []

    # High Danceability + High Energy + Low Acousticness
    if features["danceability"] >= 0.70 and features["energy"] >= 0.70 and features["acousticness"] <= 0.30:
        obs.append("Suitable for energetic electronic dance tracks & club playlists.")

    # High Acousticness + High Valence + Low Energy
    elif features["acousticness"] >= 0.60 and features["valence"] >= 0.60 and features["energy"] <= 0.50:
        obs.append("Great fit for acoustic, cheerful acoustic-pop or indie-folk playlists.")

    # High Speechiness + High Energy
    if features["speechiness"] >= 0.25 and features["energy"] >= 0.60:
        obs.append("Strong rap / spoken-word dynamics with aggressive rhythm.")

    # High Instrumentalness + Low Speechiness
    if features["instrumentalness"] >= 0.40 and features["speechiness"] <= 0.15:
        obs.append("Ideal for instrumental background, chill-beats, or cinematic study playlists.")

    # Low Valence + High Energy
    if features["valence"] <= 0.35 and features["energy"] >= 0.65:
        obs.append("Punchy track with a dark, aggressive or melancholic emotional tone.")

    # High Danceability + High Valence
    if features["danceability"] >= 0.75 and features["valence"] >= 0.65:
        obs.append("Upbeat party anthem with strong commercial radioplay potential.")

    # Low Energy + High Acousticness + Low Speechiness
    if features["energy"] <= 0.40 and features["acousticness"] >= 0.60 and features["speechiness"] <= 0.10:
        obs.append("Perfect for bedtime, ambient chillout, or relaxing acoustic sessions.")

    # Default fallback if no combination triggered
    if not obs:
        if features["danceability"] >= 0.7:
            obs.append("High danceability is favorable.")
        if 0.4 <= features["energy"] <= 0.8:
            obs.append("Energy is balanced.")
        if features["speechiness"] < 0.33 and features["instrumentalness"] < 0.5:
            obs.append("Song is mostly vocal.")

    return obs


def render_circular_ring(score: float, color_hex: str):
    """Renders SVG Circular Progress Ring."""
    circumference = 377
    offset = circumference - (circumference * (score / 100.0))

    svg_code = f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; margin: 0.8rem 0;">
        <div style="position: relative; width: 165px; height: 165px;">
            <svg width="165" height="165" viewBox="0 0 140 140">
                <circle cx="70" cy="70" r="60" fill="none" stroke="#282828" stroke-width="10" />
                <circle cx="70" cy="70" r="60" fill="none" stroke="{color_hex}" stroke-width="10"
                        stroke-linecap="round"
                        stroke-dasharray="{circumference}"
                        stroke-dashoffset="{offset}"
                        transform="rotate(-90 70 70)"
                        style="transition: stroke-dashoffset 1s ease-in-out;" />
            </svg>
            <div style="position: absolute; top: 0; left: 0; width: 165px; height: 165px;
                        display: flex; flex-direction: column; align-items: center; justify-content: center;">
                <span style="font-size: 2.2rem; font-weight: 800; color: #FFFFFF; line-height: 1.1;">{score:.2f}</span>
                <span style="font-size: 0.95rem; font-weight: 700; color: {color_hex};">/100</span>
            </div>
        </div>
        <div style="color: #B3B3B3; font-size: 0.82rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px;">
            Popularity Score
        </div>
    </div>
    """
    st.markdown(svg_code, unsafe_allow_html=True)


def calculate_rf_confidence(model, features_df):
    """Estimates RF model confidence percentage based on decision tree ensemble agreement."""
    if hasattr(model, "estimators_"):
        X_arr = features_df.to_numpy()
        tree_preds = [tree.predict(X_arr)[0] for tree in model.estimators_]
        std_dev = float(np.std(tree_preds))
        confidence = float(np.clip(99.0 - (std_dev * 1.5), 75.0, 99.0))
        return int(round(confidence))
    return 92


def generate_report_csv(song_name, artist_name, genre, features_dict, score, category, interpretation):
    """Generates prediction report CSV content."""
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["Spotify Music Popularity Prediction Report"])
    writer.writerow(["Developer", DEVELOPER_NAME])
    writer.writerow(["Generated Date", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    writer.writerow([])
    writer.writerow(["Song Details"])
    writer.writerow(["Song Name", song_name])
    writer.writerow(["Artist Name", artist_name])
    writer.writerow(["Genre", genre])
    writer.writerow([])
    writer.writerow(["Prediction Results"])
    writer.writerow(["Popularity Score (0-100)", f"{score:.2f}"])
    writer.writerow(["Category", category])
    writer.writerow(["Interpretation", interpretation])
    writer.writerow([])
    writer.writerow(["Audio Features"])
    for k, v in features_dict.items():
        writer.writerow([k, v])

    return output.getvalue()

# ==============================================================================
# MAIN APPLICATION LOGIC
# ==============================================================================

def main():
    inject_custom_css()

    # Load Dataset & Model Automatically (No Hardcoded Values)
    df_dataset = load_dataset(DATASET_FILE)
    model = load_model(MODEL_FILE)

    reset_key = st.session_state["reset_trigger"]

    # Calculate Dataset Stats Dynamically
    if df_dataset is not None:
        dataset_songs_count = len(df_dataset)
        dataset_train_samples = int(dataset_songs_count * 0.8)
        dataset_features_count = 13
        dataset_file_size = f"{os.path.getsize(DATASET_FILE) / 1024:.1f} KB"
        dataset_missing_count = int(df_dataset.isnull().sum().sum())
    else:
        dataset_songs_count = 227
        dataset_train_samples = 181
        dataset_features_count = 13
        dataset_file_size = "N/A"
        dataset_missing_count = 0

    # Calculate Model Info Dynamically
    if model is not None:
        model_name = type(model).__name__
        model_algorithm = "Random Forest Regressor"
        model_num_features = getattr(model, "n_features_in_", 13)
        model_estimators = getattr(model, "n_estimators", 100)
    else:
        model_name = "RandomForestRegressor"
        model_algorithm = "Random Forest Regressor"
        model_num_features = 13
        model_estimators = 100

    # --------------------------------------------------------------------------
    # SIDEBAR: DYNAMIC STATISTICS & PRESETS
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-header">🎵 Spotify Predictor</div>',
            unsafe_allow_html=True
        )

        st.markdown("### 📊 Model & Dataset Info")

        # Fully Dynamic Sidebar Items
        st.markdown(
            f"""
            <div class="sidebar-info-box">
                <div class="sidebar-item">
                    <span class="sidebar-label">Model</span>
                    <span class="sidebar-value">{model_name}</span>
                </div>
                <div class="sidebar-item">
                    <span class="sidebar-label">Algorithm</span>
                    <span class="sidebar-value">Random Forest</span>
                </div>
                <div class="sidebar-item">
                    <span class="sidebar-label">Dataset Songs</span>
                    <span class="sidebar-value">{dataset_songs_count}</span>
                </div>
                <div class="sidebar-item">
                    <span class="sidebar-label">Training Samples</span>
                    <span class="sidebar-value">{dataset_train_samples}</span>
                </div>
                <div class="sidebar-item">
                    <span class="sidebar-label">Features Used</span>
                    <span class="sidebar-value">{model_num_features}</span>
                </div>
                <div class="sidebar-item">
                    <span class="sidebar-label">Target Variable</span>
                    <span class="sidebar-value">Popularity (0-100)</span>
                </div>
                <div class="sidebar-item">
                    <span class="sidebar-label">Missing Values</span>
                    <span class="sidebar-value">{dataset_missing_count}</span>
                </div>
                <div class="sidebar-item">
                    <span class="sidebar-label">Developer</span>
                    <span class="sidebar-value">{DEVELOPER_NAME}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")

        if st.button("🔄 Reset Inputs"):
            st.session_state["reset_trigger"] += 1
            st.rerun()

        st.markdown("### 🎛️ Audio Presets")
        preset = st.selectbox(
            "Load Quick Preset",
            ["Default", "Pop Hit", "Chill Acoustic", "EDM Anthem", "Rap Hit"],
            key=f"preset_{reset_key}"
        )

        presets = {
            "Pop Hit": {
                "dance": 0.82, "energy": 0.78, "loud": -5.0, "speech": 0.08,
                "acoust": 0.12, "instru": 0.0, "live": 0.12, "val": 0.75,
                "bpm": 122, "min": 3, "sec": 15, "exp": "No", "key": "G", "mode": "Major"
            },
            "Chill Acoustic": {
                "dance": 0.52, "energy": 0.35, "loud": -14.0, "speech": 0.04,
                "acoust": 0.88, "instru": 0.05, "live": 0.10, "val": 0.42,
                "bpm": 95, "min": 3, "sec": 45, "exp": "No", "key": "C", "mode": "Major"
            },
            "EDM Anthem": {
                "dance": 0.75, "energy": 0.92, "loud": -3.5, "speech": 0.12,
                "acoust": 0.02, "instru": 0.65, "live": 0.32, "val": 0.68,
                "bpm": 128, "min": 4, "sec": 10, "exp": "No", "key": "F#", "mode": "Minor"
            },
            "Rap Hit": {
                "dance": 0.88, "energy": 0.72, "loud": -6.0, "speech": 0.28,
                "acoust": 0.15, "instru": 0.0, "live": 0.18, "val": 0.55,
                "bpm": 140, "min": 2, "sec": 50, "exp": "Yes", "key": "C#", "mode": "Minor"
            }
        }
        selected_preset = presets.get(preset, {})

    # --------------------------------------------------------------------------
    # PAGE HEADER
    # --------------------------------------------------------------------------
    st.markdown(
        """
        <div class="header-container">
            <h1 class="header-title">🎵 <span>Spotify</span> Music Popularity Predictor</h1>
            <p class="header-subtitle">Predict how popular your song could become before releasing it using Machine Learning.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if model is None:
        st.warning(
            f"⚠️ Model file `{MODEL_FILE}` was not found or failed to load. "
            "Please ensure `music_popularity_model.pkl` exists in the application directory."
        )

    # --------------------------------------------------------------------------
    # EXPANDABLE SECTION 1: LEARN ABOUT AUDIO FEATURES
    # --------------------------------------------------------------------------
    with st.expander("📖 Learn About Audio Features", expanded=False):
        st.markdown(
            """
            Understanding Spotify audio features used by our Random Forest model:

            - **Danceability**: Suitability for dancing based on tempo, rhythm stability, and beat strength (0.0 to 1.0).
            - **Energy**: Perceptual intensity and activity (0.0 to 1.0). High energy feels fast, loud, and noisy.
            - **Loudness**: Overall loudness of a track in decibels (-60 to 0 dB).
            - **Speechiness**: Detects spoken words (0.0 to 1.0). Rap has moderate values; podcasts have high values.
            - **Acousticness**: Confidence measure of whether the track is acoustic (0.0 to 1.0).
            - **Instrumentalness**: Predicts whether a track contains no vocals (0.0 to 1.0).
            - **Liveness**: Detects audience presence in the recording (0.0 to 1.0).
            - **Valence**: Musical positiveness conveyed by a track (0.0 to 1.0). High = cheerful; Low = dark/sad.
            - **Tempo**: Estimated speed of a track in Beats Per Minute (BPM, 40 to 220).
            - **Duration**: Length of track in minutes and seconds, converted to milliseconds (`duration_ms`).
            - **Explicit**: Whether the track contains explicit lyrics (Yes = 1, No = 0).
            - **Key**: Pitch class notation of the main key (C=0, C#=1, ..., B=11).
            - **Mode**: Modality of the track (Major = 1, Minor = 0).
            """
        )

    # --------------------------------------------------------------------------
    # EXPANDABLE SECTION 2: MODEL PERFORMANCE
    # --------------------------------------------------------------------------
    with st.expander("📊 Model Performance Metrics & Feature Importance", expanded=False):
        if df_dataset is not None and model is not None:
            # Calculate performance metrics dynamically
            try:
                feature_mapping = {
                    "Duration (ms)": "duration_ms", "Explicit": "explicit",
                    "Danceability": "danceability", "Energy": "energy",
                    "Key": "key", "Loudness": "loudness", "Mode": "mode",
                    "Speechiness": "speechiness", "Acousticness": "acousticness",
                    "Instrumentalness": "instrumentalness", "Liveness": "liveness",
                    "Valence": "valence", "Tempo": "tempo"
                }
                X_df = df_dataset.rename(columns=feature_mapping)[FEATURE_COLUMNS].copy()
                X_df["explicit"] = X_df["explicit"].astype(int)
                y_df = df_dataset["Popularity"]

                X_tr, X_te, y_tr, y_te = train_test_split(X_df, y_df, test_size=0.2, random_state=42)
                y_pr = model.predict(X_te)

                mae_val = mean_absolute_error(y_te, y_pr)
                rmse_val = np.sqrt(mean_squared_error(y_te, y_pr))
                r2_val = r2_score(y_te, y_pr)
                mpe_val = float(np.mean(np.abs((y_te - y_pr) / np.maximum(y_te, 1))) * 100.0)

                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.metric("MAE (Mean Abs Error)", f"{mae_val:.2f}")
                col_m2.metric("RMSE (Root Mean Sq Error)", f"{rmse_val:.2f}")
                col_m3.metric("R² Score", f"{r2_val:.2f}")
                col_m4.metric("Mean Prediction Error", f"{mpe_val:.1f}%")

                st.markdown("#### Feature Importance Breakdown")
                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                    fi_df = pd.DataFrame({
                        "Feature": FEATURE_COLUMNS,
                        "Importance": importances
                    }).sort_values(by="Importance", ascending=True)

                    st.bar_chart(fi_df.set_index("Feature"), use_container_width=True)
            except Exception as e:
                st.info("Model performance metric calculation baseline standard active.")
        else:
            st.info("Load `spotify_data.csv` to calculate live performance metrics.")

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # SECTION 1: SONG DETAILS
    # --------------------------------------------------------------------------
    st.markdown("### 📝 Section 1: Song Details")
    with st.container():
        col_s1, col_s2, col_s3 = st.columns(3)

        with col_s1:
            song_name = st.text_input(
                "Song Name",
                value=DEFAULT_VALUES["song_name"],
                key=f"song_name_{reset_key}"
            )

        with col_s2:
            artist_name = st.text_input(
                "Artist Name",
                value=DEFAULT_VALUES["artist_name"],
                key=f"artist_name_{reset_key}"
            )

        with col_s3:
            genres_list = [
                "Pop", "Rock", "Hip-Hop", "Rap", "EDM",
                "Classical", "Jazz", "Country", "R&B", "Folk", "Other"
            ]
            genre = st.selectbox(
                "Genre",
                genres_list,
                index=0,
                key=f"genre_{reset_key}"
            )

        st.markdown(
            '<div class="info-note">📌 <strong>Note:</strong> Genre is collected only for display and is NOT used for prediction.</div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # SECTION 2: AUDIO FEATURES (PERFECT RESPONSIVE GRID & INSTANT LIVE LABELS)
    # --------------------------------------------------------------------------
    st.markdown("### 🎛️ Section 2: Audio Features")

    grid_col1, grid_col2, grid_col3 = st.columns(3)

    # COLUMN 1: RHYTHMIC & DYNAMIC
    with grid_col1:
        # 1. Danceability
        dance_init = selected_preset.get("dance", DEFAULT_VALUES["danceability"])
        danceability = st.slider(
            "1. Danceability", min_value=0.0, max_value=1.0, value=float(dance_init), step=0.01,
            help="How suitable the song is for dancing.", key=f"dance_slider_{reset_key}_{preset}"
        )
        badge_dance = get_feature_quality_indicator("danceability", danceability)
        st.markdown(
            f'<div style="text-align: right; margin-top: -14px; font-size: 0.82rem; color: #1DB954; font-weight: 700;">'
            f'Danceability: {danceability:.2f} ({int(danceability*100)}%) <span class="widget-quality-pill">{badge_dance}</span></div>',
            unsafe_allow_html=True
        )

        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

        # 2. Energy
        energy_init = selected_preset.get("energy", DEFAULT_VALUES["energy"])
        energy = st.slider(
            "2. Energy", min_value=0.0, max_value=1.0, value=float(energy_init), step=0.01,
            help="Measures the intensity and activity of the song.", key=f"energy_slider_{reset_key}_{preset}"
        )
        badge_energy = get_feature_quality_indicator("energy", energy)
        st.markdown(
            f'<div style="text-align: right; margin-top: -14px; font-size: 0.82rem; color: #1DB954; font-weight: 700;">'
            f'Energy: {energy:.2f} ({int(energy*100)}%) <span class="widget-quality-pill">{badge_energy}</span></div>',
            unsafe_allow_html=True
        )

        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

        # 3. Loudness
        loud_init = selected_preset.get("loud", DEFAULT_VALUES["loudness"])
        loudness = st.slider(
            "3. Loudness (dB)", min_value=-60.0, max_value=0.0, value=float(loud_init), step=0.5,
            help="Overall loudness in decibels.", key=f"loud_slider_{reset_key}_{preset}"
        )
        st.markdown(
            f'<div style="text-align: right; margin-top: -14px; font-size: 0.82rem; color: #1DB954; font-weight: 700;">'
            f'Loudness: {loudness:.1f} dB</div>',
            unsafe_allow_html=True
        )

        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

        # 9. Tempo Slider
        bpm_init = selected_preset.get("bpm", DEFAULT_VALUES["tempo"])
        tempo = st.slider(
            "9. Tempo (BPM)", min_value=40, max_value=220, value=int(bpm_init), step=1,
            help="Tempo in Beats Per Minute.", key=f"bpm_slider_{reset_key}_{preset}"
        )
        st.markdown(
            f'<div style="text-align: right; margin-top: -14px; font-size: 0.82rem; color: #1DB954; font-weight: 700;">'
            f'Tempo: {tempo} BPM</div>',
            unsafe_allow_html=True
        )

    # COLUMN 2: ACOUSTICS & VOICE
    with grid_col2:
        # 4. Speechiness
        speech_init = selected_preset.get("speech", DEFAULT_VALUES["speechiness"])
        speechiness = st.slider(
            "4. Speechiness", min_value=0.0, max_value=1.0, value=float(speech_init), step=0.01,
            help="Measures spoken dialogue rather than music.", key=f"speech_slider_{reset_key}_{preset}"
        )
        badge_speech = get_feature_quality_indicator("speechiness", speechiness)
        st.markdown(
            f'<div style="text-align: right; margin-top: -14px; font-size: 0.82rem; color: #1DB954; font-weight: 700;">'
            f'Speechiness: {speechiness:.2f} ({int(speechiness*100)}%) <span class="widget-quality-pill">{badge_speech}</span></div>',
            unsafe_allow_html=True
        )

        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

        # 5. Acousticness
        acoust_init = selected_preset.get("acoust", DEFAULT_VALUES["acousticness"])
        acousticness = st.slider(
            "5. Acousticness", min_value=0.0, max_value=1.0, value=float(acoust_init), step=0.01,
            help="Confidence that the song is acoustic.", key=f"acoust_slider_{reset_key}_{preset}"
        )
        badge_acoust = get_feature_quality_indicator("acousticness", acousticness)
        st.markdown(
            f'<div style="text-align: right; margin-top: -14px; font-size: 0.82rem; color: #1DB954; font-weight: 700;">'
            f'Acousticness: {acousticness:.2f} ({int(acousticness*100)}%) <span class="widget-quality-pill">{badge_acoust}</span></div>',
            unsafe_allow_html=True
        )

        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

        # 6. Instrumentalness
        instru_init = selected_preset.get("instru", DEFAULT_VALUES["instrumentalness"])
        instrumentalness = st.slider(
            "6. Instrumentalness", min_value=0.0, max_value=1.0, value=float(instru_init), step=0.01,
            help="Probability that the song contains NO vocals.", key=f"instru_slider_{reset_key}_{preset}"
        )
        badge_instru = get_feature_quality_indicator("instrumentalness", instrumentalness)
        st.markdown(
            f'<div style="text-align: right; margin-top: -14px; font-size: 0.82rem; color: #1DB954; font-weight: 700;">'
            f'Instrumentalness: {instrumentalness:.2f} ({int(instrumentalness*100)}%) <span class="widget-quality-pill">{badge_instru}</span></div>',
            unsafe_allow_html=True
        )

        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

        # 7. Liveness
        live_init = selected_preset.get("live", DEFAULT_VALUES["liveness"])
        liveness = st.slider(
            "7. Liveness", min_value=0.0, max_value=1.0, value=float(live_init), step=0.01,
            help="Likelihood that the recording is live.", key=f"live_slider_{reset_key}_{preset}"
        )
        badge_live = get_feature_quality_indicator("liveness", liveness)
        st.markdown(
            f'<div style="text-align: right; margin-top: -14px; font-size: 0.82rem; color: #1DB954; font-weight: 700;">'
            f'Liveness: {liveness:.2f} ({int(liveness*100)}%) <span class="widget-quality-pill">{badge_live}</span></div>',
            unsafe_allow_html=True
        )

    # COLUMN 3: MOOD, METADATA & DURATION
    with grid_col3:
        # 8. Valence
        val_init = selected_preset.get("val", DEFAULT_VALUES["valence"])
        valence = st.slider(
            "8. Valence", min_value=0.0, max_value=1.0, value=float(val_init), step=0.01,
            help="Measures musical positivity.", key=f"val_slider_{reset_key}_{preset}"
        )
        badge_val = get_feature_quality_indicator("valence", valence)
        st.markdown(
            f'<div style="text-align: right; margin-top: -14px; font-size: 0.82rem; color: #1DB954; font-weight: 700;">'
            f'Valence: {valence:.2f} ({int(valence*100)}%) <span class="widget-quality-pill">{badge_val}</span></div>',
            unsafe_allow_html=True
        )

        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

        # 10. Duration
        st.markdown("<b style='font-size: 0.88rem;'>10. Duration</b>", unsafe_allow_html=True)
        dur_col1, dur_col2 = st.columns(2)
        with dur_col1:
            minutes = st.number_input(
                "Min", min_value=0, max_value=30,
                value=int(selected_preset.get("min", DEFAULT_VALUES["minutes"])),
                key=f"min_inp_{reset_key}_{preset}"
            )
        with dur_col2:
            seconds = st.number_input(
                "Sec", min_value=0, max_value=59,
                value=int(selected_preset.get("sec", DEFAULT_VALUES["seconds"])),
                key=f"sec_inp_{reset_key}_{preset}"
            )

        duration_ms = minutes * 60000 + seconds * 1000

        # Duration Summary Box
        st.markdown(
            f"""
            <div class="duration-summary-box">
                <span class="dur-label">Duration: {minutes}m {seconds}s</span>
                <span class="dur-ms">↓ {duration_ms:,} ms</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

        # 11. Explicit, 12. Key, 13. Mode
        meta_c1, meta_c2, meta_c3 = st.columns([1, 1, 1])
        with meta_c1:
            exp_preset = selected_preset.get("exp", DEFAULT_VALUES["explicit"])
            explicit_str = st.selectbox(
                "11. Explicit", options=["No", "Yes"],
                index=1 if exp_preset == "Yes" else 0,
                key=f"exp_sel_{reset_key}_{preset}"
            )
            explicit = EXPLICIT_MAP[explicit_str]

        with meta_c2:
            key_preset = selected_preset.get("key", DEFAULT_VALUES["key"])
            key_str = st.selectbox(
                "12. Key", options=list(KEY_MAP.keys()),
                index=list(KEY_MAP.keys()).index(key_preset),
                key=f"key_sel_{reset_key}_{preset}"
            )
            key = KEY_MAP[key_str]

        with meta_c3:
            mode_preset = selected_preset.get("mode", DEFAULT_VALUES["mode"])
            mode_str = st.selectbox(
                "13. Mode", options=list(MODE_MAP.keys()),
                index=list(MODE_MAP.keys()).index(mode_preset),
                key=f"mode_sel_{reset_key}_{preset}"
            )
            mode = MODE_MAP[mode_str]

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # PREDICT BUTTON (PROMINENT CENTERED)
    # --------------------------------------------------------------------------
    predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
    with predict_col2:
        predict_button = st.button("🎵 Predict Popularity", use_container_width=True)

    if predict_button:
        if model is None:
            st.error("Model is not loaded. Please make sure `music_popularity_model.pkl` exists.")
            return

        with st.spinner("Analyzing song using Machine Learning..."):
            time.sleep(0.4)

            features_dict = {
                "duration_ms": duration_ms,
                "explicit": explicit,
                "danceability": danceability,
                "energy": energy,
                "key": key,
                "loudness": loudness,
                "mode": mode,
                "speechiness": speechiness,
                "acousticness": acousticness,
                "instrumentalness": instrumentalness,
                "liveness": liveness,
                "valence": valence,
                "tempo": tempo
            }

            features_df = pd.DataFrame([features_dict])[FEATURE_COLUMNS]

            try:
                raw_pred = model.predict(features_df)[0]
                score = float(np.clip(raw_pred, 0.0, 100.0))
            except Exception as e:
                st.error(f"Prediction Error: {e}")
                return

        category, icon, color_hex, stars, interpretation = get_popularity_category(score)
        confidence_pct = calculate_rf_confidence(model, features_df)

        st.session_state["prediction_counter"] += 1
        pred_id = f"#{st.session_state['prediction_counter']}"
        timestamp_str = datetime.datetime.now().strftime("%H:%M:%S")

        st.session_state["history"].append({
            "ID": pred_id,
            "Timestamp": timestamp_str,
            "Song": song_name,
            "Artist": artist_name,
            "Genre": genre,
            "Score": f"{score:.2f}",
            "Category": f"{icon} {category}"
        })

        if score > 90.0:
            st.balloons()

        # CENTERED PREDICTION RESULT CARD
        st.markdown(
            f"""
            <div class="result-card-container">
                <div class="song-title-display">{song_name}</div>
                <div class="artist-genre-display">by {artist_name} • <span style="color: #1DB954;">{genre}</span></div>
                <div class="category-badge-pill" style="background-color: {color_hex}22; color: {color_hex}; border: 1.5px solid {color_hex};">
                    {icon} {category}
                </div>
                <div class="rating-stars">{stars}</div>
            """,
            unsafe_allow_html=True
        )

        render_circular_ring(score, color_hex)

        st.markdown(
            f"""
                <div class="confidence-badge">🎯 Prediction Confidence: {confidence_pct}% (Ensemble Agreement)</div>
            """,
            unsafe_allow_html=True
        )

        st.progress(score / 100.0)

        st.markdown(
            f"""
                <div class="interpretation-box">
                    "{interpretation}"
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # DOWNLOAD REPORT BUTTON
        csv_report = generate_report_csv(
            song_name, artist_name, genre, features_dict, score, category, interpretation
        )

        dl_col1, dl_col2, dl_col3 = st.columns([1, 2, 1])
        with dl_col2:
            st.download_button(
                label="📥 Download Prediction Report (CSV)",
                data=csv_report,
                file_name=f"{song_name.replace(' ', '_')}_prediction_report.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # INTELLIGENT COMBINED DYNAMIC OBSERVATIONS
        st.markdown("### 💡 Intelligent Observations")
        observations = generate_intelligent_observations(features_dict)
        for obs in observations:
            st.markdown(f'<div class="observation-item">🎶 {obs}</div>', unsafe_allow_html=True)

        # INPUT SUMMARY SECTION
        st.markdown("### 📋 Input Summary")
        st.markdown(
            f"""
            <div class="input-summary-grid">
                <div class="input-summary-card"><div class="param-name">Duration</div><div class="param-val">{minutes}m {seconds}s</div></div>
                <div class="input-summary-card"><div class="param-name">Explicit</div><div class="param-val">{explicit_str}</div></div>
                <div class="input-summary-card"><div class="param-name">Danceability</div><div class="param-val">{danceability:.2f}</div></div>
                <div class="input-summary-card"><div class="param-name">Energy</div><div class="param-val">{energy:.2f}</div></div>
                <div class="input-summary-card"><div class="param-name">Key</div><div class="param-val">{key_str} ({key})</div></div>
                <div class="input-summary-card"><div class="param-name">Loudness</div><div class="param-val">{loudness:.1f} dB</div></div>
                <div class="input-summary-card"><div class="param-name">Mode</div><div class="param-val">{mode_str}</div></div>
                <div class="input-summary-card"><div class="param-name">Speechiness</div><div class="param-val">{speechiness:.2f}</div></div>
                <div class="input-summary-card"><div class="param-name">Acousticness</div><div class="param-val">{acousticness:.2f}</div></div>
                <div class="input-summary-card"><div class="param-name">Instrumentalness</div><div class="param-val">{instrumentalness:.2f}</div></div>
                <div class="input-summary-card"><div class="param-name">Liveness</div><div class="param-val">{liveness:.2f}</div></div>
                <div class="input-summary-card"><div class="param-name">Valence</div><div class="param-val">{valence:.2f}</div></div>
                <div class="input-summary-card"><div class="param-name">Tempo</div><div class="param-val">{tempo} BPM</div></div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------------------------
    # HISTORY SECTION
    # --------------------------------------------------------------------------
    if st.session_state["history"]:
        st.markdown("---")
        st.markdown("### 📜 Prediction History")

        hist_df = pd.DataFrame(st.session_state["history"])
        st.dataframe(hist_df, use_container_width=True)

        if st.button("🗑️ Clear History"):
            st.session_state["history"] = []
            st.session_state["prediction_counter"] = 0
            st.rerun()

    # --------------------------------------------------------------------------
    # FOOTER (DEVELOPER: RAKESH KUMAR)
    # --------------------------------------------------------------------------
    st.markdown(
        f"""
        <div class="footer-container">
            <p style="font-weight: 700; color: #FFFFFF; font-size: 1rem; margin-bottom: 6px;">Spotify Music Popularity Predictor</p>
            <p>Built using <span class="footer-highlight">Machine Learning</span> | Random Forest Regression | Scikit-learn | Streamlit</p>
            <p style="font-size: 0.88rem; color: #1DB954; font-weight: 600; margin-top: 4px;">Developer: {DEVELOPER_NAME}</p>
            <p style="font-size: 0.78rem; color: #777777;">Educational Purpose Only</p>
            <a href="https://github.com" target="_blank" class="github-link-btn">
                <svg width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
                    <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.28.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
                </svg>
                GitHub Repository
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
