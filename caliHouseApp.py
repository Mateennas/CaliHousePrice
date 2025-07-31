import streamlit as st
import joblib
import pandas as pd
import numpy as np
import gdown
import os
from shapely.geometry import Point, Polygon

# --- Page Configuration (MUST be at the very top of your script) ---
st.set_page_config(page_title="California Housing Price Prediction", layout="centered")

# --- Load model and expected features ---
file_id = "10ne32XEWjXf9SX1zi2q2ic3VIbodZqAN"
url = f"https://drive.google.com/uc?id={file_id}"
model_path = "best_house_price_model.joblib"

# Download the model if needed
if not os.path.exists(model_path):
    st.info(f"Downloading model: {model_path}...")
    try:
        gdown.download(url, model_path, quiet=False)
        st.success("Model downloaded successfully!")
    except Exception as e:
        st.error(f"Failed to download model: {e}")
        st.stop() # Stop if model can't be loaded

# Load model and expected columns with caching
@st.cache_resource
def load_resources():
    model = joblib.load(model_path)
    # If your model is a complete pipeline, 'model_features.joblib' might not be strictly needed for prediction,
    # as the pipeline internally manages feature names and order.
    # However, it could still be useful for validation or if your pipeline expects a specific DataFrame structure.
    # I'm keeping the loading here but acknowledging its potential redundancy for a truly end-to-end pipeline.
    try:
        model_features = joblib.load('model_features.joblib')
    except FileNotFoundError:
        st.warning("model_features.joblib not found. Ensure your model is a complete pipeline that handles all feature transformations and ordering internally.")
        model_features = None # Set to None or handle as appropriate for your specific model
    return model, model_features

model, model_features = load_resources()

# --- Define California Polygon for Precise Check ---
# This is a VERY simplified, hand-drawn polygon for demonstration.
# For high accuracy, you would load a proper GeoJSON or shapefile.
# Coordinates are (longitude, latitude)
california_border_coords = [
    (-124.48, 32.53), # South-west (near San Diego / border)
    (-117.0, 32.53),  # South-east (near Calexico / border)
    (-114.13, 34.99), # East-central (near Needles, border with AZ)
    (-119.99, 41.99), # North-east (near Lake Tahoe, border with NV)
    (-124.2, 42.01),  # North-west (near Crescent City / Oregon border)
    (-124.48, 32.53)  # Closing the loop to the start point
]
california_polygon = Polygon(california_border_coords)


# --- Streamlit app title ---
st.title('🏡 California Housing Price Prediction')

# --- Input options ---
ocean_proximity_options = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']

# --- User Inputs ---
st.header("Enter Block Group Details:")

# Using columns to put Lat/Lon sliders side-by-side for a combined feel
loc_col1, loc_col2 = st.columns(2)

with loc_col1:
    latitude = st.slider(
        "Latitude",
        min_value=32.54, # California's approximate min latitude
        max_value=42.01, # California's approximate max latitude
        value=37.0,
        step=0.01
    )
with loc_col2:
    longitude = st.slider(
        "Longitude",
        min_value=-124.48, # California's approximate min longitude
        max_value= -114.13, # California's approximate max longitude
        value=-122.0,
        step=0.01
    )

# --- Remaining inputs in original columns ---
col1, col2 = st.columns(2)

with col1:
    housing_median_age = st.slider("Housing Median Age", 1, 52, 30)
    median_income = st.slider("Median Income (in tens of thousands)", 0.5, 15.0, 5.0, step=0.1)

with col2:
    total_rooms = st.slider("Total Rooms (block group)", 2, 39320, 2000)
    total_bedrooms = st.slider("Total Bedrooms", 1, 6445, 400)
    population = st.slider("Population", 3, 35682, 1000)
    households = st.slider("Households", 1, 6082, 350)
    ocean_proximity = st.selectbox("Ocean Proximity", ocean_proximity_options)

# --- Predict Button ---
if st.button("Predict Median House Value"):
    # *** MAIN FIX: DATA PREPARATION SIMPLIFIED TO ONLY RAW INPUTS ***
    # <--- MODIFIED CODE BLOCK START ---
    # Create the DataFrame with only the RAW input features as received from the user.
    # Assumes the loaded 'model' is a scikit-learn Pipeline that internally handles
    # feature engineering (e.g., rooms_per_household, bedrooms_per_room),
    # one-hot encoding for 'ocean_proximity', feature alignment, and scaling.
    input_df_raw = pd.DataFrame({
        'longitude': [longitude],
        'latitude': [latitude],
        'housing_median_age': [housing_median_age],
        'total_rooms': [total_rooms],
        'total_bedrooms': [total_bedrooms],
        'population': [population],
        'households': [households],
        'median_income': [median_income],
        'ocean_proximity': [ocean_proximity]
    })
    # <--- MODIFIED CODE BLOCK END ---

    # *** GEOGRAPHICAL CHECK BEFORE PREDICTION ***
    user_point = Point(longitude, latitude) # Create a shapely Point from user input (lon, lat)

    if california_polygon.contains(user_point):
        # The point is within the (simplified) California landmass, proceed with prediction
        # Step: Predict using the loaded model (pipeline) on the raw input DataFrame.
        prediction = model.predict(input_df_raw)[0]
        st.success(f"Predicted Median House Value: ${prediction:,.2f}")
    else:
        # The point is outside the (simplified) California landmass
        st.error("❌ Prediction aborted: The selected location is outside California's landmass. Please select a point within the state's boundaries for an accurate prediction.")

# --- Optional Styling ---
st.markdown(
    f"""
    <style>
    .stApp {{
        background: url("https://static.vecteezy.com/system/resources/thumbnails/026/185/327/small_2x/waterfall-and-stone-copy-space-blurred-background-ai-generated-photo.jpg");
        background-size: cover;
        background-attachment: fixed;
    }}
    .stApp::before {{
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(255,255,255,0.7);
        z-index: -1;
    }}
    h1, h2, h3, h4, h5, h6, label {{
        color: #222;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# --- California Housing Location Viewer ---
st.title("California Housing Location Viewer")
st.markdown("### 🎯 Chosen Coordinates")

# --- Create DataFrame for Map ---
# Uses the 'latitude' and 'longitude' variables obtained from the separate sliders
map_df = pd.DataFrame({'lat': [latitude], 'lon': [longitude]})

# --- Display Map ---
st.subheader("🗺️ Location on Map")
st.map(map_df, zoom=6)

# --- Optional Info/Image Display ---
st.subheader("📍 Location Info (Simplified Demo)")
if -123 < longitude < -122 and 37 < latitude < 38:
    st.image("https://upload.wikimedia.org/wikipedia/commons/1/12/San_Francisco_from_Twin_Peaks_November_2019_panorama_2.jpg", width=600)
    st.markdown("**You are likely viewing the San Francisco Bay Area!**")
elif -119 < longitude < -117 and 33 < latitude < 35:
    st.image("https://upload.wikimedia.org/wikipedia/commons/8/89/Los_Angeles_Skyline_at_Night.jpg", width=600)
    st.markdown("**This looks like the Los Angeles region.**")
else:
    st.markdown("🌎 This point is somewhere else in California.")
