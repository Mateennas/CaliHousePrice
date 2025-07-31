import streamlit as st
import joblib
import pandas as pd
import numpy as np
import gdown
import os
from shapely.geometry import Point, Polygon 


st.set_page_config(page_title="California Housing Price Prediction", layout="centered")


# --- Load model and expected features
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

# Load model and expected columns 
@st.cache_resource
def load_resources():
    model = joblib.load(model_path)
    model_features = joblib.load('model_features.joblib') 
    return model, model_features

model, model_features = load_resources()

# --- Define California Polygon for Precise Check ---
california_border_coords = [
    (-124.48, 32.53), 
    (-117.0, 32.53), 
    (-114.13, 34.99), 
    (-119.99, 41.99),
    (-124.2, 42.01),
    (-124.48, 32.53)  
]
california_polygon = Polygon(california_border_coords)


st.title(' California Housing Price Prediction')

# --- Input options ---
ocean_proximity_options = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']

# --- User Inputs ---
st.header("Enter Block Group Details:")

loc_col1, loc_col2 = st.columns(2)

with loc_col1:
    latitude = st.slider(
        "Latitude",
        min_value=32.54,
        max_value=42.01,
        value=37.0,
        step=0.01
    )
with loc_col2:
    longitude = st.slider(
        "Longitude",
        min_value=-124.48,
        max_value= -114.13,
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
    user_point = Point(longitude, latitude) #create point where users values are

    if california_polygon.contains(user_point): #If witin the points, then proceed
        
        input_data = pd.DataFrame({
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

        input_data['rooms_per_household'] = input_data['total_rooms'] / input_data['households']
        input_data['bedrooms_per_room'] = input_data['total_bedrooms'] / input_data['total_rooms']

        input_data = pd.get_dummies(input_data, columns=['ocean_proximity'], prefix='ocean_proximity')

        input_data = input_data.reindex(columns=model_features, fill_value=0)

        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Median House Value: ${prediction:,.2f}")
    else:
        st.error(" Prediction aborted: The selected location is outside California's landmass. Please select a point within the state's boundaries for an accurate prediction.")


st.markdown(
    f"""
    <style>
    .stApp {{
        background: url("https://img.freepik.com/premium-photo/dark-photo-with-minimalistic-design_1148436-2169.jpg");
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
st.markdown("###  Chosen Coordinates")

map_df = pd.DataFrame({'lat': [latitude], 'lon': [longitude]})

# --- Display Map ---
st.subheader(" Location on Map")
st.map(map_df, zoom=6)

