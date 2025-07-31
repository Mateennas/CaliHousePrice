import streamlit as st
import joblib
import pandas as pd
import numpy as np
import gdown
import os

# --- Load model and expected features ---
file_id = "1WxEGMMnhVcAFR065XJNPIowo4yXBVXKw"
url = f"https://drive.google.com/uc?id={file_id}"
model_path = "best_house_price_model.joblib"

# Download the model if needed
if not os.path.exists(model_path):
    gdown.download(url, model_path, quiet=False)

# Load model and expected columns with caching
@st.cache_resource
def load_model():
    return joblib.load(model_path)

model = load_model()
model_features = joblib.load('model_features.joblib')  # feature names used during training

# --- Streamlit app title ---
st.title('California Housing Price Prediction')

# --- Input options ---
ocean_proximity_options = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']

# --- User Inputs ---
st.header("Enter Block Group Details:")

col1, col2 = st.columns(2)

with col1:
    longitude = st.slider("Longitude", -124.35, -114.31, -122.0)
    latitude = st.slider("Latitude", 32.54, 41.95, 37.0)
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
    # Step 1: Form raw DataFrame
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

    # Step 2: Feature Engineering
    input_data['rooms_per_household'] = input_data['total_rooms'] / input_data['households']
    input_data['bedrooms_per_room'] = input_data['total_bedrooms'] / input_data['total_rooms']

    # Step 3: One-Hot Encoding for 'ocean_proximity'
    input_data = pd.get_dummies(input_data, columns=['ocean_proximity'], prefix='ocean_proximity')

    # Step 4: Align input to match training features
    input_data = input_data.reindex(columns=model_features, fill_value=0)

    # Step 5: Predict
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Median House Value: ${prediction:,.2f}")

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
