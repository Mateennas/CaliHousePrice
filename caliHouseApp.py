import streamlit as st
import joblib
import pandas as pd
import numpy as np
import gdown
import os

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
    model_features = joblib.load('model_features.joblib') # feature names used during training
    # IMPORTANT: If your model was trained on SCALED data, you MUST load your scaler here too!
    # For example:
    # if os.path.exists('scaling_params.joblib'):
    #     scaling_params = joblib.load('scaling_params.joblib')
    # else:
    #     st.warning("Scaling parameters not found. Predictions might be inaccurate if model was trained on scaled data.")
    #     scaling_params = None # Handle this case as needed
    # return model, model_features, scaling_params # If you also load scaling_params
    return model, model_features

model, model_features = load_resources()
# model, model_features, scaling_params = load_resources() # If you load scaling_params

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
        max_value=-114.13, # California's approximate max longitude
        value=-122.0,
        step=0.01
    )

# --- YOUR REQUESTED LOCATION CHECK ---
# This will stop the app if the coordinates are outside the specified rectangle.
if not (32.54 <= latitude <= 42.01 and -124.48 <= longitude <= -114.13):
    st.error("❌ Selected coordinates are outside of California.")
    st.stop()

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

    # Step 2: Feature Engineering (Ensure this matches your training preprocessing!)
    input_data['rooms_per_household'] = input_data['total_rooms'] / input_data['households']
    input_data['bedrooms_per_room'] = input_data['total_bedrooms'] / input_data['total_rooms']
    # If you had population_per_household in training, add it here:
    # input_data['population_per_household'] = input_data['population'] / input_data['households']


    # Step 3: One-Hot Encoding for 'ocean_proximity' (consistent with training)
    input_data = pd.get_dummies(input_data, columns=['ocean_proximity'], prefix='ocean_proximity')

    # Step 4: Align input to match training features (crucial step!)
    input_data = input_data.reindex(columns=model_features, fill_value=0)

    # Step 5: Apply Scaling (CRUCIAL IF MODEL WAS TRAINED ON SCALED DATA)
    # If you loaded 'scaling_params' or an 'sklearn.preprocessing.StandardScaler' object
    # you would apply it here using that loaded object. Example:
    # if scaling_params: # If you loaded a dict of params
    #    numerical_cols_for_scaling_in_app = [col for col in model_features if not col.startswith('ocean_proximity_')]
    #    for col in numerical_cols_for_scaling_in_app:
    #        if col in scaling_params:
    #            mean_val = scaling_params[col]['mean']
    #            std_val = scaling_params[col]['std']
    #            if std_val == 0:
    #                input_data[col] = 0
    #            else:
    #                input_data[col] = (input_data[col] - mean_val) / std_val
    # elif isinstance(scaler_object, StandardScaler): # If you loaded an sklearn scaler object
    #    input_data[numerical_cols_for_scaling_in_app] = scaler_object.transform(input_data[numerical_cols_for_scaling_in_app])
    # else:
    #    st.warning("Scaling was not applied. Ensure your model was trained on unscaled data or load the scaler.")


    # Step 6: Predict
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
    st.image("
