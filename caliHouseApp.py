import streamlit as st
import joblib
import pandas as pd
import numpy as np
import gdown
import os

# --- Page Configuration (should be at the very top of the script) ---
st.set_page_config(page_title="California Housing App", layout="centered")

# --- Load model and expected features ---
file_id = "10ne32XEWjXf5SX1zi2q2ic3VIbodZqAN" # Re-check this ID if you updated the model
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

# --- Load model and expected columns with caching ---
@st.cache_resource
def load_resources():
    model = joblib.load(model_path)
    model_features = joblib.load('model_features.joblib') # feature names used during training
    # Assuming you've saved your scaling_params dictionary for consistency
    # If you used sklearn.preprocessing.StandardScaler, save that object instead
    # For now, let's assume 'scaling_params.joblib' exists based on previous discussion
    if os.path.exists('scaling_params.joblib'):
        scaling_params = joblib.load('scaling_params.joblib')
    else:
        st.warning("Scaling parameters not found. Predictions might be inaccurate if model was trained on scaled data.")
        scaling_params = None # Handle case where it's not available
    return model, model_features, scaling_params

model, model_features, scaling_params = load_resources()

# --- Streamlit app title ---
st.title('🏡 California Housing Price Prediction')

# --- Input options ---
ocean_proximity_options = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']

# --- User Inputs - Consolidated into a single section ---
st.header("🎯 Enter Location and Housing Details:")

# Combined Latitude/Longitude slider (fixing the 'step' error)
lat_lon = st.slider(
    "Select (Latitude, Longitude)",
    min_value=(32.54, -124.48),
    max_value=(42.01, -114.13),
    value=(37.0, -122.0),
    step=0.01 # FIX: Changed step to a single float value
)
latitude, longitude = lat_lon

# Validate coordinates immediately
if not (32.54 <= latitude <= 42.01 and -124.48 <= longitude <= -114.13):
    st.error("❌ Coordinates are outside California's typical range. Please select a valid location within California.")
    # st.stop() # Removed st.stop() here to allow other inputs to show, but prediction will be prevented below.

col1, col2 = st.columns(2)

with col1:
    housing_median_age = st.slider("Housing Median Age", 1, 52, 30)
    median_income = st.slider("Median Income (in tens of thousands)", 0.5, 15.0, 5.0, step=0.1)
    # Correcting for median_income conversion if your model expects actual dollars, not tens of thousands.
    # If your model's median_income was in actual tens of thousands, this is fine.
    # Otherwise, you might need: median_income_actual = median_income * 10000

with col2:
    total_rooms = st.slider("Total Rooms (block group)", 2, 39320, 2000)
    total_bedrooms = st.slider("Total Bedrooms", 1, 6445, 400)
    population = st.slider("Population", 3, 35682, 1000)
    households = st.slider("Households", 1, 6082, 350)
    ocean_proximity = st.selectbox("Ocean Proximity", ocean_proximity_options)


# --- Predict Button ---
if st.button("Predict Median House Value"):
    # Check if coordinates are valid before predicting
    if not (32.54 <= latitude <= 42.01 and -124.48 <= longitude <= -114.13):
        st.warning("Please adjust coordinates to be within California before predicting.")
    else:
        # Step 1: Form raw DataFrame
        input_data = pd.DataFrame({
            'longitude': [longitude],
            'latitude': [latitude],
            'housing_median_age': [housing_median_age],
            'total_rooms': [total_rooms],
            'total_bedrooms': [total_bedrooms],
            'population': [population],
            'households': [households],
            'median_income': [median_income], # Assuming this matches your training feature
            'ocean_proximity': [ocean_proximity]
        })

        # Step 2: Feature Engineering (consistent with training)
        input_data['rooms_per_household'] = input_data['total_rooms'] / input_data['households']
        input_data['bedrooms_per_room'] = input_data['total_bedrooms'] / input_data['total_rooms']
        input_data['population_per_household'] = input_data['population'] / input_data['households'] # Added this if you used it during training

        # Step 3: One-Hot Encoding for 'ocean_proximity' (consistent with training)
        input_data = pd.get_dummies(input_data, columns=['ocean_proximity'], prefix='ocean_proximity')

        # Step 4: Align input to match training features (crucial step!)
        # Ensure all columns expected by the model are present, fill missing with 0
        input_data = input_data.reindex(columns=model_features, fill_value=0)

        # Step 5: Apply Scaling (CRUCIAL IF MODEL WAS TRAINED ON SCALED DATA)
        # This assumes scaling_params is a dict saved using joblib, containing {'mean': mean_val, 'std': std_val} for each col
        # If you used an sklearn scaler object (e.g., StandardScaler), load that instead and use its .transform() method
        if scaling_params: # Only attempt if scaling_params were loaded
            numerical_cols_for_scaling_in_app = [col for col in model_features if not col.startswith('ocean_proximity_')]
            for col in numerical_cols_for_scaling_in_app:
                if col in scaling_params: # Check if params exist for this col
                    mean_val = scaling_params[col]['mean']
                    std_val = scaling_params[col]['std']
                    if std_val == 0:
                        input_data[col] = 0
                    else:
                        input_data[col] = (input_data[col] - mean_val) / std_val
                else:
                    st.warning(f"Scaling parameters for '{col}' not found. Prediction might be inaccurate.")
                    # You might want to handle this more robustly, e.g., stop prediction

        # Step 6: Predict
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Median House Value: ${prediction:,.2f}")

# --- Location Map Viewer (uses the same lat/lon from the prediction section) ---
st.markdown("---") # Separator
st.subheader("🗺️ Location on Map")
st.markdown("### 🎯 Chosen Coordinates:")

# Create DataFrame for Map using the lat/lon from the single slider
map_df = pd.DataFrame({'lat': [latitude], 'lon': [longitude]})
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

# --- Optional Styling (moved to the end to ensure it applies after all content) ---
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
