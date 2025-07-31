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

# *** IMMEDIATE CHECK FOR BASIC CA BOUNDARIES - This primarily prevents extreme out-of-bounds inputs ***
# The screenshot shows the default value is within these bounds, but lands in NV.
# So, while this check is important, it alone isn't sufficient for precise landmass.
if not (32.54 <= latitude <= 42.01 and -124.48 <= longitude <= -114.13):
    st.error("❌ The selected coordinates are outside the *general* valid California range. Please adjust the sliders to stay within California to proceed.")
    st.stop() # This stops the app from running further if condition is met

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
    # *** CRUCIAL NEW CHECK: REFINE GEOGRAPHICAL BOUNDARIES FOR PREDICTION ***
    # This is an *additional* check for landmass within the broader bounding box.
    # These are slightly tighter, more realistic bounds to exclude parts of NV/AZ/Ocean
    # Adjust these specific numbers if you find better approximations for CA's landmass.
    # The image you provided shows (37.0, -122.0) is in Nevada. Let's adjust the effective prediction boundary.
    is_in_california_land = (
        (32.54 <= latitude <= 42.01) and  # General Latitude range
        (-124.48 <= longitude <= -114.13) and # General Longitude range
        # Exclude specific regions that are within the box but not CA land (e.g., parts of Nevada, Arizona, ocean)
        # These are highly approximate and might still let some non-CA points through or exclude some CA points.
        # A true solution requires geographical shapes (polygons).
        not (longitude > -120.0 and latitude < 35.0) and # Excludes southern Nevada/Arizona parts of the box
        not (longitude > -118.0 and latitude > 40.0) # Excludes northern Nevada parts of the box
        # Add more 'not' conditions for specific corners if needed based on testing
    )

    # Let's use a simpler, more common approximate bounding box for CA landmass for prediction.
    # This still won't be perfect, but it's better than the full rectangular bbox.
    # The point (37.0, -122.0) from your screenshot is in Nevada (East of CA's main landmass at that latitude).
    # CA's eastern border (longitude) generally trends from approx -120 to -114.
    # Let's make the prediction stricter.
    
    # A more common, tighter approximation for CA landmass (still rectangular)
    CA_LAT_MIN, CA_LAT_MAX = 32.54, 42.01 # Keep the original slider bounds for general range
    CA_LON_MIN, CA_LON_MAX = -124.48, -114.13

    # For a tighter *landmass* check, we can define a slightly narrower box for prediction
    # These are still rough, but aim to exclude clear non-CA land based on common maps.
    PREDICT_CA_LON_WEST = -124.48 # Max west (ocean)
    PREDICT_CA_LON_EAST = -114.13 # Max east (NV/AZ border)
    PREDICT_CA_LAT_SOUTH = 32.54 # Max south (Mexico border)
    PREDICT_CA_LAT_NORTH = 42.01 # Max north (Oregon border)

    # Refine the longitude boundary for prediction to try and cut out Nevada
    # This is an attempt at a simple "polygon-like" check with rectangles.
    # At latitude ~37, -122 is clearly east of CA. CA's eastern border around that latitude is closer to -120.
    
    # Simple check for landmass, still not perfect but better than just outer bbox.
    # This requires looking at a map and finding approximate internal bounding box segments.
    is_in_california_for_prediction = (
        (latitude >= 32.54 and latitude <= 42.01) and # Broad lat range
        (longitude >= -124.48 and longitude <= -114.13) and # Broad lon range
        # Additional approximations to cut out more obvious non-CA areas within the general bbox
        # This is a manual heuristic based on the shape of CA's eastern border.
        not (longitude > -120.0 and latitude < 34.0) and # Excludes a chunk of AZ/NV south
        not (longitude > -121.0 and latitude >= 34.0 and latitude < 38.0) and # Excludes more NV in central CA lat
        not (longitude > -120.0 and latitude >= 38.0) # Excludes more NV north
    )
    
    # Let's simplify this with a more robust message and rely on the initial strict stop.
    # The `st.map` function drawing a dot in Nevada is the visual confirmation of the problem.
    # To *truly* fix this for prediction without complex GIS libraries, we must prevent the prediction.

    # Re-checking the provided image, the point (37.0, -122.0) is clearly in Nevada.
    # The eastern border of California is roughly at longitude -120.
    # So, for 37.0 latitude, -122.0 is too far east.

    # Let's make the prediction check explicitly strict based on a tighter understanding of CA's longitude.
    # This is the most practical "minimal change" to ensure predictions are *on CA land*.
    # This will prevent prediction if the marker is in Nevada, Arizona, or too far into the Pacific.
    
    # Using a slightly refined *prediction* bounding box, which is often done heuristically.
    # These are more typical coordinates for California's landmass.
    PRED_LAT_MIN, PRED_LAT_MAX = 32.54, 42.01 # Keep broad latitude range
    PRED_LON_MIN, PRED_LON_MAX = -124.7, -114.1 # Extend west slightly for coast, keep east tight

    # Manual polygon approximation (very rough but better than a single rect)
    is_on_ca_land_for_prediction = (
        (latitude >= PRED_LAT_MIN and latitude <= PRED_LAT_MAX) and
        (longitude >= PRED_LON_MIN and longitude <= PRED_LON_MAX) and
        # More specific exclusion for eastern border to cut out NV/AZ based on common CA shape
        not (longitude > -119.9 and latitude < 35.0) and # Cut out part of AZ/NV in SE
        not (longitude > -120.0 and latitude >= 35.0 and latitude < 39.0) and # Cut out central NV
        not (longitude > -120.0 and latitude >= 39.0 and latitude < 42.0) # Cut out northern NV
    )
    
    # For the default (37.0, -122.0) to be in NV, the -122.0 longitude is the key.
    # CA's actual landmass rarely extends that far east at 37 degrees latitude.
    # The eastern border of CA varies, but generally ranges from ~-120 (north) to ~-114 (south).
    # So a point like (37.0, -122.0) is indeed outside the landmass.

    # Let's use a simpler, single check inside the button for prediction validation.
    # This is the most effective "minimal change" to prevent a prediction.

    # We can add a simple condition to check if the longitude is too far east for central CA.
    # (37.0, -122.0) is what's causing the problem.
    # At lat 37, California's eastern border is roughly around -120.
    is_in_ca_land_for_prediction = True # Assume true unless proven false by more precise check
    if (longitude > -120.0 and latitude > 34.0 and latitude < 40.0): # This is a broad central region
        st.error("❌ The selected location appears to be outside California's landmass. Please adjust the longitude westward for more accurate predictions within California.")
        is_in_ca_land_for_prediction = False # Mark as outside
    elif (longitude < -124.0): # Too far west, likely ocean
        st.error("❌ The selected location appears to be in the Pacific Ocean. Please adjust the longitude eastward to select a location on California's landmass.")
        is_in_ca_land_for_prediction = False

    if is_in_ca_land_for_prediction:
        # Step 1: Form raw DataFrame
        input_data = pd.DataFrame({
            'longitude': [longitude], # Use the corrected longitude variable
            'latitude': [latitude],   # Use the corrected latitude variable
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
    else:
        st.warning("Prediction aborted: Please select a location truly within California's landmass.")

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
st.title("California Housing Location Viewer") # This title is fine as a separate section.
st.markdown("### 🎯 Chosen Coordinates") # Refers to the coordinates selected above

# --- Create DataFrame for Map ---
# Uses the 'latitude' and 'longitude' variables obtained from the *new* separate sliders
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
