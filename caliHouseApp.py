import streamlit as st
import joblib
import pandas as pd
import numpy as np

model = joblib.load('best_house_price_model.pkl')


try:
    model = joblib.load('cali_house_model.joblib')
    scaling_params = joblib.load('cali_scaling_params.joblib')
    st.sidebar.success("Model and scaling parameters loaded successfully!")
except FileNotFoundError:
    st.error("Error: Model files (cali_house_model.joblib or cali_scaling_params.joblib) not found. "
             "Please ensure they are in the same directory as this app.")
    st.stop() # Stop the app if essential files are missing

# --- 2. Streamlit App Title and Description ---
st.title('California Housing Price Prediction')
st.markdown("""
    Predict the median house value for a California block group based on its characteristics.
    Enter the details below to get an estimated price.
""")

# --- 3. Define Input Options and Ranges ---
# These ranges and options are based on typical values found in the California Housing dataset.
# They ensure that user inputs are within a reasonable and expected range for the model.

ocean_proximity_options = ['<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND']

# Approximate min/max values from the California housing dataset for sliders
# These should ideally be derived from your training data's min/max for each feature.
longitude_range = (-124.35, -114.31)
latitude_range = (32.54, 41.95)
housing_median_age_range = (1, 52)
total_rooms_range = (2, 39320) # These are block-group totals, not per house
total_bedrooms_range = (1, 6445)
population_range = (3, 35682)
households_range = (1, 6082)
median_income_range = (0.4999, 15.0001) # Capped at 15.0001 in original dataset

# --- 4. User Input Widgets ---
# Streamlit widgets allow users to interactively provide input for prediction.
st.header("Enter Block Group Characteristics:")

col1, col2 = st.columns(2)
with col1:
    longitude = st.slider("Longitude", float(longitude_range[0]), float(longitude_range[1]), -122.0)
    latitude = st.slider("Latitude", float(latitude_range[0]), float(latitude_range[1]), 37.0)
    housing_median_age = st.slider("Housing Median Age", int(housing_median_age_range[0]), int(housing_median_age_range[1]), 30)
    median_income = st.slider("Median Income (in tens of thousands)", float(median_income_range[0]), float(median_income_range[1]), 5.0, step=0.1)

with col2:
    total_rooms = st.slider("Total Rooms (per block group)", int(total_rooms_range[0]), int(total_rooms_range[1]), 2000)
    total_bedrooms = st.slider("Total Bedrooms (per block group)", int(total_bedrooms_range[0]), int(total_bedrooms_range[1]), 400)
    population = st.slider("Population (per block group)", int(population_range[0]), int(population_range[1]), 1000)
    households = st.slider("Households (per block group)", int(households_range[0]), int(households_range[1]), 350)
    ocean_proximity = st.selectbox("Ocean Proximity", ocean_proximity_options)


# --- 5. Prediction Button and Logic ---
if st.button("Predict Median House Value"):
    # Create a DataFrame from user inputs.
    # Ensure column names match the original training features before engineering/encoding.
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

    # --- Feature Engineering (must match training pipeline) ---
    # Create the same engineered features as done during model training.
    input_df_raw['rooms_per_household'] = input_df_raw['total_rooms'] / input_df_raw['households']
    input_df_raw['bedrooms_per_room'] = input_df_raw['total_bedrooms'] / input_df_raw['total_rooms']
    input_df_raw['population_per_household'] = input_df_raw['population'] / input_df_raw['households']

    # --- One-Hot Encoding (must match training pipeline) ---
    # Convert 'ocean_proximity' categorical feature into numerical dummy variables.
    # The 'prefix' argument helps create clear column names like 'ocean_proximity_<category>'.
    df_encoded_input = pd.get_dummies(input_df_raw, columns=['ocean_proximity'], prefix='ocean_proximity')

    # --- Align Columns (CRITICAL for consistent prediction) ---
    # The model expects a specific set of features in a specific order.
    # `model.feature_names_in_` provides the exact column names and order from training.
    # `reindex` adds missing columns (e.g., other ocean_proximity categories not selected)
    # with a fill_value of 0 and reorders existing columns.
    final_features_df = df_encoded_input.reindex(columns=model.feature_names_in_, fill_value=0)

    # --- Scaling Numerical Features (must match training pipeline) ---
    # Identify numerical features that need scaling.
    # Exclude one-hot encoded columns and any non-numerical columns.
    numerical_cols_for_scaling = [col for col in final_features_df.columns if col in scaling_params]

    # Apply scaling using the loaded mean and std for each numerical feature.
    # This ensures the input data is transformed identically to the training data.
    scaled_input_data = final_features_df.copy()
    for col in numerical_cols_for_scaling:
        mean_val = scaling_params[col]['mean']
        std_val = scaling_params[col]['std']
        if std_val > 0: # Avoid division by zero for constant features
            scaled_input_data[col] = (scaled_input_data[col] - mean_val) / std_val
        else:
            scaled_input_data[col] = 0 # Or handle as appropriate for constant features

    # --- Make Prediction ---
    # Use the prepared and scaled input data to get a prediction from the model.
    predicted_value = model.predict(scaled_input_data)[0]

    # --- Display Result ---
    # Format the predicted value as currency and display it to the user.
    st.success(f"Predicted Median House Value: ${predicted_value:,.2f}")

# --- 6. Custom Styling (Optional) ---
# This section applies custom CSS to the Streamlit app for aesthetic purposes.
# It sets a background image and adjusts text colors for readability.
st.markdown(
    f"""
    <style>
    .stApp {{
        background: url("https://static.vecteezy.com/system/resources/thumbnails/026/185/327/small_2x/waterfall-and-stone-copy-space-blurred-background-ai-generated-photo.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .stApp::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(255, 255, 255, 0.7); /* Light overlay for readability */
        z-index: -1;
    }}
    h1, h2, h3, h4, h5, h6, label, .stMarkdown, .stSelectbox > label, .stSlider > label, .stButton > button {{
        color: #333333; /* Darker text for contrast */
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.5); /* Subtle shadow */
    }}
    .stButton > button {{
        background-color: #4CAF50; /* Green button */
        color: white;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        font-size: 16px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
    }}
    .stButton > button:hover {{
        background-color: #45a049;
        box-shadow: 3px 3px 8px rgba(0,0,0,0.3);
    }}
    </style>
    """
)
