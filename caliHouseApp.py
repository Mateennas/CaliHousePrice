import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Geographic Map Viewer", layout="centered")
st.title("California Housing Location Viewer")

st.markdown("### 🎯 Choose Coordinates")

# --- User Inputs ---
longitude = st.slider("Longitude", -124.35, -114.31, -122.0, step=0.01)
latitude = st.slider("Latitude", 32.54, 41.95, 37.0, step=0.01)
if not (32.54 <= latitude <= 42.01 and -124.48 <= longitude <= -114.13):
    st.error("❌ Coordinates are outside California. Please select a valid location within California.")
    st.stop()  # Stop the app from running further

# --- Create DataFrame for Map ---
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
