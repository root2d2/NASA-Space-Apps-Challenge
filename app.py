# app.py - Streamlit Prediction UI (CORRECTED FINAL 14-FEATURE CONFIGURATION)
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

# --- Configuration and Initialization ---
st.set_page_config(page_title='Exoplanet Classifier', layout='wide')

MODEL_PATH = 'models/best_model.joblib'

# The standard, correct list of 14 features from the Kepler cumulative data.
FEATURE_NAMES = [
    'teff', 'teff_err1', 'teff_err2', 
    'logg', 'logg_err1', 'logg_err2', 
    'feh', 
    'koi_period', 'koi_duration', 'koi_impact', 
    'koi_depth', 'koi_prad', 'koi_teq', 
    'koi_model_snr'
]

# Estimated ranges and default values for input widgets
FEATURE_RANGES = {
    # Stellar Features
    'teff': (2000, 10000, 5000), 'teff_err1': (-500, 500, 100), 'teff_err2': (-500, 500, -100),
    'logg': (0, 5, 4.0), 'logg_err1': (-2, 2, 0.5), 'logg_err2': (-2, 2, -0.5),
    'feh': (-3.0, 1.0, 0.0), # Metallicity
    
    # Planet Candidate Features (KOI)
    'koi_period': (0.1, 1000.0, 10.0),      # Orbital Period (days)
    'koi_duration': (0.1, 10.0, 3.0),       # Transit Duration (hours)
    'koi_impact': (0.0, 2.0, 0.5),          # Impact Parameter
    'koi_depth': (10.0, 100000.0, 1000.0),  # Transit Depth (ppm)
    'koi_prad': (0.1, 50.0, 5.0),           # Planet Radius (Earth radii)
    'koi_teq': (200.0, 5000.0, 1000.0),     # Equilibrium Temperature (K)
    'koi_model_snr': (1.0, 1000.0, 50.0)    # Model SNR
}

st.title('🚀 Exoplanet Identifier Tool')
st.markdown('''
    **The final, stable configuration!** This application is now configured for the **14 standard features** to match the newly trained, clean model file. Enter parameters and predict below.
    ''')

# --- Model Loading (Essential for Prediction) ---
@st.cache_resource
def load_trained_model(path):
    """Loads the trained model from the joblib file."""
    if not os.path.exists(path):
        st.error(f"Error: Trained model file not found at {path}. Please ensure 'python train.py' completed successfully.")
        return None
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

clf = load_trained_model(MODEL_PATH)
if clf is None:
    st.stop()


# --- Input Fields (The core user interaction) ---
st.header('Stellar and Planet Candidate Measurements')
st.markdown('***Enter the parameters below for the system you wish to classify:***')

user_input = {}
cols = st.columns(3) 

for i, feature in enumerate(FEATURE_NAMES):
    col = cols[i % 3] 
    
    min_val, max_val, default_val = FEATURE_RANGES.get(feature, (0.0, 100.0, 1.0))
    
    user_input[feature] = col.number_input(
        f'{feature.upper().replace("_", " ")}', 
        min_value=float(min_val), 
        max_value=float(max_val), 
        value=float(default_val), 
        step=0.01,
        format='%.4f',
        help=f"Input for the {feature} feature. Default value is typical for Kepler data."
    )

# --- Prediction Logic ---
st.markdown('---')
st.header('Prediction Result')

# Convert user input into a DataFrame row ready for the model
input_data = pd.DataFrame([user_input], columns=FEATURE_NAMES) 

# Button to trigger prediction
if st.button('Predict Exoplanet Status', type='primary'):
    try:
        # Pass all 14 required columns in the correct order
        X_predict = input_data[FEATURE_NAMES]
        
        # Get probability for the positive class (1, which is Exoplanet)
        prediction_proba = clf.predict_proba(X_predict)[0][1] * 100 
        prediction = clf.predict(X_predict)[0]

        st.subheader('Classification:')

        if prediction == 1:
            st.success(f"### ⭐ LIKELY EXOPLANET CANDIDATE")
            st.metric("Confidence Score (Positive Class)", f"{prediction_proba:.2f}%")
            st.balloons()
        else:
            st.warning(f"### 🔴 LIKELY FALSE POSITIVE")
            st.metric("Confidence Score (Negative Class)", f"{100 - prediction_proba:.2f}%")

    except Exception as e:
        st.error(f"Prediction failed. An internal error occurred. (Feature count mismatch is likely)")
        # Display the specific error for better debugging
        st.code(e) 

# --- Footer and Information ---
st.sidebar.markdown('---')
st.sidebar.info('This app is configured for the standard 14 features. Run **Step 2** to generate the matching model file.')
st.sidebar.markdown(f'**Model File Used:** `{MODEL_PATH}`')
st.sidebar.markdown('**Goal:** Ensure the features listed here match the features printed by `train.py`.')

