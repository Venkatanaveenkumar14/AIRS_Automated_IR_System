# dashboard/app.py
import streamlit as st
import requests
import pandas as pd
import time
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from joblib import load

# Fix Import Path Issue
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Now Import Required Modules
from ml_model.model_predictor import predict
from data_preprocessing.ddos_preprocessor import preprocess_ddos_dataset
from data_preprocessing.feature_engineering import extract_features  # FIXED: Import extract_features

# AlienVault OTX API Details
OTX_API_KEY = "7879a75c029b05f58bf9a68427795c0171a22fc67e3ec7e9f123176c9df90b4b"
OTX_INDICATORS_URL = "https://otx.alienvault.com/api/v1/pulses/subscribed"

# Load Trained Model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "gb_model_ddos.joblib")
model = load(MODEL_PATH)

# Function to Fetch Live Threat Feeds
def fetch_threat_intelligence():
    headers = {"X-OTX-API-KEY": OTX_API_KEY}
    response = requests.get(OTX_INDICATORS_URL, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        threats = []

        for pulse in data['results'][:5]:  # Limit to latest 5 threats
            threats.append({
                "Name": pulse["name"],
                "Created": pulse["created"],
                "Tags": ", ".join(pulse["tags"]),
                "References": pulse["references"][0] if pulse["references"] else "N/A"
            })

        return pd.DataFrame(threats)
    else:
        return pd.DataFrame(columns=["Name", "Created", "Tags", "References"])

# Function to Generate Threat Trends Over Time
def plot_threat_trends(threat_log):
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=threat_log, x="timestamp", y="threat_count", marker="o")
    plt.title("Live Attack Trends Over Time")
    plt.xlabel("Time")
    plt.ylabel("Number of Threats")
    st.pyplot(plt)

# Function to Simulate Real-Time Predictions
def simulate_predictions():
    """
    Simulates real-time threat detection by running predictions on sample data.
    Ensures feature consistency between training and prediction.
    """
    dataset_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "small_dataset.csv")
    sample_data = pd.read_csv(dataset_path).sample(5)

    # Apply the same preprocessing and feature extraction
    preprocessed_data, _ = preprocess_ddos_dataset(sample_data)
    feature_data = extract_features(preprocessed_data)  # Ensures consistent features

    # Ensure feature names match training data
    model_features = model.feature_names_in_  # Get the features model was trained on
    for col in model_features:
        if col not in feature_data.columns:
            feature_data[col] = 0  # Fill missing features with 0

    # Reorder features to match training order
    feature_data = feature_data[model_features]

    # Run the prediction
    predictions = predict(MODEL_PATH, feature_data)
    results = pd.DataFrame({"Timestamp": datetime.now(), "Prediction": predictions})
    return results

# Streamlit Dashboard UI
st.title("Automated Incident Response Dashboard")
st.markdown("### **🔍 Live Threat Intelligence from AlienVault OTX**")
threat_df = fetch_threat_intelligence()
st.dataframe(threat_df)

st.markdown("### **Real-Time Attack Trends**")
# Maintain a rolling attack log
if "threat_log" not in st.session_state:
    st.session_state["threat_log"] = pd.DataFrame(columns=["timestamp", "threat_count"])

# Simulate threat detections
new_detections = simulate_predictions()
new_entry = pd.DataFrame({"timestamp": [datetime.now()], "threat_count": [new_detections.shape[0]]})
st.session_state["threat_log"] = pd.concat([st.session_state["threat_log"], new_entry], ignore_index=True)
plot_threat_trends(st.session_state["threat_log"])

st.markdown("### **Recently Detected Threats**")
st.dataframe(new_detections)

# Auto-refresh every 30 seconds
time.sleep(30)
st.rerun()