# main.py
import pandas as pd
from data_preprocessing.ddos_preprocessor import preprocess_ddos_dataset
from data_preprocessing.feature_engineering import extract_features
from ml_model.train_model import train_model
from ml_model.model_predictor import predict
from incident_response.response_engine import trigger_response

def main():
    """
    Main pipeline execution for the Cybersecurity Incident Response ML system.
    """
    # Load and preprocess dataset with corrected CSV reading
    df = pd.read_csv("dataset/ready_dataset.csv", dtype=str, sep=',')
    
    # Preprocess dataset
    df, label_encoder = preprocess_ddos_dataset(df)
    df = extract_features(df)
    
    # Define features and labels
    X = df.drop(columns=['label'])
    y = df['label']

    print("Unique values in y after encoding:", y.unique())   # Debugging step

    # Train model
    model = train_model(X, y)
    
    # Simulate prediction and response
    sample_input = X.iloc[[0]]
    prediction = predict('models/gb_model_ddos.joblib', sample_input)
    trigger_response(prediction)

if __name__ == "__main__":
    main()