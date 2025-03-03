import sys
import os
import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from data_preprocessing.ddos_preprocessor import preprocess_ddos_dataset
from data_preprocessing.feature_engineering import extract_features

# Add project root to system path (fix ModuleNotFoundError)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
'''
def evaluate_trained_model():
    """
    Loads the trained model and evaluates it on the test dataset.
    """
    print("\nStep 1: Loading trained model...")
    model_path = "models/gb_model_ddos.joblib"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"ERROR: Model file {model_path} not found! Did you train it?")
        return
    
    # Load model
    model = load(model_path)
    print("Model loaded successfully!\n")

    print("Step 2: Loading dataset...")
    dataset_path = "dataset/ready_dataset.csv"
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset file {dataset_path} not found! Ensure it exists.")
        return

    df = pd.read_csv(dataset_path, dtype=str, sep=',')
    print(f"Dataset loaded! Shape: {df.shape}\n")

    print("⚙️ Step 3: Preprocessing dataset...")
    df, label_encoder = preprocess_ddos_dataset(df)
    print(f"Preprocessing done! Shape: {df.shape}\n")

    print("Step 4: Extracting features...")
    df = extract_features(df)
    print(f"Feature extraction completed! Shape: {df.shape}\n")

    # Define features and labels
    print("Step 5: Defining features and labels...")
    if 'label' not in df.columns:
        print("ERROR: 'label' column missing after preprocessing!")
        return
    
    X = df.drop(columns=['label'])
    y = df['label']
    print(f"Features shape: {X.shape}, Labels shape: {y.shape}\n")

    # Split into training and testing sets
    print("Step 6: Splitting dataset into train & test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"Training size: {X_train.shape}, Testing size: {X_test.shape}\n")

    # Run model evaluation
    print("Step 7: Running model evaluation...")
    y_pred = model.predict(X_test)
    print("Prediction completed!\n")

    # Display evaluation results
    print("\n Model Performance Summary")
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    print("Classification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    evaluate_trained_model()
    '''



# ml_model/evaluate_model.py (Updated)

def evaluate_trained_model():
    """
    Loads the trained model and evaluates it on the test dataset.
    """
    print("\nStarting Model Evaluation...\n")

    print("Step 1: Loading trained model...")
    model_path = "models/gb_model_ddos.joblib"

    if not os.path.exists(model_path):
        print(f"ERROR: Model file {model_path} not found! Did you train it?")
        return

    model = load(model_path)
    print("Model loaded successfully!\n")

    print("Step 2: Loading dataset...")
    dataset_path = "dataset/ready_dataset.csv"

    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset file {dataset_path} not found! Ensure it exists.")
        return

    df = pd.read_csv(dataset_path, sep=',', dtype=str)
    print(f"Dataset loaded! Shape: {df.shape}\n")

    print("⚙️ Step 3: Preprocessing dataset...")
    df, label_encoder = preprocess_ddos_dataset(df)

    # Ensure 'label' exists after preprocessing
    if 'label' not in df.columns:
        print("ERROR: 'label' column missing after preprocessing! Debug required.")
        print("Columns available after preprocessing:", df.columns.tolist())
        return

    print(f"Preprocessing done! Shape: {df.shape}\n")

    print("Step 4: Extracting features...")
    df = extract_features(df)

    # Debug: Ensure label exists after feature extraction
    print("Columns available after feature extraction:", df.columns.tolist())

    if 'label' not in df.columns:
        print("ERROR: 'label' column missing after feature extraction! Debug required.")
        return

    print(f"Feature extraction completed! Shape: {df.shape}\n")

    print("Step 5: Defining features and labels...")
    try:
        X = df.drop(columns=['label'])
        y = df['label']
    except KeyError:
        print("ERROR: 'label' column not found in dataset after feature extraction.")
        print("Available columns:", df.columns.tolist())
        return

    print(f"Features shape: {X.shape}, Labels shape: {y.shape}\n")

    print("Step 6: Splitting dataset into train & test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"Training size: {X_train.shape}, Testing size: {X_test.shape}\n")

    print("Step 7: Running model evaluation...")
    y_pred = model.predict(X_test)
    print("Prediction completed!\n")

    print("\nModel Performance Summary")
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    print("Classification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    print("Starting Model Evaluation...")
    evaluate_trained_model()