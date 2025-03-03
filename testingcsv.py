from joblib import load

# Load the trained model
model = load("models/gb_model_ddos.joblib")

# Print the feature names the model was trained on
print(model.feature_names_in_)