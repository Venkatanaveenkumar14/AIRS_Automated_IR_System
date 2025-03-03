from joblib import load

def predict(model_path, X_input):
    """
    Loads the trained model and predicts on new data.
    """
    model = load(model_path)
    return model.predict(X_input)
