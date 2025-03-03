def trigger_response(prediction):
    """
    Triggers an appropriate response based on model prediction.
    """
    if prediction == 1:
        print("ALERT: Malicious activity detected! Initiating response.")
    else:
        print("No threat detected.")
