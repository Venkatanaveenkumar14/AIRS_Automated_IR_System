from sklearn.ensemble import IsolationForest

def detect_anomalies(X_train, X_test):
    """
    Uses Isolation Forest to detect anomalies in network traffic.
    """
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X_train)
    anomalies = model.predict(X_test)
    return anomalies
