from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump
from tqdm import tqdm

class ProgressBoostingClassifier(GradientBoostingClassifier):
    """
    Custom Gradient Boosting Classifier that shows a live training progress bar.
    """
    def fit(self, X, y):
        n_estimators = self.n_estimators
        pbar = tqdm(total=n_estimators, desc="Training Progress", unit="tree")

        def monitor(stage, estimator, local_vars):
            """ Progress update function for each boosting stage. """
            pbar.update(1)

        result = super().fit(X, y, monitor=monitor)  # Pass the correctly formatted function

        pbar.close()
        return result

def train_model(X, y):
    """
    Trains a Gradient Boosting model on the dataset with a live progress bar.
    """
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print("Training model (this may take some time)...")
    model = ProgressBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3)

    model.fit(X_train, y_train)  # Training with progress bar

    print("\nModel training complete. Running evaluation...")
    y_pred = model.predict(X_test)

    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    dump(model, 'models/gb_model_ddos.joblib')
    return model