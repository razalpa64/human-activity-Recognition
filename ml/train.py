import os
import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from preprocessing import load_uci_data, generate_synthetic_data, clean_data, get_preprocessor, ACTIVITIES

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ml_dir = os.path.join(base_dir, "ml")
    os.makedirs(ml_dir, exist_ok=True)
    
    dataset_source = "real"
    try:
        X_train, y_train, X_test, y_test, feature_names = load_uci_data(base_dir)
    except Exception as e:
        print(f"Notice: {e}")
        print("Using synthetic UCI HAR dataset generation...")
        X_train, y_train, X_test, y_test, feature_names = generate_synthetic_data()
        dataset_source = "synthetic"
        
    X_train = clean_data(X_train)
    X_test = clean_data(X_test)
    
    scaler, X_train_scaled = get_preprocessor(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train_scaled, y_train)
    
    y_pred = clf.predict(X_test_scaled)
    accuracy = float(accuracy_score(y_test, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
    
    report = classification_report(y_test, y_pred, target_names=ACTIVITIES, output_dict=True, zero_division=0)
    
    metrics = {
        "dataset_source": dataset_source,
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "classification_report": report,
        "training_samples": int(X_train.shape[0]),
        "testing_samples": int(X_test.shape[0]),
        "features_count": int(X_train.shape[1]),
        "activities": ACTIVITIES
    }
    
    joblib.dump(clf, os.path.join(ml_dir, "model.pkl"))
    joblib.dump(scaler, os.path.join(ml_dir, "scaler.pkl"))
    with open(os.path.join(ml_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("Model trained and serialized successfully.")

if __name__ == "__main__":
    main()
