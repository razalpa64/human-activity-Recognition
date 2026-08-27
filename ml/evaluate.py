import os
import json
import joblib
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from preprocessing import load_uci_data, generate_synthetic_data, clean_data, ACTIVITIES

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ml_dir = os.path.join(base_dir, "ml")
    
    model_path = os.path.join(ml_dir, "model.pkl")
    scaler_path = os.path.join(ml_dir, "scaler.pkl")
    metrics_path = os.path.join(ml_dir, "metrics.json")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("Model or scaler artifacts not found. Please run python ml/train.py first.")
        return
        
    clf = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    try:
        _, _, X_test, y_test, _ = load_uci_data(base_dir)
    except Exception:
        _, _, X_test, y_test, _ = generate_synthetic_data()
        
    X_test = clean_data(X_test)
    X_test_scaled = scaler.transform(X_test)
    
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    report_dict = classification_report(y_test, y_pred, target_names=ACTIVITIES, output_dict=True, zero_division=0)
    
    print("==================================================")
    print(" HUMAN ACTIVITY RECOGNITION MODEL")
    print("==================================================")
    print(f"\nAccuracy: {acc * 100:.2f}%\n")
    print(f"{'Activity':<22} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 58)
    
    for act in ACTIVITIES:
        metrics = report_dict.get(act, {})
        p = metrics.get('precision', 0.0)
        r = metrics.get('recall', 0.0)
        f1 = metrics.get('f1-score', 0.0)
        print(f"{act:<22} {p:<12.4f} {r:<12.4f} {f1:<12.4f}")
        
    print("\n==================================================")
    
    # Also save or update metrics.json if needed
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            data = json.load(f)
        data["evaluated_accuracy"] = float(acc)
        with open(metrics_path, "w") as f:
            json.dump(data, f, indent=4)

if __name__ == "__main__":
    main()
