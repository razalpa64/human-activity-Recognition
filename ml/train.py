import os
import json

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    classification_report,
)

try:  # supports both `python ml/train.py` and `python -m ml.train`
    from ml.preprocessing import ACTIVITIES, clean_data, get_preprocessor
    from ml.feature_extraction import (
        build_dataset_features,
        extract_features_batch,
        generate_synthetic_windows,
    )
except ImportError:  # pragma: no cover
    from preprocessing import ACTIVITIES, clean_data, get_preprocessor
    from feature_extraction import (
        build_dataset_features,
        extract_features_batch,
        generate_synthetic_windows,
    )


def _features_from_windows(windows):
    return extract_features_batch(*[windows[:, i, :] for i in range(windows.shape[1])])


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ml_dir = os.path.join(base_dir, "ml")
    os.makedirs(ml_dir, exist_ok=True)

    dataset_source = "real"
    try:
        print("Extracting UCI HAR features from the raw Inertial Signals...")
        X_train, y_train = build_dataset_features(base_dir, "train")
        X_test, y_test = build_dataset_features(base_dir, "test")
        print(f"Real dataset features: train={X_train.shape}, test={X_test.shape}")
    except Exception as e:
        print(f"Notice: {e}")
        print("Falling back to synthetic activity-coherent window generation...")
        dataset_source = "synthetic"
        w_train, y_train = generate_synthetic_windows(7352, seed=42)
        w_test, y_test = generate_synthetic_windows(2947, seed=7)
        X_train = _features_from_windows(w_train)
        X_test = _features_from_windows(w_test)

    # Per-class feature profiles (used by the descriptive demo endpoint).
    # Computed from real extracted features when the dataset is present.
    profiles_mean = np.stack([X_train[y_train == c].mean(axis=0) for c in range(len(ACTIVITIES))])
    profiles_std = np.stack([X_train[y_train == c].std(axis=0) for c in range(len(ACTIVITIES))])

    X_train = clean_data(X_train)
    X_test = clean_data(X_test)

    scaler, X_train_scaled = get_preprocessor(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training Random Forest Classifier (300 trees)...")
    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    accuracy = float(accuracy_score(y_test, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    report = classification_report(y_test, y_pred, target_names=ACTIVITIES,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "dataset_source": dataset_source,
        "feature_pipeline": "UCI raw Inertial Signals -> median filter -> Butterworth "
                            "gravity/body split -> jerk -> magnitudes -> FFT -> 561 features",
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "classification_report": report,
        "confusion_matrix": cm,
        "activities": ACTIVITIES,
        "training_samples": int(X_train.shape[0]),
        "testing_samples": int(X_test.shape[0]),
        "features_count": int(X_train.shape[1]),
    }

    joblib.dump(clf, os.path.join(ml_dir, "model.pkl"))
    joblib.dump(scaler, os.path.join(ml_dir, "scaler.pkl"))
    np.savez(os.path.join(ml_dir, "class_profiles.npz"), mean=profiles_mean, std=profiles_std)
    with open(os.path.join(ml_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Done. Test accuracy: {accuracy:.4f} ({dataset_source} data)")
    print("Model, scaler, class profiles and metrics saved to ml/.")


if __name__ == "__main__":
    main()
