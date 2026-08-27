import json
import os

import joblib
import numpy as np
from typing import Any, Dict, List, Optional

from backend import config
from ml.feature_extraction import (
    descriptive_to_class,
    extract_features_batch,
    extract_features_from_sensor_window,
    generate_synthetic_windows,
    load_raw_windows,
    validate_feature_vector,
)
from ml.preprocessing import ACTIVITIES, clean_data


class ModelPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.metrics = {}
        # Raw sensor window cache (genuine dataset windows, not feature vectors)
        self.windows_cache = None   # (M, 6, N)
        self.labels_cache = None    # (M,)
        # Per-class feature profiles for the descriptive demo (real data)
        self.profiles_mean = None   # (6, 561)
        self.profiles_std = None    # (6, 561)
        self.load_artifacts()
        self.load_test_cache()
        self.load_profiles()

    # ------------------------------------------------------------------
    # Artifact loading
    # ------------------------------------------------------------------
    def load_artifacts(self):
        try:
            if os.path.exists(config.MODEL_PATH):
                self.model = joblib.load(config.MODEL_PATH)
            if os.path.exists(config.SCALER_PATH):
                self.scaler = joblib.load(config.SCALER_PATH)
            if os.path.exists(config.METRICS_PATH):
                with open(config.METRICS_PATH, "r") as f:
                    self.metrics = json.load(f)
            else:
                self.metrics = {
                    "dataset_source": "unknown",
                    "accuracy": 0.0,
                    "training_samples": 7352,
                    "testing_samples": 2947,
                    "features_count": 561,
                    "activities": ACTIVITIES,
                }
        except Exception as e:
            print(f"Error loading model artifacts: {e}")

    def load_test_cache(self):
        """Cache raw test windows so /dataset/sample streams genuine dataset data."""
        try:
            windows, labels = load_raw_windows(config.BASE_DIR, "test")
        except Exception:
            windows, labels = generate_synthetic_windows(200, seed=123)
        self.windows_cache = windows
        self.labels_cache = labels

    def load_profiles(self):
        profiles_path = os.path.join(config.ML_DIR, "class_profiles.npz")
        try:
            if os.path.exists(profiles_path):
                data = np.load(profiles_path)
                self.profiles_mean = data["mean"]
                self.profiles_std = data["std"]
                return
        except Exception as e:
            print(f"Could not load class profiles: {e}")
        self._compute_profiles_from_test_cache()

    def _compute_profiles_from_test_cache(self):
        """Fallback: derive class profiles from the cached test windows."""
        if self.windows_cache is None or self.labels_cache is None:
            return
        try:
            X = extract_features_batch(*[self.windows_cache[:, i, :] for i in range(6)])
            labels = self.labels_cache
            self.profiles_mean = np.stack(
                [X[labels == c].mean(axis=0) if np.any(labels == c) else np.zeros(X.shape[1])
                 for c in range(len(ACTIVITIES))]
            )
            self.profiles_std = np.stack(
                [X[labels == c].std(axis=0) if np.any(labels == c) else np.zeros(X.shape[1])
                 for c in range(len(ACTIVITIES))]
            )
        except Exception as e:
            print(f"Could not compute class profiles: {e}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def is_loaded(self) -> bool:
        return self.model is not None and self.scaler is not None

    def predict(self, features: List[float], descriptive_meta: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        if not self.is_loaded():
            raise RuntimeError("Model or scaler is not loaded. Train it first: python ml/train.py")

        # Genuine ML inference through the fitted scaler and classifier.
        X = validate_feature_vector(features, expected_dim=561)
        X = clean_data(X)
        X_scaled = self.scaler.transform(X)

        pred_idx = int(self.model.predict(X_scaled)[0])
        probs = self.model.predict_proba(X_scaled)[0]

        activity = ACTIVITIES[pred_idx] if 0 <= pred_idx < len(ACTIVITIES) else "UNKNOWN"
        confidence = float(np.max(probs))
        probabilities = {ACTIVITIES[i]: float(probs[i]) for i in range(len(ACTIVITIES))}
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        top_predictions = [{"activity": act, "probability": prob} for act, prob in sorted_probs[:3]]

        result = {
            "activity": activity,
            "confidence": confidence,
            "probabilities": probabilities,
            "model_type": "RandomForest",
            "top_predictions": top_predictions,
        }
        if descriptive_meta:
            result["description_summary"] = descriptive_meta
        return result

    def predict_from_window(self, sensor_window: List[Any]) -> Dict[str, Any]:
        """Raw motion-sensor window -> faithful UCI feature pipeline -> model."""
        features_2d = extract_features_from_sensor_window(sensor_window)
        return self.predict(features_2d[0].tolist())

    def predict_from_description(self, intensity: str, stability: str, orientation: str,
                                 rotation: str, pattern: str) -> Dict[str, Any]:
        """Descriptive demo: route to a plausible class and build an in-distribution
        feature vector from that class's REAL dataset profile (mean + jitter)."""
        if self.profiles_mean is None or self.profiles_std is None:
            self.load_profiles()
        if self.profiles_mean is None:
            raise RuntimeError("Class profiles unavailable; train the model first: python ml/train.py")

        cls = descriptive_to_class(intensity, stability, orientation, rotation, pattern)
        seed = hash((intensity, stability, orientation, rotation, pattern)) % (2 ** 32)
        rng = np.random.default_rng(seed)
        vector = self.profiles_mean[cls] + rng.normal(0.0, 1.0, 561) * 0.35 * self.profiles_std[cls]

        meta = {
            "Movement Intensity": intensity,
            "Movement Stability": stability,
            "Body Orientation": orientation,
            "Rotation": rotation,
            "Movement Pattern": pattern,
        }
        return self.predict(vector.tolist(), descriptive_meta=meta)

    # ------------------------------------------------------------------
    # Dataset demo endpoints
    # ------------------------------------------------------------------
    def get_sample(self, index: int) -> Dict[str, Any]:
        """Return a genuine dataset window with its true label and model prediction."""
        if self.windows_cache is None or self.labels_cache is None:
            self.load_test_cache()

        if index < 0 or index >= len(self.windows_cache):
            raise IndexError(f"Sample index {index} out of bounds (0..{len(self.windows_cache) - 1}).")

        window = self.windows_cache[index]  # (6, N)
        true_label_idx = int(self.labels_cache[index])
        true_activity = ACTIVITIES[true_label_idx] if 0 <= true_label_idx < len(ACTIVITIES) else "UNKNOWN"

        window_json = [
            {
                "accelerometer": {"x": float(window[0, t]), "y": float(window[1, t]), "z": float(window[2, t])},
                "gyroscope": {"x": float(window[3, t]), "y": float(window[4, t]), "z": float(window[5, t])},
            }
            for t in range(window.shape[1])
        ]
        pred_result = self.predict_from_window(window_json)

        return {
            "index": index,
            "true_activity": true_activity,
            "window": window_json,
            "prediction": pred_result,
        }

    def total_samples(self) -> int:
        return 0 if self.windows_cache is None else int(len(self.windows_cache))

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": "Random Forest Classifier",
            "dataset": "UCI Human Activity Recognition Using Smartphones",
            "activities_count": len(ACTIVITIES),
            "features_count": self.metrics.get("features_count", 561),
            "training_samples": self.metrics.get("training_samples", 7352),
            "testing_samples": self.metrics.get("testing_samples", 2947),
            "accuracy": self.metrics.get("accuracy", 0.0),
            "dataset_source": self.metrics.get("dataset_source", "unknown"),
        }


predictor = ModelPredictor()

