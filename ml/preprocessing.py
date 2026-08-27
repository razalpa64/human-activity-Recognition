import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

ACTIVITIES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
]

def get_dataset_paths(base_dir=None):
    if base_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__,), ".."))
    
    dataset_dir = os.path.join(base_dir, "dataset", "UCI HAR Dataset")
    train_dir = os.path.join(dataset_dir, "train")
    test_dir = os.path.join(dataset_dir, "test")
    
    return {
        "dataset_dir": dataset_dir,
        "train_dir": train_dir,
        "test_dir": test_dir,
        "features_path": os.path.join(dataset_dir, "features.txt"),
        "activity_labels_path": os.path.join(dataset_dir, "activity_labels.txt"),
        "x_train_path": os.path.join(train_dir, "X_train.txt"),
        "y_train_path": os.path.join(train_dir, "y_train.txt"),
        "x_test_path": os.path.join(test_dir, "X_test.txt"),
        "y_test_path": os.path.join(test_dir, "y_test.txt"),
    }

def load_uci_data(base_dir=None):
    paths = get_dataset_paths(base_dir)
    
    # Check if files exist
    required_files = [
        paths["x_train_path"], paths["y_train_path"],
        paths["x_test_path"], paths["y_test_path"],
        paths["features_path"]
    ]
    
    for f in required_files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"UCI HAR dataset file not found: {f}. Please ensure the UCI HAR Dataset is placed correctly in dataset/UCI HAR Dataset/")

    print("Loading UCI HAR dataset from disk...")
    
    # Load features names
    features_df = pd.read_csv(paths["features_path"], sep=r'\s+', header=None, names=['index', 'feature_name'])
    feature_names = features_df['feature_name'].tolist()
    
    # Load data using numpy loadtxt or pandas for robustness against whitespace formatting
    X_train = np.loadtxt(paths["x_train_path"])
    y_train = np.loadtxt(paths["y_train_path"]).astype(int) - 1  # 1-6 -> 0-5
    
    X_test = np.loadtxt(paths["x_test_path"])
    y_test = np.loadtxt(paths["y_test_path"]).astype(int) - 1
    
    return X_train, y_train, X_test, y_test, feature_names

def generate_synthetic_data():
    """Generates synthetic UCI HAR-like data (7352 train samples, 2947 test samples, 561 features)
    when the real dataset is not yet present, enabling immediate testing and app verification."""
    print("Generating synthetic UCI HAR dataset for demonstration/fallback mode...")
    np.random.seed(42)
    n_train = 7352
    n_test = 2947
    n_features = 561
    n_classes = 6
    
    X_train = np.random.randn(n_train, n_features) * 2.0 + 0.5
    y_train = np.random.randint(0, n_classes, size=n_train)
    
    X_test = np.random.randn(n_test, n_features) * 2.0 + 0.5
    y_test = np.random.randint(0, n_classes, size=n_test)
    
    feature_names = [f"tBodyAcc-mean()-{i}" for i in range(n_features)]
    return X_train, y_train, X_test, y_test, feature_names

def clean_data(X):
    """Checks for NaN or infinity and cleans them."""
    if np.isnan(X).any() or np.isinf(X).any():
        print("Warning: NaN or Inf detected in data. Cleaning...")
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
    return X

def get_preprocessor(X_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    return scaler, X_train_scaled
