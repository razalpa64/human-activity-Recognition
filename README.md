# Human Activity Recognition (HAR) Web Application

A production-grade, full-stack Human Activity Recognition web application built with Python, FastAPI, Scikit-Learn, and a cinematic warm neutral frontend. It utilizes the UCI Human Activity Recognition Using Smartphones dataset and a Random Forest machine learning classifier to classify human motion into 6 distinct activities: **WALKING**, **WALKING_UPSTAIRS**, **WALKING_DOWNSTAIRS**, **SITTING**, **STANDING**, and **LAYING**.

---

## 🐳 Containerized Deployment & Hosting

This project is fully prepared for cloud hosting (Render, Railway, Fly.io, AWS, or Docker) with a production-ready `Dockerfile` and `docker-compose.yml`.

### Build & Run with Docker:
```bash
docker-compose up --build
```
The app will be accessible at `http://localhost:8000`.

---

## 🚀 Local Installation & Setup

### 1. Create Virtual Environment
```bash
python -m venv .venv
```

### 2. Activate Virtual Environment
- **Windows:** `.venv\Scripts\activate`
- **macOS / Linux:** `source .venv/bin/activate`

### 3. Install Dependencies
```bash
pip install -r backend/requirements.txt
```

---

## 🧠 Training & Running

### 1. Train Model
```bash
python ml/train.py
```

The model is trained on the **raw UCI HAR Inertial Signals** (128-sample windows @ 50 Hz). The
feature pipeline faithfully reconstructs the official 561-dimensional UCI feature space:
median filter, 0.3 Hz Butterworth gravity/body separation, jerk derivatives, Euclidean
magnitudes, FFT, and every statistic enumerated in `features.txt` (mean, std, mad, max, min,
sma, energy, iqr, entropy, Burg arCoeff, correlation, maxInds, meanFreq, skewness, kurtosis,
bandsEnergy, angles). Live motion-sensor windows go through the **exact same pipeline** at
inference time, with phone units auto-normalized (m/s² to g, deg/s to rad/s).
Achieved test accuracy: **~92.9%**.

### 2. Start Backend & Static Frontend
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
Open `http://localhost:8000` in your browser.
# human-activity-Recognition
