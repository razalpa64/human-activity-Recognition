"""
Faithful UCI HAR feature pipeline.

Reconstructs the 561-dimensional UCI HAR feature space directly from RAW
motion-sensor windows (accelerometer + gyroscope), following the official
dataset pipeline (see dataset/UCI HAR Dataset/features_info.txt):

    raw accelerometer (total acceleration incl. gravity) + gyroscope @ 50 Hz
      -> width-3 median filter (noise removal)
      -> gravity / body separation (0.3 Hz Butterworth low-pass)
      -> jerk signals (time derivative of body accel & angular velocity)
      -> Euclidean magnitudes
      -> FFT (frequency domain, 64 bins for a 128-sample window)
      -> 561 features enumerated in dataset/UCI HAR Dataset/features.txt

The feature layout is parsed from features.txt at runtime, so the extractor,
the training script and the backend always share the exact same semantics.

Units: accelerometer input is expected in g (UCI convention). Phone data in
m/s^2 is auto-detected and converted. Gyroscope input is rad/s (deg/s is
auto-detected and converted).
"""

import os
import threading

import numpy as np

FS = 50.0                     # UCI sampling rate (Hz)
GRAVITY = 9.80665             # m/s^2 per g
FFT_BINS_128 = 64             # 128-point FFT -> 64 positive-frequency bins
AR_ORDER = 4                  # Burg autoregression order (per features_info.txt)
GRAVITY_CUTOFF_HZ = 0.3       # Butterworth low-pass corner for gravity
MIN_WINDOW = 64               # smallest raw window we accept
RECOMMENDED_WINDOW = 128      # UCI window size (2.56 s @ 50 Hz)

_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(_DIR, ".."))
FEATURES_PATH = os.path.join(BASE_DIR, "dataset", "UCI HAR Dataset", "features.txt")

_cache_lock = threading.Lock()
_feature_names_cache = None
_freq_bands_64 = [  # 14 bands per axis (1-based, inclusive, out of 64 bins)
    (1, 8), (9, 16), (17, 24), (25, 32), (33, 40), (41, 48), (49, 56), (57, 64),
    (1, 16), (17, 32), (33, 48), (49, 64), (1, 24), (25, 48),
]

_TIME_XYZ_GROUPS = ["tBodyAcc", "tGravityAcc", "tBodyAccJerk", "tBodyGyro", "tBodyGyroJerk"]
_TIME_MAG_GROUPS = [
    "tBodyAccMag", "tGravityAccMag", "tBodyAccJerkMag", "tBodyGyroMag", "tBodyGyroJerkMag",
]
_FREQ_XYZ_GROUPS = ["fBodyAcc", "fBodyAccJerk", "fBodyGyro"]
_FREQ_MAG_GROUPS = ["fBodyAccMag", "fBodyBodyAccJerkMag", "fBodyBodyGyroMag", "fBodyBodyGyroJerkMag"]
_FREQ_TO_TIME = {
    "fBodyAcc": "tBodyAcc",
    "fBodyAccJerk": "tBodyAccJerk",
    "fBodyGyro": "tBodyGyro",
    "fBodyAccMag": "tBodyAccMag",
    "fBodyBodyAccJerkMag": "tBodyAccJerkMag",
    "fBodyBodyGyroMag": "tBodyGyroMag",
    "fBodyBodyGyroJerkMag": "tBodyGyroJerkMag",
}

_GROUP_PREFIXES = sorted(
    _TIME_XYZ_GROUPS + _TIME_MAG_GROUPS + _FREQ_XYZ_GROUPS + _FREQ_MAG_GROUPS,
    key=len, reverse=True,
)

_AXIS_IDX = {"X": 0, "Y": 1, "Z": 2}


def load_feature_names(expected=561, features_path=None):
    """Parse dataset features.txt -> ordered list of the 561 feature names."""
    global _feature_names_cache
    with _cache_lock:
        if _feature_names_cache is not None and features_path is None:
            return list(_feature_names_cache)
        path = features_path or FEATURES_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"features.txt not found at {path}. The UCI HAR Dataset must be present."
            )
        names = []
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 2:
                    names.append(parts[1])
        if len(names) != expected:
            raise ValueError(f"Expected {expected} feature names, parsed {len(names)} from {path}")
        if features_path is None:
            _feature_names_cache = list(names)
        return names


def parse_feature_name(name):
    """Parse one features.txt entry into (group, func, axis, args).

    Examples:
      'tBodyAcc-mean()-X'          -> ('tBodyAcc', 'mean', 0, None)
      'tBodyAcc-arCoeff()-X,2'     -> ('tBodyAcc', 'arCoeff', 0, 2)
      'tBodyAcc-correlation()-X,Y' -> ('tBodyAcc', 'correlation', None, (0, 1))
      'fBodyAcc-bandsEnergy()-1,8' -> ('fBodyAcc', 'bandsEnergy', 0, (1, 8))
      'fBodyAcc-maxInds-X'         -> ('fBodyAcc', 'maxInds', 0, None)
      'tBodyAccMag-sma()'          -> ('tBodyAccMag', 'sma', None, None)
      'angle(X,gravityMean)'       -> ('angle', ('X', 'gravityMean'), None, None)
    """
    if name.startswith("angle("):
        inner = name[len("angle("):-1]
        left, _, right = inner.partition(",")
        left = left.strip().rstrip(")")
        right = right.strip()
        return ("angle", (left, right), None, None)

    group = next((p for p in _GROUP_PREFIXES if name.startswith(p + "-")), None)
    if group is None:
        raise ValueError(f"Cannot parse feature name: {name}")
    rest = name[len(group) + 1:]

    if "(" in rest:
        func = rest.split("(", 1)[0]
        inner = rest.split("(", 1)[1].split(")", 1)[0]  # parens are empty in UCI names
        tail = rest.split(")", 1)[1] if ")" in rest else ""  # e.g. '-X,1'
        axis_part = tail.lstrip("-").strip()
        tokens = [t.strip() for t in axis_part.split(",")] if axis_part else []
        args = None
        axis = None
        if func == "arCoeff" and len(tokens) >= 2:
            axis, args = _AXIS_IDX.get(tokens[0], 0), int(tokens[1])
        elif func == "arCoeff" and len(tokens) == 1 and tokens[0].isdigit():
            axis, args = None, int(tokens[0])
        elif func == "correlation" and len(tokens) >= 2:
            args = (_AXIS_IDX.get(tokens[0], 0), _AXIS_IDX.get(tokens[1], 1))
        elif func == "bandsEnergy" and len(tokens) >= 2:
            args = (int(tokens[0]), int(tokens[1]))
        elif len(tokens) == 1 and tokens[0] in _AXIS_IDX:
            axis = _AXIS_IDX[tokens[0]]
        return (group, func, axis, args)

    # functions without parentheses, e.g. 'maxInds-X'
    pieces = rest.split("-")
    func = pieces[0]
    axis = _AXIS_IDX.get(pieces[1].strip()) if len(pieces) > 1 else None
    return (group, func, axis, None)


# --------------------------------------------------------------------------
# Signal processing primitives (vectorised over a leading batch dimension)
# --------------------------------------------------------------------------

def _median_filter3(X):
    """Width-3 median filter along axis=1 (UCI noise-removal step). Edges kept."""
    if X.shape[1] < 3:
        return X.copy()
    left = np.concatenate([X[:, :1], X[:, :-1]], axis=1)
    right = np.concatenate([X[:, 1:], X[:, -1:]], axis=1)
    return np.median(np.stack([left, X, right], axis=0), axis=0)


def _butterworth_lowpass(X, cutoff_hz=GRAVITY_CUTOFF_HZ, order=4, fs=FS):
    """Zero-phase Butterworth low-pass along axis=1, with FFT-domain fallback."""
    try:
        from scipy.signal import butter, filtfilt
        b, a = butter(order, cutoff_hz / (fs / 2.0), btype="low")
        return filtfilt(b, a, X, axis=1)
    except Exception:
        n = X.shape[1]
        spec = np.fft.rfft(X, n=n, axis=1)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        spec[:, freqs > cutoff_hz] = 0.0
        return np.fft.irfft(spec, n=n, axis=1)


def _burg_ar(X, order=AR_ORDER):
    """Burg autoregressive reflection coefficients, vectorised over rows."""
    Xc = X - X.mean(axis=1, keepdims=True)
    ef = Xc.copy()
    eb = Xc.copy()
    out = np.zeros((X.shape[0], order))
    for k in range(order):
        if ef.shape[1] < 2:
            break
        denom = (ef[:, 1:] ** 2).sum(axis=1) + (eb[:, :-1] ** 2).sum(axis=1)
        num = (eb[:, :-1] * ef[:, 1:]).sum(axis=1)
        kk = -2.0 * num / np.maximum(denom, 1e-12)
        out[:, k] = kk
        ef, eb = ef[:, 1:] + kk[:, None] * eb[:, :-1], eb[:, :-1] + kk[:, None] * ef[:, 1:]
    return np.clip(out, -0.999, 0.999)


def _entropy(X):
    """Normalised Shannon entropy of the |signal| distribution, in [0, 1]."""
    absx = np.abs(X)
    total = absx.sum(axis=1)
    safe = np.where(total > 1e-12, total, 1.0)
    p = absx / safe[:, None]
    logp = np.zeros_like(p)
    np.log2(p, out=logp, where=p > 0)
    h = -(p * logp).sum(axis=1)
    h = np.where(total > 1e-12, h, 0.0)
    return h / np.log2(max(X.shape[1], 2))


def _time_stats(X):
    """Per-row statistics dict for a time-domain signal matrix (M, N)."""
    med = np.median(X, axis=1)
    p25, p75 = np.percentile(X, [25, 75], axis=1)
    return {
        "mean": X.mean(axis=1),
        "std": X.std(axis=1, ddof=1) if X.shape[1] > 1 else np.zeros(X.shape[0]),
        "mad": np.median(np.abs(X - med[:, None]), axis=1),
        "max": X.max(axis=1),
        "min": X.min(axis=1),
        "energy": (X ** 2).mean(axis=1),
        "iqr": p75 - p25,
        "entropy": _entropy(X),
    }


def _pearson_rows(A, B):
    """Row-wise Pearson correlation between two same-shape matrices."""
    a = A - A.mean(axis=1, keepdims=True)
    b = B - B.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum(axis=1))
    ok = den > 1e-12
    return np.where(ok, num / np.where(ok, den, 1.0), 0.0)


def _freq_stats(C, freqs):
    """Per-row statistics dict for a frequency-domain signal (complex, (M, n_bins))."""
    mag = np.abs(C)
    n_bins = C.shape[1]
    stats = _time_stats(mag)
    stats["maxInds"] = mag.argmax(axis=1).astype(float) / max(n_bins - 1, 1)
    total = mag.sum(axis=1)
    safe = np.where(total > 1e-12, total, 1.0)
    mean_freq = (mag * freqs[None, :]).sum(axis=1) / safe
    stats["meanFreq"] = np.where(total > 1e-12, mean_freq, 0.0) / (FS / 2.0)
    m2 = (mag ** 2).mean(axis=1)
    m3 = (mag ** 3).mean(axis=1)
    m4 = (mag ** 4).mean(axis=1)
    mu = stats["mean"]
    var = np.maximum(m2 - mu ** 2, 1e-12)
    stats["skewness"] = (m3 - 3 * mu * m2 + 2 * mu ** 3) / var ** 1.5
    stats["kurtosis"] = m4 / var ** 2 - 3.0
    stats["_mag"] = mag
    return stats


def _band_energies(mag, n_bins):
    """Energy in the 14 standard frequency bands -> list of (M,) arrays."""
    out = []
    for a, b in _freq_bands_64:
        lo = int(round((a - 1) * n_bins / FFT_BINS_128))
        hi = int(round(b * n_bins / FFT_BINS_128))
        lo = max(0, min(lo, n_bins - 1))
        hi = max(lo + 1, min(hi, n_bins))
        out.append((mag[:, lo:hi] ** 2).sum(axis=1) / n_bins)
    return out


# --------------------------------------------------------------------------
# UCI signal construction (batched)
# --------------------------------------------------------------------------

def _build_time_signals(AX, AY, AZ, GX, GY, GZ):
    """Raw sensor matrices (M, N) -> dict of UCI time-domain signal matrices."""
    AXf, AYf, AZf = (_median_filter3(s) for s in (AX, AY, AZ))
    GXf, GYf, GZf = (_median_filter3(s) for s in (GX, GY, GZ))

    grav = [_butterworth_lowpass(s) for s in (AXf, AYf, AZf)]
    body = [AXf - grav[0], AYf - grav[1], AZf - grav[2]]

    dt = 1.0 / FS
    body_jerk = [np.gradient(s, dt, axis=1) for s in body]
    gyro_jerk = [np.gradient(s, dt, axis=1) for s in (GXf, GYf, GZf)]

    return {
        "tBodyAcc": body,
        "tGravityAcc": grav,
        "tBodyAccJerk": body_jerk,
        "tBodyGyro": [GXf, GYf, GZf],
        "tBodyGyroJerk": gyro_jerk,
        "tBodyAccMag": np.sqrt(body[0] ** 2 + body[1] ** 2 + body[2] ** 2),
        "tGravityAccMag": np.sqrt(grav[0] ** 2 + grav[1] ** 2 + grav[2] ** 2),
        "tBodyAccJerkMag": np.sqrt(body_jerk[0] ** 2 + body_jerk[1] ** 2 + body_jerk[2] ** 2),
        "tBodyGyroMag": np.sqrt(GXf ** 2 + GYf ** 2 + GZf ** 2),
        "tBodyGyroJerkMag": np.sqrt(gyro_jerk[0] ** 2 + gyro_jerk[1] ** 2 + gyro_jerk[2] ** 2),
    }


def _cos_angle(A, B):
    """Row-wise cosine of the angle between 3-vector matrices (M, 3)."""
    num = (A * B).sum(axis=1)
    den = np.sqrt((A ** 2).sum(axis=1) * (B ** 2).sum(axis=1))
    ok = den > 1e-12
    return np.where(ok, num / np.where(ok, den, 1.0), 0.0)


def extract_features_batch(AX, AY, AZ, GX, GY, GZ, feature_names=None):
    """Compute the 561 UCI features for a batch of raw windows.

    Each input is an (M, N) matrix (M windows, N samples @ 50 Hz).
    Returns an (M, 561) float matrix ordered exactly like features.txt.
    """
    AX, AY, AZ, GX, GY, GZ = (
        np.atleast_2d(np.asarray(a, dtype=float)) for a in (AX, AY, AZ, GX, GY, GZ)
    )
    M, N = AX.shape
    if N < MIN_WINDOW:
        raise ValueError(f"Raw window too short: {N} samples (minimum {MIN_WINDOW} @ 50 Hz).")
    for name, mat in (("accelerometer Y", AY), ("accelerometer Z", AZ),
                      ("gyroscope X", GX), ("gyroscope Y", GY), ("gyroscope Z", GZ)):
        if mat.shape != AX.shape:
            raise ValueError(f"Channel '{name}' shape {mat.shape} != accelerometer X shape {AX.shape}")

    time_sigs = _build_time_signals(AX, AY, AZ, GX, GY, GZ)

    n_bins = N // 2
    freqs = np.fft.rfftfreq(N, d=1.0 / FS)[:n_bins]
    freq_sigs = {}
    for fgroup, tgroup in _FREQ_TO_TIME.items():
        src = time_sigs[tgroup]
        if isinstance(src, list):
            freq_sigs[fgroup] = [np.fft.rfft(s, axis=1)[:, :n_bins] for s in src]
        else:
            freq_sigs[fgroup] = np.fft.rfft(src, axis=1)[:, :n_bins]

    columns = {}

    def t_stats_of(group):
        cached = columns.get(("__stats__", group))
        if cached is None:
            src = time_sigs[group]
            cached = [_time_stats(s) for s in src] if isinstance(src, list) else [_time_stats(src)]
            columns[("__stats__", group)] = cached
        return cached

    def f_stats_of(group):
        cached = columns.get(("__fstats__", group))
        if cached is None:
            src = freq_sigs[group]
            cached = [_freq_stats(s, freqs) for s in src] if isinstance(src, list) else [_freq_stats(src, freqs)]
            columns[("__fstats__", group)] = cached
        return cached

    # ---- time-domain XYZ groups --------------------------------------------
    for group in _TIME_XYZ_GROUPS:
        st = t_stats_of(group)
        x, y, z = time_sigs[group]
        n = x.shape[1]
        for ax_i, stat_dict in enumerate(st):
            for func, values in stat_dict.items():
                columns[(group, func, ax_i)] = values
        columns[(group, "sma", None)] = (
            np.abs(x).sum(1) + np.abs(y).sum(1) + np.abs(z).sum(1)
        ) / n
        ar = [_burg_ar(s) for s in (x, y, z)]
        for ax_i in range(3):
            for o in range(AR_ORDER):
                columns[(group, "arCoeff", ax_i, o + 1)] = ar[ax_i][:, o]
        columns[(group, "correlation", None, (0, 1))] = _pearson_rows(x, y)
        columns[(group, "correlation", None, (0, 2))] = _pearson_rows(x, z)
        columns[(group, "correlation", None, (1, 2))] = _pearson_rows(y, z)

    # ---- time-domain magnitude groups ---------------------------------------
    for group in _TIME_MAG_GROUPS:
        (st,) = t_stats_of(group)
        sig = time_sigs[group]
        for func, values in st.items():
            columns[(group, func, None)] = values
        columns[(group, "sma", None)] = np.abs(sig).sum(1) / sig.shape[1]
        ar = _burg_ar(sig)
        for o in range(AR_ORDER):
            columns[(group, "arCoeff", None, o + 1)] = ar[:, o]

    # ---- frequency-domain XYZ groups ----------------------------------------
    for group in _FREQ_XYZ_GROUPS:
        st = f_stats_of(group)  # list of 3 per-axis stats dicts (X, Y, Z)
        comps = freq_sigs[group]
        nb = comps[0].shape[1]
        for ax_i, stat_dict in enumerate(st):
            for func, values in stat_dict.items():
                if func == "_mag":
                    continue
                columns[(group, func, ax_i)] = values
        columns[(group, "sma", None)] = sum(np.abs(c).sum(1) for c in comps) / nb
        for ax_i, c in enumerate(comps):
            bands = _band_energies(st[ax_i]["_mag"], nb)
            for b_i, (band, energy) in enumerate(zip(_freq_bands_64, bands)):
                columns[(group, "bandsEnergy", ax_i, band)] = energy

    # ---- frequency-domain magnitude groups -----------------------------------
    for group in _FREQ_MAG_GROUPS:
        (st,) = f_stats_of(group)
        for func, values in st.items():
            if func == "_mag":
                continue
            columns[(group, func, None)] = values
        c = freq_sigs[group]
        columns[(group, "sma", None)] = np.abs(c).sum(1) / c.shape[1]

    # ---- angle features -------------------------------------------------------
    def mean_vec(group):
        return np.stack([time_sigs[group][i].mean(axis=1) for i in range(3)], axis=1)

    grav_mean = mean_vec("tGravityAcc")
    mean_vecs = {
        "gravityMean": grav_mean,
        "gravity": grav_mean,
        "tBodyAccMean": mean_vec("tBodyAcc"),
        "tBodyAccJerkMean": mean_vec("tBodyAccJerk"),
        "tBodyGyroMean": mean_vec("tBodyGyro"),
        "tBodyGyroJerkMean": mean_vec("tBodyGyroJerk"),
    }

    names = feature_names or load_feature_names()
    out = np.zeros((M, len(names)), dtype=float)
    band_counter = {}
    for j, fname in enumerate(names):
        parsed = parse_feature_name(fname)
        if parsed[0] == "angle":
            _, (left, right), _, _ = parsed
            v1 = mean_vecs.get(left)
            v2 = mean_vecs.get(right)
            if v1 is None or v2 is None:
                continue
            if left in _AXIS_IDX:
                unit = np.zeros((M, 3))
                unit[:, _AXIS_IDX[left]] = 1.0
                v1 = unit
            out[:, j] = _cos_angle(v1, v2)
            continue
        group, func, axis, args = parsed
        if func == "bandsEnergy" and axis is None:
            # features.txt encodes bandsEnergy axes positionally: 14 bands per axis
            band_counter[group] = band_counter.get(group, 0) + 1
            k = band_counter[group] - 1
            axis, args = k // len(_freq_bands_64), _freq_bands_64[k % len(_freq_bands_64)]
        key = (group, func, axis, args) if args is not None else (group, func, axis)
        col = columns.get(key)
        if col is None:
            raise ValueError(f"No computed value for feature '{fname}' (key {key})")
        out[:, j] = col

    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------------
# Sensor-window front end (JSON payloads -> feature vectors)
# --------------------------------------------------------------------------

def _window_to_matrices(window):
    """Convert a list of sensor samples into six (M, N) channel matrices.

    Accepted sample formats:
      {"accelerometer": {"x": .., "y": .., "z": ..}, "gyroscope": {...}}
      [ax, ay, az, gx, gy, gz]
    """
    ax, ay, az, gx, gy, gz = [], [], [], [], [], []
    for sample in window:
        if isinstance(sample, dict):
            acc = sample.get("accelerometer") or {}
            gyr = sample.get("gyroscope") or {}
            ax.append(float(acc.get("x", 0.0) or 0.0))
            ay.append(float(acc.get("y", 0.0) or 0.0))
            az.append(float(acc.get("z", 0.0) or 0.0))
            gx.append(float(gyr.get("x", 0.0) or 0.0))
            gy.append(float(gyr.get("y", 0.0) or 0.0))
            gz.append(float(gyr.get("z", 0.0) or 0.0))
        elif isinstance(sample, (list, tuple)) and len(sample) == 6:
            vals = [float(v) for v in sample]
            ax.append(vals[0]); ay.append(vals[1]); az.append(vals[2])
            gx.append(vals[3]); gy.append(vals[4]); gz.append(vals[5])
        else:
            raise ValueError("Sensor window samples must be dicts or 6-element arrays.")
    return (np.array(ax)[:, None], np.array(ay)[:, None], np.array(az)[:, None],
            np.array(gx)[:, None], np.array(gy)[:, None], np.array(gz)[:, None])


def normalize_sensor_units(AX, AY, AZ, GX, GY, GZ):
    """Auto-detect and convert units so accelerometer is in g and gyro in rad/s."""
    acc_ref = np.percentile(np.abs(np.concatenate([AX, AY, AZ])), 95)
    if acc_ref > 3.0:  # m/s^2 detected -> convert to g
        AX, AY, AZ = AX / GRAVITY, AY / GRAVITY, AZ / GRAVITY
    gyro_ref = np.percentile(np.abs(np.concatenate([GX, GY, GZ])), 95)
    if gyro_ref > 25.0:  # deg/s detected -> convert to rad/s
        GX, GY, GZ = (np.deg2rad(G) for G in (GX, GY, GZ))
    clamp_acc, clamp_gyr = 6.0, 35.0
    AX, AY, AZ = (np.clip(s, -clamp_acc, clamp_acc) for s in (AX, AY, AZ))
    GX, GY, GZ = (np.clip(s, -clamp_gyr, clamp_gyr) for s in (GX, GY, GZ))
    return AX, AY, AZ, GX, GY, GZ


def extract_features_from_sensor_window(sensor_window):
    """JSON sensor window -> (1, 561) UCI feature matrix (genuine inference path)."""
    if not sensor_window or len(sensor_window) < MIN_WINDOW:
        raise ValueError(
            f"Sensor window has insufficient samples ({0 if not sensor_window else len(sensor_window)}). "
            f"Collect at least {MIN_WINDOW} samples ({MIN_WINDOW / FS:.1f}s @ 50Hz); "
            f"{RECOMMENDED_WINDOW} recommended."
        )
    mats = _window_to_matrices(sensor_window)      # each channel: (N, 1)
    mats = tuple(m.T for m in mats)                # -> one window: (1, N)
    mats = normalize_sensor_units(*mats)
    feats = extract_features_batch(*mats)
    return feats


def validate_feature_vector(features, expected_dim=561):
    """Validate an incoming 561-dimensional feature vector -> (1, 561) matrix."""
    if features is None:
        raise ValueError("Feature vector is None")
    arr = np.asarray(features, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != expected_dim:
        raise ValueError(f"Invalid feature dimension: expected {expected_dim}, got {arr.shape[1]}")
    if not np.isfinite(arr).all():
        raise ValueError("Feature vector contains NaN or infinite values")
    return arr


# --------------------------------------------------------------------------
# Dataset building (from the RAW Inertial Signals of the UCI HAR dataset)
# --------------------------------------------------------------------------

def _inertial_paths(base_dir, split):
    root = os.path.join(base_dir, "dataset", "UCI HAR Dataset", split, "Inertial Signals")
    names = ["total_acc_x", "total_acc_y", "total_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z"]
    return [os.path.join(root, f"{n}_{split}.txt") for n in names]


def load_raw_windows(base_dir=None, split="test"):
    """Load raw sensor windows -> (windows (M, 6, N) float, labels (M,) int 0-5)."""
    base = base_dir or BASE_DIR
    paths = _inertial_paths(base, split)
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"UCI Inertial Signals file not found: {p}")
    label_path = os.path.join(base, "dataset", "UCI HAR Dataset", split, f"y_{split}.txt")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"UCI labels file not found: {label_path}")

    channels = [np.loadtxt(p) for p in paths]
    n = min(c.shape[0] for c in channels)
    windows = np.stack([c[:n] for c in channels], axis=1)  # (M, 6, N)
    labels = np.loadtxt(label_path).astype(int)[:n] - 1    # 1..6 -> 0..5
    return windows, labels


def build_dataset_features(base_dir=None, split="train"):
    """Extract the 561-dim features for every raw window of a dataset split."""
    windows, labels = load_raw_windows(base_dir, split)
    AX, AY, AZ, GX, GY, GZ = (windows[:, i, :] for i in range(6))
    AX, AY, AZ, GX, GY, GZ = normalize_sensor_units(AX, AY, AZ, GX, GY, GZ)
    X = extract_features_batch(AX, AY, AZ, GX, GY, GZ)
    return X, labels


# --------------------------------------------------------------------------
# Synthetic fallback (activity-coherent raw windows, used only when the
# real dataset is absent so that the app still demonstrably works)
# --------------------------------------------------------------------------

_SYNTH_GRAVITY = {
    "WALKING": (0.0, 0.15, 0.99), "WALKING_UPSTAIRS": (0.0, 0.55, 0.83),
    "WALKING_DOWNSTAIRS": (0.0, -0.5, 0.86), "SITTING": (0.1, 0.95, 0.3),
    "STANDING": (0.05, 0.1, 0.99), "LAYING": (0.98, 0.1, 0.15),
}


def generate_synthetic_windows(n_windows, fs=FS, seed=42):
    """Generate activity-coherent raw windows with labels for fallback mode."""
    rng = np.random.default_rng(seed)
    n = int(RECOMMENDED_WINDOW)
    t = np.arange(n) / fs
    windows = np.zeros((n_windows, 6, n))
    labels = np.zeros(n_windows, dtype=int)
    for i in range(n_windows):
        lbl = int(rng.integers(0, 6))
        labels[i] = lbl
        gx_, gy_, gz_ = _SYNTH_GRAVITY[ACTIVITY_KEYS[lbl]]
        if lbl in (0, 1, 2):  # walking family
            f = {0: 1.8, 1: 1.4, 2: 1.7}[lbl]
            amp = {0: 0.16, 1: 0.26, 2: 0.22}[lbl]
            wob = {0: 0.0, 1: 0.25, 2: -0.25}[lbl] * np.sin(2 * np.pi * 0.35 * t)
            ax_ = amp * np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) + wob
            ay_ = 0.6 * amp * np.sin(2 * np.pi * f * 2 * t) + gy_ + wob * 0.5
            az_ = gz_ + amp * np.sin(2 * np.pi * f * t + 1.0) + wob
            gyrox = 0.7 * np.sin(2 * np.pi * f * t)
            gyroy = 0.5 * np.cos(2 * np.pi * f * t)
            gyroz = 0.4 * np.sin(2 * np.pi * f * t + 0.7)
        else:  # static postures
            ax_, ay_, az_ = gx_, gy_, gz_
            gyrox = gyroy = gyroz = np.zeros(n)
        noise = rng.normal(0, 0.012, (6, n))
        windows[i, 0] = ax_ + noise[0]
        windows[i, 1] = ay_ + noise[1]
        windows[i, 2] = az_ + noise[2]
        windows[i, 3] = gyrox + noise[3]
        windows[i, 4] = gyroy + noise[4]
        windows[i, 5] = gyroz + noise[5]
    return windows, labels


ACTIVITY_KEYS = [
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING",
]


# --------------------------------------------------------------------------
# Descriptive demo mapping (heuristic class routing; the feature vector is
# drawn from real dataset class profiles maintained by the predictor)
# --------------------------------------------------------------------------

def descriptive_to_class(intensity, stability, orientation, rotation, pattern):
    """Map descriptive movement parameters to the most plausible activity class."""
    orientation = (orientation or "Standing").capitalize()
    intensity = (intensity or "Medium").capitalize()
    rotation = (rotation or "Moderate").capitalize()
    pattern = (pattern or "Regular").capitalize()

    if orientation == "Lying":
        return 5  # LAYING
    if orientation == "Sitting":
        return 3  # SITTING
    if intensity == "Low" or pattern == "Still":
        return 4  # STANDING
    if rotation == "High" and pattern == "Rhythmic":
        return 1  # WALKING_UPSTAIRS
    if rotation == "High":
        return 2  # WALKING_DOWNSTAIRS
    return 0      # WALKING
