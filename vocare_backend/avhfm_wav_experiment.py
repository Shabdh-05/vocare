import os
import warnings
import numpy as np
import pandas as pd
import librosa
import parselmouth

from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "..", "vocal_features.csv")
WAV_ROOT = r"C:\healthcare_model2\data"

# ============================================================
# LOAD LABELS
# ============================================================

print("Loading labeled recordings...")

df = pd.read_csv(CSV_PATH)

# Map every WAV by filename
print("Indexing WAV files...")

wav_map = {}

for root, dirs, files in os.walk(WAV_ROOT):
    for file in files:
        if file.lower().endswith(".wav"):
            wav_map[file.lower()] = os.path.join(root, file)

paths = []
valid_rows = []

for i, path in enumerate(df["file_path"]):
    filename = os.path.basename(path).lower()

    if filename in wav_map:
        paths.append(wav_map[filename])
        valid_rows.append(i)

df = df.iloc[valid_rows].reset_index(drop=True)

print("Labeled WAV recordings:", len(df))

# ============================================================
# RICH WAV FEATURES
# ============================================================

def extract_rich_features(path):

    try:
        y, sr = librosa.load(
            path,
            sr=16000,
            mono=True
        )

        if len(y) < 1000:
            return None

        # Normalize safely
        y = y.astype(np.float64)

        peak = np.max(np.abs(y))

        if peak > 0:
            y = y / peak

        # ----------------------------
        # Time-domain
        # ----------------------------

        rms = librosa.feature.rms(
            y=y,
            frame_length=2048,
            hop_length=512
        )[0]

        zcr = librosa.feature.zero_crossing_rate(
            y,
            frame_length=2048,
            hop_length=512
        )[0]

        # ----------------------------
        # Spectral
        # ----------------------------

        centroid = librosa.feature.spectral_centroid(
            y=y,
            sr=sr,
            n_fft=2048,
            hop_length=512
        )[0]

        bandwidth = librosa.feature.spectral_bandwidth(
            y=y,
            sr=sr,
            n_fft=2048,
            hop_length=512
        )[0]

        rolloff = librosa.feature.spectral_rolloff(
            y=y,
            sr=sr,
            n_fft=2048,
            hop_length=512
        )[0]

        contrast = librosa.feature.spectral_contrast(
            y=y,
            sr=sr,
            n_fft=2048,
            hop_length=512
        )

        flatness = librosa.feature.spectral_flatness(
            y=y,
            n_fft=2048,
            hop_length=512
        )[0]

        # ----------------------------
        # MFCC
        # ----------------------------

        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=20,
            n_fft=2048,
            hop_length=512
        )

        mfcc_delta = librosa.feature.delta(mfcc)

        mfcc_delta2 = librosa.feature.delta(
            mfcc,
            order=2
        )

        # ----------------------------
        # Chroma
        # ----------------------------

        chroma = librosa.feature.chroma_stft(
            y=y,
            sr=sr,
            n_fft=2048,
            hop_length=512
        )

        # ----------------------------
        # Spectral flux
        # ----------------------------

        onset = librosa.onset.onset_strength(
            y=y,
            sr=sr,
            hop_length=512
        )

        # ----------------------------
        # Pitch + voice quality
        # ----------------------------

        snd = parselmouth.Sound(
            y,
            sampling_frequency=sr
        )

        pitch_obj = snd.to_pitch(
            time_step=0.01,
            pitch_floor=75,
            pitch_ceiling=500
        )

        pitch_values = pitch_obj.selected_array["frequency"]
        pitch_values = pitch_values[pitch_values > 0]

        if len(pitch_values) == 0:
            pitch_mean = 0.0
            pitch_std = 0.0
            pitch_min = 0.0
            pitch_max = 0.0
            pitch_range = 0.0
            pitch_median = 0.0
        else:
            pitch_mean = np.mean(pitch_values)
            pitch_std = np.std(pitch_values)
            pitch_min = np.min(pitch_values)
            pitch_max = np.max(pitch_values)
            pitch_range = pitch_max - pitch_min
            pitch_median = np.median(pitch_values)

        # HNR
        harmonicity = snd.to_harmonicity(
            time_step=0.01,
            minimum_pitch=75
        )

        hnr_values = harmonicity.values[0]
        hnr_values = hnr_values[
            np.isfinite(hnr_values)
            & (hnr_values > -100)
        ]

        hnr = (
            np.mean(hnr_values)
            if len(hnr_values)
            else 0.0
        )

        # Jitter / shimmer
        try:
            point_process = parselmouth.praat.call(
                [snd, pitch_obj],
                "To PointProcess (cc)"
            )

            jitter = parselmouth.praat.call(
                point_process,
                "Get jitter (local)",
                0,
                0,
                0.0001,
                0.02,
                1.3
            )

            shimmer = parselmouth.praat.call(
                [snd, point_process],
                "Get shimmer (local)",
                0,
                0,
                0.0001,
                0.02,
                1.3,
                1.6
            )

            if not np.isfinite(jitter):
                jitter = 0.0

            if not np.isfinite(shimmer):
                shimmer = 0.0

        except Exception:
            jitter = 0.0
            shimmer = 0.0

        # ====================================================
        # FEATURE DICTIONARY
        # ====================================================

        f = {}

        def add_stats(prefix, values):
            values = np.asarray(values).flatten()
            values = values[np.isfinite(values)]

            if len(values) == 0:
                values = np.array([0.0])

            f[prefix + "_mean"] = np.mean(values)
            f[prefix + "_std"] = np.std(values)
            f[prefix + "_min"] = np.min(values)
            f[prefix + "_max"] = np.max(values)
            f[prefix + "_median"] = np.median(values)

        add_stats("rms", rms)
        add_stats("zcr", zcr)
        add_stats("centroid", centroid)
        add_stats("bandwidth", bandwidth)
        add_stats("rolloff", rolloff)
        add_stats("flatness", flatness)
        add_stats("onset", onset)

        add_stats("contrast", contrast)

        add_stats("chroma", chroma)

        add_stats("mfcc", mfcc)
        add_stats("mfcc_delta", mfcc_delta)
        add_stats("mfcc_delta2", mfcc_delta2)

        f["pitch_mean"] = pitch_mean
        f["pitch_std"] = pitch_std
        f["pitch_min"] = pitch_min
        f["pitch_max"] = pitch_max
        f["pitch_range"] = pitch_range
        f["pitch_median"] = pitch_median

        f["jitter"] = jitter
        f["shimmer"] = shimmer
        f["hnr"] = hnr

        # ====================================================
        # AVHFM DERIVED FEATURES
        # ====================================================

        eps = 1e-8

        f["perturbation_index"] = (
            jitter + shimmer
        )

        f["vocal_stability_index"] = (
            hnr /
            (pitch_std + jitter + shimmer + eps)
        )

        f["phonation_energy_index"] = (
            f["rms_mean"] /
            (f["zcr_mean"] + eps)
        )

        f["spectral_irregularity_index"] = (
            f["centroid_std"]
            + f["bandwidth_std"]
            + f["rolloff_std"]
        )

        f["pitch_jitter_interaction"] = (
            pitch_std * jitter
        )

        f["pitch_shimmer_interaction"] = (
            pitch_std * shimmer
        )

        f["hnr_jitter_ratio"] = (
            hnr / (jitter + eps)
        )

        f["hnr_shimmer_ratio"] = (
            hnr / (shimmer + eps)
        )

        f["voice_quality_index"] = (
            hnr /
            (
                jitter
                + shimmer
                + f["zcr_mean"]
                + eps
            )
        )

        return f

    except Exception:
        return None


# ============================================================
# EXTRACT WAV FEATURES
# ============================================================

print("\nExtracting rich WAV features...")
print("This may take several minutes.\n")

records = []

for i, path in enumerate(paths):

    features = extract_rich_features(path)

    if features is not None:
        records.append(features)

    if (i + 1) % 500 == 0:
        print(
            f"Processed {i + 1}/{len(paths)}"
        )

X_wav = pd.DataFrame(records)

print("\nWAV feature matrix:", X_wav.shape)

# Keep only rows successfully extracted
df_used = df.iloc[:len(X_wav)].copy()

# ============================================================
# COMBINE WITH ORIGINAL AVHFM FEATURES
# ============================================================

# Original numeric features
DROP = [
    "file_path",
    "disease",
    "patient_id",
    "file_name",
    "label"
]

original_features = [
    c for c in df.columns
    if c not in DROP
]

X_original = df[original_features].copy()

# Align lengths
X_original = X_original.iloc[:len(X_wav)].reset_index(drop=True)
X_wav = X_wav.reset_index(drop=True)

X = pd.concat(
    [X_original, X_wav],
    axis=1
)

# Remove duplicate columns
X = X.loc[
    :,
    ~X.columns.duplicated()
]

y = df["label"].iloc[
    :len(X)
].astype(int).reset_index(drop=True)

groups = df["patient_id"].iloc[
    :len(X)
].reset_index(drop=True)

print("Combined feature matrix:", X.shape)
print("Patients:", groups.nunique())

# ============================================================
# PATIENT-LEVEL SPLIT
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

print("\nTrain patients:", groups.iloc[train_idx].nunique())
print("Test patients :", groups.iloc[test_idx].nunique())

# ============================================================
# IMPUTATION
# ============================================================

imputer = SimpleImputer(
    strategy="median"
)

X_train = imputer.fit_transform(
    X_train
)

X_test = imputer.transform(
    X_test
)

# ============================================================
# EXTRA TREES
# ============================================================

print("\nTraining ExtraTrees...")

model = ExtraTreesClassifier(
    n_estimators=1000,
    max_features="sqrt",
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

# ============================================================
# EVALUATION
# ============================================================

prob = model.predict_proba(X_test)[:, 1]

pred = (
    prob >= 0.50
).astype(int)

accuracy = accuracy_score(
    y_test,
    pred
)

precision = precision_score(
    y_test,
    pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    pred,
    zero_division=0
)

auc = roc_auc_score(
    y_test,
    prob
)

print("\n" + "=" * 60)
print("RICH WAV + AVHFM RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1 Score : {f1 * 100:.2f}%")
print(f"ROC-AUC  : {auc * 100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))

print("=" * 60)