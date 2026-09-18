import os
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

# ---------------------------------------------------------
# Load the existing feature dataset
# ---------------------------------------------------------

base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)

csv_path = os.path.join(project_root, "vocal_features.csv")

print(f"Loading: {csv_path}")

df = pd.read_csv(csv_path)

print(f"Recordings: {len(df)}")
print(f"Patients: {df['patient_id'].nunique()}")

# ---------------------------------------------------------
# Create the same 8 AVHFM features used by the project
# ---------------------------------------------------------

eps = 1e-6

df["vocal_stability_index"] = (
    df["hnr"] /
    (df["pitch_std"] + df["jitter"] + df["shimmer"] + eps)
)

df["perturbation_index"] = (
    df["jitter"] + df["shimmer"]
)

df["phonation_energy_index"] = (
    df["rms_mean"] / (df["zcr_mean"] + eps)
)

df["spectral_irregularity_index"] = (
    df["centroid_std"]
    + df["bandwidth_std"]
    + df["rolloff_std"]
)

mfcc_std_cols = [
    f"mfcc_{i+1}_std"
    for i in range(13)
]

df["mfcc_irregularity_index"] = df[mfcc_std_cols].mean(axis=1)

df["avhi"] = (
    0.30 * df["perturbation_index"]
    + 0.25 * df["pitch_std"]
    + 0.20 * df["spectral_irregularity_index"]
    + 0.15 * df["mfcc_irregularity_index"]
    - 0.10 * df["hnr"]
)

df["pitch_jitter_interaction"] = (
    df["pitch_std"] * df["jitter"]
)

df["pitch_shimmer_interaction"] = (
    df["pitch_std"] * df["shimmer"]
)

# ---------------------------------------------------------
# Select the 77 AVHFM features
# ---------------------------------------------------------

exclude_cols = {
    "file_path",
    "disease",
    "patient_id",
    "file_name",
    "label",
}

feature_cols = [
    c for c in df.columns
    if c not in exclude_cols
]

print(f"Features per recording: {len(feature_cols)}")

# ---------------------------------------------------------
# Clean invalid values
# ---------------------------------------------------------

X_recordings = df[feature_cols].copy()
X_recordings = X_recordings.replace(
    [np.inf, -np.inf],
    np.nan
)

y_recordings = df["label"].astype(int)
groups = df["patient_id"]

# ---------------------------------------------------------
# Patient-level train/test split
# ---------------------------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(
        X_recordings,
        y_recordings,
        groups=groups,
    )
)

train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()

print()
print("Patient-level split")
print("-------------------")
print(f"Training recordings: {len(train_df)}")
print(f"Testing recordings : {len(test_df)}")
print(
    f"Training patients  : {train_df['patient_id'].nunique()}"
)
print(
    f"Testing patients   : {test_df['patient_id'].nunique()}"
)

overlap = set(
    train_df["patient_id"]
).intersection(
    set(test_df["patient_id"])
)

print(f"Patient overlap    : {len(overlap)}")

# ---------------------------------------------------------
# IMPORTANT:
# Aggregate recordings by patient.
#
# We calculate mean and standard deviation for every
# acoustic/AVHFM feature.
#
# We DO NOT use recording count as a feature.
# ---------------------------------------------------------

train_features = train_df[feature_cols].copy()
test_features = test_df[feature_cols].copy()

train_features = train_features.replace(
    [np.inf, -np.inf],
    np.nan,
)

test_features = test_features.replace(
    [np.inf, -np.inf],
    np.nan,
)

train_features["patient_id"] = train_df["patient_id"].values
test_features["patient_id"] = test_df["patient_id"].values

train_features["label"] = train_df["label"].values
test_features["label"] = test_df["label"].values

train_mean = (
    train_features
    .groupby("patient_id")[feature_cols]
    .mean()
)

train_std = (
    train_features
    .groupby("patient_id")[feature_cols]
    .std()
    .fillna(0)
)

test_mean = (
    test_features
    .groupby("patient_id")[feature_cols]
    .mean()
)

test_std = (
    test_features
    .groupby("patient_id")[feature_cols]
    .std()
    .fillna(0)
)

train_X = pd.concat(
    [
        train_mean.add_suffix("_mean"),
        train_std.add_suffix("_std"),
    ],
    axis=1,
)

test_X = pd.concat(
    [
        test_mean.add_suffix("_mean"),
        test_std.add_suffix("_std"),
    ],
    axis=1,
)

train_y = (
    train_df
    .groupby("patient_id")["label"]
    .first()
)

test_y = (
    test_df
    .groupby("patient_id")["label"]
    .first()
)

print()
print("Patient-level representation")
print("----------------------------")
print(f"Training patients: {len(train_X)}")
print(f"Testing patients : {len(test_X)}")
print(f"Patient features : {train_X.shape[1]}")

# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

model = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median"),
    ),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight=None,
            random_state=42,
            n_jobs=-1,
        ),
    ),
])

print()
print("Training patient-level Random Forest...")

model.fit(train_X, train_y)

# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

pred = model.predict(test_X)
prob = model.predict_proba(test_X)[:, 1]

accuracy = accuracy_score(test_y, pred)
precision = precision_score(test_y, pred)
recall = recall_score(test_y, pred)
f1 = f1_score(test_y, pred)
auc = roc_auc_score(test_y, prob)

print()
print("==============================")
print("PATIENT-LEVEL BASELINE RESULTS")
print("==============================")
print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1 Score : {f1 * 100:.2f}%")
print(f"ROC-AUC  : {auc:.4f}")

print()
print("Classification Report")
print("----------------------")
print(
    classification_report(
        test_y,
        pred,
        target_names=["Healthy", "Disorder"],
    )
)