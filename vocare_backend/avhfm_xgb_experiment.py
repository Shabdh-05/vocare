import os
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from xgboost import XGBClassifier


# ============================================================
# 1. Paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "..", "vocal_features.csv")


# ============================================================
# 2. Load dataset
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(CSV_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# 3. Create AVHFM features
# ============================================================

EPS = 1e-8

df["vocal_stability_index"] = (
    df["hnr"] /
    (df["pitch_std"] + df["jitter"] + df["shimmer"] + EPS)
)

df["perturbation_index"] = (
    df["jitter"] + df["shimmer"]
)

df["phonation_energy_index"] = (
    df["rms_mean"] /
    (df["zcr_mean"] + EPS)
)

df["spectral_irregularity_index"] = (
    df["centroid_std"] +
    df["bandwidth_std"] +
    df["rolloff_std"]
)

mfcc_std_cols = [
    f"mfcc_{i}_std"
    for i in range(1, 14)
    if f"mfcc_{i}_std" in df.columns
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


# ============================================================
# 4. Select features
# ============================================================

DROP_COLUMNS = [
    "file_path",
    "disease",
    "patient_id",
    "file_name",
    "label",
]

feature_columns = [
    c for c in df.columns
    if c not in DROP_COLUMNS
]

X = df[feature_columns].copy()
y = df["label"].astype(int)
groups = df["patient_id"]


print("Number of AVHFM features:", len(feature_columns))
print("Number of patients:", groups.nunique())


# ============================================================
# 5. Patient-level train/test split
# ============================================================

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    gss.split(X, y, groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]


print("\nTrain recordings:", len(X_train))
print("Test recordings :", len(X_test))

print(
    "Train patients:",
    groups.iloc[train_idx].nunique()
)

print(
    "Test patients :",
    groups.iloc[test_idx].nunique()
)

print(
    "Patient overlap:",
    len(
        set(groups.iloc[train_idx])
        & set(groups.iloc[test_idx])
    )
)


# ============================================================
# 6. Handle missing values
# ============================================================

imputer = SimpleImputer(strategy="median")

X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)


# ============================================================
# 7. XGBoost
# ============================================================

print("\nTraining XGBoost...")

model = XGBClassifier(
    n_estimators=700,
    max_depth=5,
    learning_rate=0.035,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.05,
    reg_lambda=2.0,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
)

model.fit(
    X_train_imp,
    y_train
)


# ============================================================
# 8. Evaluate
# ============================================================

probabilities = model.predict_proba(X_test_imp)[:, 1]

predictions = (
    probabilities >= 0.50
).astype(int)


accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

auc = roc_auc_score(
    y_test,
    probabilities
)


# ============================================================
# 9. Results
# ============================================================

print("\n" + "=" * 60)
print("AVHFM + XGBOOST RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1 Score : {f1 * 100:.2f}%")
print(f"ROC-AUC  : {auc * 100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        digits=4,
        zero_division=0
    )
)

print("=" * 60)