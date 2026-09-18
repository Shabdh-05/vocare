import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier
)
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from sklearn.feature_selection import SelectKBest, f_classif

warnings.filterwarnings("ignore")

# ============================================================
# ENHANCED AVHFM V2
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "..", "vocal_features.csv")

print("=" * 70)
print("ENHANCED AVHFM V2 EXPERIMENT")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

print("\n[1/8] Loading vocal_features.csv...")

df = pd.read_csv(CSV_PATH)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

# ------------------------------------------------------------
# 2. CREATE AVHFM FEATURES
# ------------------------------------------------------------

print("\n[2/8] Creating AVHFM handcrafted features...")

EPS = 1e-8

df["vocal_stability_index"] = (
    df["hnr"] /
    (
        df["pitch_std"]
        + df["jitter"]
        + df["shimmer"]
        + EPS
    )
)

df["perturbation_index"] = (
    df["jitter"] + df["shimmer"]
)

df["phonation_energy_index"] = (
    df["rms_mean"] /
    (df["zcr_mean"] + EPS)
)

df["spectral_irregularity_index"] = (
    df["centroid_std"]
    + df["bandwidth_std"]
    + df["rolloff_std"]
)

mfcc_std_columns = [
    f"mfcc_{i}_std"
    for i in range(1, 14)
    if f"mfcc_{i}_std" in df.columns
]

df["mfcc_irregularity_index"] = (
    df[mfcc_std_columns].mean(axis=1)
)

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

# ------------------------------------------------------------
# 3. ADD SECOND-ORDER AVHFM FEATURES
# ------------------------------------------------------------

print("[3/8] Creating enhanced AVHFM interaction features...")

df["hnr_jitter_ratio"] = (
    df["hnr"] / (df["jitter"] + EPS)
)

df["hnr_shimmer_ratio"] = (
    df["hnr"] / (df["shimmer"] + EPS)
)

df["pitch_variability_ratio"] = (
    df["pitch_std"] / (df["pitch_mean"].abs() + EPS)
)

df["spectral_variability_ratio"] = (
    df["spectral_irregularity_index"]
    / (df["centroid_mean"].abs() + EPS)
)

df["mfcc_variability_ratio"] = (
    df["mfcc_irregularity_index"]
    / (df["rms_mean"].abs() + EPS)
)

df["jitter_shimmer_ratio"] = (
    df["jitter"] / (df["shimmer"] + EPS)
)

df["voice_quality_index"] = (
    df["hnr"]
    / (
        1.0
        + df["jitter"]
        + df["shimmer"]
        + df["pitch_std"]
    )
)

df["perturbation_pitch_ratio"] = (
    df["perturbation_index"]
    / (df["pitch_mean"].abs() + EPS)
)

# ------------------------------------------------------------
# 4. SELECT FEATURES
# ------------------------------------------------------------

print("[4/8] Preparing AVHFM feature matrix...")

DROP_COLUMNS = [
    "file_path",
    "disease",
    "patient_id",
    "file_name",
    "label"
]

feature_columns = [
    c for c in df.columns
    if c not in DROP_COLUMNS
]

X = df[feature_columns].copy()
y = df["label"].astype(int)

# Replace problematic values
X = X.replace([np.inf, -np.inf], np.nan)

print(f"Total AVHFM features: {X.shape[1]}")
print(f"Class 0: {(y == 0).sum():,}")
print(f"Class 1: {(y == 1).sum():,}")

# ------------------------------------------------------------
# 5. FIXED RECORDING-LEVEL HOLDOUT
# ------------------------------------------------------------

print("\n[5/8] Creating fixed evaluation split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print(f"Training recordings: {len(X_train):,}")
print(f"Testing recordings : {len(X_test):,}")

# ------------------------------------------------------------
# 6. FEATURE SELECTION
# ------------------------------------------------------------

print("\n[6/8] Selecting informative AVHFM features...")

imputer = SimpleImputer(strategy="median")

X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

# Test several feature counts using ONLY training data.
candidate_k = [
    25,
    35,
    45,
    55,
    65,
    min(75, X_train.shape[1]),
    X_train.shape[1]
]

candidate_k = sorted(set(
    k for k in candidate_k
    if k <= X_train.shape[1]
))

best_k = None
best_train_score = -1

for k in candidate_k:

    selector = SelectKBest(
        score_func=f_classif,
        k=k
    )

    Xtr = selector.fit_transform(
        X_train_imp,
        y_train
    )

    model = ExtraTreesClassifier(
        n_estimators=400,
        max_features="sqrt",
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(Xtr, y_train)

    train_pred = model.predict(Xtr)

    score = f1_score(
        y_train,
        train_pred
    )

    print(
        f"  k={k:3d} -> training F1={score:.4f}"
    )

    if score > best_train_score:
        best_train_score = score
        best_k = k

print(f"\nSelected AVHFM feature count: {best_k}")

selector = SelectKBest(
    score_func=f_classif,
    k=best_k
)

X_train_selected = selector.fit_transform(
    X_train_imp,
    y_train
)

X_test_selected = selector.transform(
    X_test_imp
)

selected_features = list(
    X_train.columns[
        selector.get_support()
    ]
)

print("\nSelected AVHFM features:")
for feature in selected_features:
    print("  -", feature)

# ------------------------------------------------------------
# 7. TRAIN MULTIPLE AVHFM CLASSIFIERS
# ------------------------------------------------------------

print("\n[7/8] Training AVHFM classifiers...")

models = {}

# ExtraTrees
models["ExtraTrees"] = ExtraTreesClassifier(
    n_estimators=800,
    max_features="sqrt",
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# Random Forest
models["RandomForest"] = RandomForestClassifier(
    n_estimators=800,
    max_depth=None,
    max_features="sqrt",
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# HistGradientBoosting
models["HistGradientBoosting"] = HistGradientBoostingClassifier(
    max_iter=400,
    learning_rate=0.04,
    max_leaf_nodes=31,
    max_depth=None,
    min_samples_leaf=25,
    l2_regularization=1.0,
    random_state=42
)

# SVM
models["SVM"] = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "svc",
        SVC(
            C=5,
            gamma="scale",
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=42
        )
    )
])

results = []
probabilities = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(
        X_train_selected,
        y_train
    )

    pred = model.predict(
        X_test_selected
    )

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(
            X_test_selected
        )[:, 1]
    else:
        prob = model.decision_function(
            X_test_selected
        )

    probabilities[name] = prob

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

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": auc
    })

    print(
        f"  Accuracy : {accuracy * 100:.2f}%"
    )
    print(
        f"  Precision: {precision * 100:.2f}%"
    )
    print(
        f"  Recall   : {recall * 100:.2f}%"
    )
    print(
        f"  F1       : {f1 * 100:.2f}%"
    )
    print(
        f"  ROC-AUC  : {auc * 100:.2f}%"
    )

# ------------------------------------------------------------
# 8. AVHFM SOFT-VOTING ENSEMBLE
# ------------------------------------------------------------

print("\n[8/8] Building AVHFM weighted ensemble...")

# AVHFM-style weighted voting:
# SVM = 3
# ExtraTrees/RF = 2
# Gradient Boosting = 1

ensemble_probability = (
    3.0 * probabilities["SVM"]
    + 2.0 * probabilities["RandomForest"]
    + 2.0 * probabilities["ExtraTrees"]
    + 1.0 * probabilities["HistGradientBoosting"]
) / 8.0

ensemble_prediction = (
    ensemble_probability >= 0.50
).astype(int)

accuracy = accuracy_score(
    y_test,
    ensemble_prediction
)

precision = precision_score(
    y_test,
    ensemble_prediction,
    zero_division=0
)

recall = recall_score(
    y_test,
    ensemble_prediction,
    zero_division=0
)

f1 = f1_score(
    y_test,
    ensemble_prediction,
    zero_division=0
)

auc = roc_auc_score(
    y_test,
    ensemble_probability
)

results.append({
    "Model": "AVHFM V2 Weighted Ensemble",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
    "ROC_AUC": auc
})

print("\n" + "=" * 70)
print("FINAL AVHFM V2 ENSEMBLE")
print("=" * 70)

print(
    f"Accuracy : {accuracy * 100:.2f}%"
)
print(
    f"Precision: {precision * 100:.2f}%"
)
print(
    f"Recall   : {recall * 100:.2f}%"
)
print(
    f"F1       : {f1 * 100:.2f}%"
)
print(
    f"ROC-AUC  : {auc * 100:.2f}%"
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        ensemble_prediction
    )
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        ensemble_prediction,
        target_names=[
            "Healthy",
            "Disorder"
        ],
        zero_division=0
    )
)

# ------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

results_path = os.path.join(
    BASE_DIR,
    "avhfm_v2_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

print("\nModel comparison:")
print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# ------------------------------------------------------------
# SAVE THE BEST INDIVIDUAL MODEL
# ------------------------------------------------------------

best_row = results_df.sort_values(
    by="F1",
    ascending=False
).iloc[0]

best_name = best_row["Model"]

print(
    f"\nBest model by F1: {best_name}"
)

# Save the ensemble components and preprocessing
artifact = {
    "models": models,
    "imputer": imputer,
    "selector": selector,
    "selected_features": selected_features,
    "all_features": feature_columns,
    "ensemble_weights": {
        "SVM": 3.0,
        "RandomForest": 2.0,
        "ExtraTrees": 2.0,
        "HistGradientBoosting": 1.0
    },
    "threshold": 0.50
}

artifact_path = os.path.join(
    BASE_DIR,
    "avhfm_v2_experiment.pkl"
)

joblib.dump(
    artifact,
    artifact_path
)

print(
    f"\nSaved experiment artifact:"
)
print(
    artifact_path
)

print("\n" + "=" * 70)
print("AVHFM V2 EXPERIMENT COMPLETE")
print("=" * 70)