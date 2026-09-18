import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb

base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)
csv_path = os.path.join(project_root, "vocal_features.csv")

print("Loading data...", flush=True)
df = pd.read_csv(csv_path)

eps = 1e-6
df["vocal_stability_index"] = df["hnr"] / (df["pitch_std"] + df["jitter"] + df["shimmer"] + eps)
df["perturbation_index"] = df["jitter"] + df["shimmer"]
df["phonation_energy_index"] = df["rms_mean"] / (df["zcr_mean"] + eps)
df["spectral_irregularity_index"] = df["centroid_std"] + df["bandwidth_std"] + df["rolloff_std"]
mfcc_std_cols = [f"mfcc_{i+1}_std" for i in range(13)]
df["mfcc_irregularity_index"] = df[mfcc_std_cols].mean(axis=1)
df["avhi"] = (
    0.30 * df["perturbation_index"] +
    0.25 * df["pitch_std"] +
    0.20 * df["spectral_irregularity_index"] +
    0.15 * df["mfcc_irregularity_index"] -
    0.10 * df["hnr"]
)
df["pitch_jitter_interaction"] = df["pitch_std"] * df["jitter"]
df["pitch_shimmer_interaction"] = df["pitch_std"] * df["shimmer"]

exclude_cols = {"file_path", "disease", "patient_id", "file_name", "label"}
feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
y = df["label"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}", flush=True)

# Define candidates
candidates = {
    "HistGradientBoosting": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=15, random_state=42))
    ]),
    "RandomForest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_split=2, random_state=42, n_jobs=-1))
    ]),
    "ExtraTrees": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", ExtraTreesClassifier(n_estimators=300, max_depth=None, min_samples_split=2, random_state=42, n_jobs=-1))
    ]),
    "XGBoost": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1))
    ]),
}

for name, pipe in candidates.items():
    print(f"\nTraining {name}...", flush=True)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_prob)
    print(f"[{name}] Acc: {acc*100:.2f}%, Prec: {prec*100:.2f}%, Rec: {rec*100:.2f}%, F1: {f1*100:.2f}%, ROC-AUC: {roc*100:.2f}%", flush=True)

# Also test Hybrid Ensemble (VotingClassifier)
print("\nTraining AVHFM Hybrid Ensemble (VotingClassifier)...", flush=True)
ensemble = VotingClassifier(
    estimators=[
        ("rf", candidates["RandomForest"]),
        ("et", candidates["ExtraTrees"]),
        ("xgb", candidates["XGBoost"]),
        ("hgb", candidates["HistGradientBoosting"]),
    ],
    voting="soft"
)
ensemble.fit(X_train, y_train)
y_pred_ens = ensemble.predict(X_test)
y_prob_ens = ensemble.predict_proba(X_test)[:, 1]
acc_e = accuracy_score(y_test, y_pred_ens)
prec_e = precision_score(y_test, y_pred_ens, zero_division=0)
rec_e = recall_score(y_test, y_pred_ens, zero_division=0)
f1_e = f1_score(y_test, y_pred_ens, zero_division=0)
roc_e = roc_auc_score(y_test, y_prob_ens)
print(f"[AVHFM Hybrid Ensemble] Acc: {acc_e*100:.2f}%, Prec: {prec_e*100:.2f}%, Rec: {rec_e*100:.2f}%, F1: {f1_e*100:.2f}%, ROC-AUC: {roc_e*100:.2f}%", flush=True)

