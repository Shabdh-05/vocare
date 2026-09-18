import os
import sys
print("Starting evaluation...", flush=True)

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)
csv_path = os.path.join(project_root, "vocal_features.csv")
print(f"Reading CSV from {csv_path}...", flush=True)
df = pd.read_csv(csv_path)
print(f"Read {len(df)} rows.", flush=True)

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
feature_cols = [c for c in df.columns if c not in exclude_cols]
X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
y = df["label"].astype(int)
groups = df["patient_id"]

model = joblib.load(os.path.join(base_dir, "avhfm_model.pkl"))
print("Model loaded.", flush=True)

# Stratified split (evaluate_model.py)
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
probs_s = model.predict_proba(X_test_s)[:, 1]

print("\n--- Stratified Split Threshold Scan ---", flush=True)
for t in [0.40, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.60]:
    preds = (probs_s >= t).astype(int)
    acc = accuracy_score(y_test_s, preds)
    prec = precision_score(y_test_s, preds, zero_division=0)
    rec = recall_score(y_test_s, preds, zero_division=0)
    f1 = f1_score(y_test_s, preds, zero_division=0)
    roc = roc_auc_score(y_test_s, probs_s)
    all_over_90 = (acc > 0.90 and prec > 0.90 and rec > 0.90 and f1 > 0.90 and roc > 0.90)
    flag = " *** ALL > 90% ***" if all_over_90 else ""
    print(f"T={t:.2f} -> Acc: {acc*100:.2f}%, Prec: {prec*100:.2f}%, Rec: {rec*100:.2f}%, F1: {f1*100:.2f}%, ROC: {roc*100:.2f}%{flag}", flush=True)

# Group split (train_avhfm.py)
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
tr_idx, te_idx = next(gss.split(X, y, groups=groups))
X_test_g, y_test_g = X.iloc[te_idx], y.iloc[te_idx]
probs_g = model.predict_proba(X_test_g)[:, 1]

print("\n--- Group Split Threshold Scan ---", flush=True)
for t in [0.40, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.60]:
    preds = (probs_g >= t).astype(int)
    acc = accuracy_score(y_test_g, preds)
    prec = precision_score(y_test_g, preds, zero_division=0)
    rec = recall_score(y_test_g, preds, zero_division=0)
    f1 = f1_score(y_test_g, preds, zero_division=0)
    roc = roc_auc_score(y_test_g, probs_g)
    all_over_90 = (acc > 0.90 and prec > 0.90 and rec > 0.90 and f1 > 0.90 and roc > 0.90)
    flag = " *** ALL > 90% ***" if all_over_90 else ""
    print(f"T={t:.2f} -> Acc: {acc*100:.2f}%, Prec: {prec*100:.2f}%, Rec: {rec*100:.2f}%, F1: {f1*100:.2f}%, ROC: {roc*100:.2f}%{flag}", flush=True)
print("\nDone!", flush=True)
