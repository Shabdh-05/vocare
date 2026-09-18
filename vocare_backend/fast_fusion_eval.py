import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)
csv_path = os.path.join(project_root, "vocal_features.csv")

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

# Exact split used in evaluate_model.py
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# 1. Test RandomForest
rf = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clf", RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1))
])
rf.fit(X_train, y_train)
p_rf = rf.predict_proba(X_test)[:, 1]
pred_rf = rf.predict(X_test)
print(f"RF -> Acc: {accuracy_score(y_test, pred_rf)*100:.2f}%, Prec: {precision_score(y_test, pred_rf)*100:.2f}%, Rec: {recall_score(y_test, pred_rf)*100:.2f}%, F1: {f1_score(y_test, pred_rf)*100:.2f}%, AUC: {roc_auc_score(y_test, p_rf)*100:.2f}%")

# 2. Test ExtraTrees
et = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clf", ExtraTreesClassifier(n_estimators=200, max_depth=25, random_state=42, n_jobs=-1))
])
et.fit(X_train, y_train)
p_et = et.predict_proba(X_test)[:, 1]
pred_et = et.predict(X_test)
print(f"ET -> Acc: {accuracy_score(y_test, pred_et)*100:.2f}%, Prec: {precision_score(y_test, pred_et)*100:.2f}%, Rec: {recall_score(y_test, pred_et)*100:.2f}%, F1: {f1_score(y_test, pred_et)*100:.2f}%, AUC: {roc_auc_score(y_test, p_et)*100:.2f}%")

# 3. Test HistGradientBoosting
hgb = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clf", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=45, min_samples_leaf=10, random_state=42))
])
hgb.fit(X_train, y_train)
p_hgb = hgb.predict_proba(X_test)[:, 1]
pred_hgb = hgb.predict(X_test)
print(f"HGB -> Acc: {accuracy_score(y_test, pred_hgb)*100:.2f}%, Prec: {precision_score(y_test, pred_hgb)*100:.2f}%, Rec: {recall_score(y_test, pred_hgb)*100:.2f}%, F1: {f1_score(y_test, pred_hgb)*100:.2f}%, AUC: {roc_auc_score(y_test, p_hgb)*100:.2f}%")

# 4. Test AVHFM Hybrid Fusion (Soft Voting)
p_fusion = (p_rf + p_et + p_hgb) / 3.0
for t in [0.48, 0.50, 0.52]:
    pred_f = (p_fusion >= t).astype(int)
    print(f"Fusion (T={t:.2f}) -> Acc: {accuracy_score(y_test, pred_f)*100:.2f}%, Prec: {precision_score(y_test, pred_f)*100:.2f}%, Rec: {recall_score(y_test, pred_f)*100:.2f}%, F1: {f1_score(y_test, pred_f)*100:.2f}%, AUC: {roc_auc_score(y_test, p_fusion)*100:.2f}%")
