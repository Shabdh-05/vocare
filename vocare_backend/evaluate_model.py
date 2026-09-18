import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def prepare_dataset():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    csv_path = os.path.join(project_root, "vocal_features.csv")

    print(f"\nLoading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total samples loaded: {len(df):,}")

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

    return X_train, X_test, y_train, y_test, feature_cols, base_dir


def evaluate_saved_model(X_test, y_test, base_dir):
    model_path = os.path.join(base_dir, "avhfm_model.pkl")
    if not os.path.exists(model_path):
        print(f"ERROR: Saved model not found at {model_path}. Please run train_avhfm.py first.")
        return

    print(f"Loading saved AVHFM model from: {model_path}")
    pipeline = joblib.load(model_path)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else float("nan")
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 60)
    print("           AVHFM MODEL EVALUATION METRICS               ")
    print("=" * 60)
    print(f"  Accuracy              : {acc * 100:.2f}%")
    print(f"  Precision             : {prec * 100:.2f}%")
    print(f"  Recall (Sensitivity)  : {rec * 100:.2f}%")
    print(f"  F1 Score              : {f1 * 100:.2f}%")
    if not np.isnan(roc_auc):
        print(f"  ROC-AUC Score         : {roc_auc:.4f}")
    print("=" * 60)

    print("\nDetailed Classification Report:")
    print("-" * 60)
    print(classification_report(y_test, y_pred, target_names=["Healthy (0)", "Vocal Disorder (1)"], digits=4))

    print("Confusion Matrix:")
    print("-" * 60)
    print(f"                     Predicted Healthy    Predicted Disorder")
    print(f"  Actual Healthy          {cm[0, 0]:<20} {cm[0, 1]:<18} (TN={cm[0, 0]}, FP={cm[0, 1]})")
    print(f"  Actual Disorder         {cm[1, 0]:<20} {cm[1, 1]:<18} (FN={cm[1, 0]}, TP={cm[1, 1]})")
    print("=" * 60 + "\n")


def compare_all_models(X_train, X_test, y_train, y_test, base_dir):
    print("\nRunning Model Comparison (as documented in README)...")
    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Support Vector Machine": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", SVC(probability=True, random_state=42))
        ]),
        "Random Forest Baseline": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
        ]),
        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", GradientBoostingClassifier(random_state=42))
        ]),
    }

    results = []

    # First add saved AVHFM
    model_path = os.path.join(base_dir, "avhfm_model.pkl")
    if os.path.exists(model_path):
        avhfm = joblib.load(model_path)
        y_pred = avhfm.predict(X_test)
        y_prob = avhfm.predict_proba(X_test)[:, 1]
        results.append({
            "Model": "AVHFM (Final Model)",
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_prob)
        })

    for name, pipe in models.items():
        print(f"   Training & evaluating {name}...")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_prob)
        })

    df_res = pd.DataFrame(results)
    print("\n" + "=" * 78)
    print("                         MODEL COMPARISON TABLE                           ")
    print("=" * 78)
    print(f"{'Model':<26} {'Accuracy':<11} {'Precision':<11} {'Recall':<10} {'F1 Score':<11} {'ROC-AUC':<8}")
    print("-" * 78)
    for _, row in df_res.iterrows():
        print(f"{row['Model']:<26} {row['Accuracy']*100:>8.2f}%  {row['Precision']*100:>8.2f}%  {row['Recall']*100:>7.2f}%  {row['F1 Score']*100:>8.2f}%  {row['ROC-AUC']:>7.4f}")
    print("=" * 78 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Vocal Disorder Detection models")
    parser.add_argument("--compare", action="store_true", help="Compare AVHFM against baseline models (LR, SVM, RF, GB)")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test, feature_cols, base_dir = prepare_dataset()

    evaluate_saved_model(X_test, y_test, base_dir)

    if args.compare:
        compare_all_models(X_train, X_test, y_train, y_test, base_dir)


if __name__ == "__main__":
    main()
