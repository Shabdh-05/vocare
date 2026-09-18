import os
import warnings
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

warnings.filterwarnings("ignore")

# ============================================================
# AVHFM V3 DIAGNOSTIC
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "..", "vocal_features.csv")

print("=" * 75)
print("AVHFM V3 DIAGNOSTIC")
print("=" * 75)

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

print("\n[1/6] Loading dataset...")

df = pd.read_csv(CSV_PATH)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")

# ------------------------------------------------------------
# 2. IDENTIFY VOCAL TASK
# ------------------------------------------------------------

print("\n[2/6] Identifying vocal tasks...")

def get_task(filename):
    """
    Extract task such as:
    a_h, a_l, a_lhl, a_n,
    i_h, i_l, i_lhl, i_n,
    iau,
    u_h, u_l, u_lhl, u_n
    """

    name = str(filename).lower()

    # Remove extension
    name = os.path.splitext(name)[0]

    # Expected filename pattern:
    # patient-task-egg.wav
    parts = name.split("-")

    if len(parts) >= 2:
        return parts[1]

    return "unknown"


df["task"] = df["file_name"].apply(get_task)

print("\nTask distribution:")
print(df["task"].value_counts().sort_index())

# ------------------------------------------------------------
# 3. CREATE ORIGINAL AVHFM FEATURES
# ------------------------------------------------------------

print("\n[3/6] Creating AVHFM features...")

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

# Additional AVHFM features
df["hnr_jitter_ratio"] = (
    df["hnr"] / (df["jitter"] + EPS)
)

df["hnr_shimmer_ratio"] = (
    df["hnr"] / (df["shimmer"] + EPS)
)

df["pitch_variability_ratio"] = (
    df["pitch_std"]
    / (df["pitch_mean"].abs() + EPS)
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
        1
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
# 4. BUILD AVHFM FEATURE LIST
# ------------------------------------------------------------

DROP_COLUMNS = [
    "file_path",
    "disease",
    "patient_id",
    "file_name",
    "label",
    "task"
]

feature_columns = [
    c for c in df.columns
    if c not in DROP_COLUMNS
]

X = df[feature_columns].copy()
y = df["label"].astype(int)

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

# ------------------------------------------------------------
# 5. TRAIN/TEST SPLIT
# ------------------------------------------------------------

print("\n[4/6] Creating diagnostic recording-level split...")

from sklearn.model_selection import train_test_split

train_idx, test_idx = train_test_split(
    np.arange(len(df)),
    test_size=0.20,
    stratify=y,
    random_state=42
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

# ------------------------------------------------------------
# 6. TEST EACH VOCAL TASK
# ------------------------------------------------------------

print("\n[5/6] Evaluating AVHFM separately for each vocal task...")

task_results = []

for task in sorted(df["task"].unique()):

    if task == "unknown":
        continue

    task_mask_train = (
        df.iloc[train_idx]["task"] == task
    )

    task_mask_test = (
        df.iloc[test_idx]["task"] == task
    )

    if task_mask_train.sum() < 50:
        continue

    if task_mask_test.sum() < 20:
        continue

    Xtr = X_train.loc[
        task_mask_train.values
    ]

    Xte = X_test.loc[
        task_mask_test.values
    ]

    ytr = y_train.loc[
        task_mask_train.values
    ]

    yte = y_test.loc[
        task_mask_test.values
    ]

    imputer = SimpleImputer(
        strategy="median"
    )

    Xtr = imputer.fit_transform(Xtr)
    Xte = imputer.transform(Xte)

    model = ExtraTreesClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(Xtr, ytr)

    pred = model.predict(Xte)
    prob = model.predict_proba(Xte)[:, 1]

    acc = accuracy_score(yte, pred)
    prec = precision_score(
        yte,
        pred,
        zero_division=0
    )
    rec = recall_score(
        yte,
        pred,
        zero_division=0
    )
    f1 = f1_score(
        yte,
        pred,
        zero_division=0
    )
    auc = roc_auc_score(
        yte,
        prob
    )

    task_results.append({
        "Task": task,
        "Samples": len(yte),
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC_AUC": auc
    })

    print(
        f"{task:6s} | "
        f"Acc {acc*100:6.2f}% | "
        f"Prec {prec*100:6.2f}% | "
        f"Recall {rec*100:6.2f}% | "
        f"F1 {f1*100:6.2f}% | "
        f"AUC {auc*100:6.2f}%"
    )

# ------------------------------------------------------------
# TASK RESULTS
# ------------------------------------------------------------

task_df = pd.DataFrame(task_results)

task_path = os.path.join(
    BASE_DIR,
    "avhfm_v3_task_results.csv"
)

task_df.to_csv(
    task_path,
    index=False
)

print("\n" + "=" * 75)
print("TASK PERFORMANCE SUMMARY")
print("=" * 75)

if len(task_df) > 0:

    print(
        task_df.sort_values(
            "F1",
            ascending=False
        ).to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\nBest task by F1:")

    best_task = task_df.loc[
        task_df["F1"].idxmax()
    ]

    print(
        f"{best_task['Task']} "
        f"→ F1 = {best_task['F1']*100:.2f}%"
    )

# ------------------------------------------------------------
# PATIENT/TASK INFORMATION
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("PATIENT/TASK STRUCTURE")
print("=" * 75)

patient_task_counts = (
    df.groupby("patient_id")["task"]
    .nunique()
)

print(
    f"Patients: {patient_task_counts.shape[0]:,}"
)

print(
    f"Patients with all 13 tasks: "
    f"{(patient_task_counts == 13).sum():,}"
)

print(
    f"Patients missing at least one task: "
    f"{(patient_task_counts < 13).sum():,}"
)

# ------------------------------------------------------------
# TASK LABEL DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("LABEL DISTRIBUTION BY TASK")
print("=" * 75)

task_label_table = pd.crosstab(
    df["task"],
    df["label"]
)

task_label_table.columns = [
    "Healthy_0",
    "Disorder_1"
]

task_label_table["Total"] = (
    task_label_table["Healthy_0"]
    + task_label_table["Disorder_1"]
)

task_label_table["Disorder_%"] = (
    task_label_table["Disorder_1"]
    / task_label_table["Total"]
    * 100
)

print(
    task_label_table.to_string(
        float_format=lambda x: f"{x:.2f}"
    )
)

# ------------------------------------------------------------
# CORRELATION OF AVHFM FEATURES WITH LABEL
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("TOP AVHFM FEATURES BY LABEL CORRELATION")
print("=" * 75)

numeric_df = df[
    feature_columns + ["label"]
].copy()

correlations = (
    numeric_df.corr(
        numeric_only=True
    )["label"]
    .drop("label")
    .abs()
    .sort_values(
        ascending=False
    )
)

print(
    correlations.head(30).to_string()
)

print("\n" + "=" * 75)
print("AVHFM V3 DIAGNOSTIC COMPLETE")
print("=" * 75)

print("\nSaved:")
print(task_path)