import os
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(
    BASE_DIR,
    "..",
    "vocal_features.csv"
)

# All 13 tasks confirmed in your dataset
TASKS = [
    "a_h",
    "a_l",
    "a_lhl",
    "a_n",
    "i_h",
    "i_l",
    "i_lhl",
    "i_n",
    "iau",
    "u_h",
    "u_l",
    "u_lhl",
    "u_n",
]


# ============================================================
# AVHFM FEATURE CREATION
# ============================================================

def create_avhfm_features(df):

    df = df.copy()

    eps = 1e-8

    df["vocal_stability_index"] = (
        df["hnr"]
        /
        (
            df["pitch_std"]
            + df["jitter"]
            + df["shimmer"]
            + eps
        )
    )

    df["perturbation_index"] = (
        df["jitter"]
        + df["shimmer"]
    )

    df["phonation_energy_index"] = (
        df["rms_mean"]
        /
        (
            df["zcr_mean"]
            + eps
        )
    )

    df["spectral_irregularity_index"] = (
        df["centroid_std"]
        + df["bandwidth_std"]
        + df["rolloff_std"]
    )

    mfcc_std_cols = [
        f"mfcc_{i}_std"
        for i in range(1, 14)
        if f"mfcc_{i}_std" in df.columns
    ]

    df["mfcc_irregularity_index"] = (
        df[mfcc_std_cols].mean(axis=1)
    )

    df["avhi"] = (
        0.30 * df["perturbation_index"]
        + 0.25 * df["pitch_std"]
        + 0.20 * df["spectral_irregularity_index"]
        + 0.15 * df["mfcc_irregularity_index"]
        - 0.10 * df["hnr"]
    )

    df["pitch_jitter_interaction"] = (
        df["pitch_std"]
        * df["jitter"]
    )

    df["pitch_shimmer_interaction"] = (
        df["pitch_std"]
        * df["shimmer"]
    )

    df["hnr_jitter_ratio"] = (
        df["hnr"]
        /
        (
            df["jitter"]
            + eps
        )
    )

    df["hnr_shimmer_ratio"] = (
        df["hnr"]
        /
        (
            df["shimmer"]
            + eps
        )
    )

    df["voice_quality_index"] = (
        df["hnr"]
        /
        (
            df["jitter"]
            + df["shimmer"]
            + df["zcr_mean"]
            + eps
        )
    )

    return df


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("13-TASK RANDOM FOREST PATIENT FUSION")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(CSV_PATH)

print("Rows:", len(df))
print("Patients:", df["patient_id"].nunique())

df = create_avhfm_features(df)

# ============================================================
# DETERMINE TASK
# ============================================================

# Your confirmed filenames are like:
# 1242-a_h-egg.wav

df["task"] = (
    df["file_name"]
    .astype(str)
    .str.split("-")
    .str[1]
)

print("\nTask counts:")

print(
    df["task"]
    .value_counts()
    .sort_index()
)

# ============================================================
# FEATURE COLUMNS
# ============================================================

EXCLUDE = {
    "file_path",
    "disease",
    "patient_id",
    "file_name",
    "label",
    "task",
}

feature_columns = [
    c
    for c in df.columns
    if c not in EXCLUDE
]

print(
    "\nFeatures per task:",
    len(feature_columns)
)

# ============================================================
# CLEAN NUMERIC VALUES
# ============================================================

df[feature_columns] = df[
    feature_columns
].replace(
    [np.inf, -np.inf],
    np.nan
)

# ============================================================
# PATIENT SPLIT
# ============================================================

patients = (
    df[
        [
            "patient_id",
            "label"
        ]
    ]
    .drop_duplicates()
)

# Make sure every patient has one label
label_counts = (
    patients
    .groupby("patient_id")["label"]
    .nunique()
)

if label_counts.max() > 1:

    raise ValueError(
        "A patient has conflicting labels."
    )

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=SEED
)

patient_ids = patients["patient_id"].values
patient_labels = patients["label"].values

train_pat_idx, test_pat_idx = next(
    splitter.split(
        patient_ids,
        patient_labels,
        patient_ids
    )
)

train_patients = set(
    patient_ids[train_pat_idx]
)

test_patients = set(
    patient_ids[test_pat_idx]
)

print("\nPatient split:")
print(
    "Train patients:",
    len(train_patients)
)
print(
    "Test patients:",
    len(test_patients)
)
print(
    "Patient overlap:",
    len(
        train_patients
        & test_patients
    )
)

# ============================================================
# BUILD TASK MODELS
# ============================================================

task_models = {}
task_imputers = {}

task_test_probabilities = {}
task_test_labels = {}

print("\n" + "=" * 70)
print("TRAINING TASK-SPECIFIC RANDOM FORESTS")
print("=" * 70)

for task in TASKS:

    print(
        f"\n[{task}]"
    )

    task_df = df[
        df["task"] == task
    ].copy()

    # --------------------------------------------------------
    # Keep only patients belonging to the global split
    # --------------------------------------------------------

    train_task = task_df[
        task_df["patient_id"].isin(
            train_patients
        )
    ].copy()

    test_task = task_df[
        task_df["patient_id"].isin(
            test_patients
        )
    ].copy()

    print(
        "Train:",
        len(train_task),
        "| Test:",
        len(test_task)
    )

    if len(train_task) == 0:
        print("SKIPPED")
        continue

    if len(test_task) == 0:
        print("SKIPPED")
        continue

    X_train = train_task[
        feature_columns
    ]

    y_train = train_task[
        "label"
    ].astype(int)

    X_test = test_task[
        feature_columns
    ]

    y_test = test_task[
        "label"
    ].astype(int)

    # --------------------------------------------------------
    # Imputation
    # --------------------------------------------------------

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train = imputer.fit_transform(
        X_train
    )

    X_test = imputer.transform(
        X_test
    )

    # --------------------------------------------------------
    # RF
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=700,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight=None,
        random_state=SEED,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    probability = model.predict_proba(
        X_test
    )[:, 1]

    task_models[task] = model
    task_imputers[task] = imputer

    task_test_probabilities[task] = (
        pd.Series(
            probability,
            index=test_task["patient_id"].values
        )
    )

    task_test_labels[task] = (
        pd.Series(
            y_test.values,
            index=test_task["patient_id"].values
        )
    )


# ============================================================
# PATIENT-LEVEL FUSION
# ============================================================

print("\n" + "=" * 70)
print("PATIENT-LEVEL PROBABILITY FUSION")
print("=" * 70)

# ------------------------------------------------------------
# Build matrix:
#
# rows    = patients
# columns = 13 task probabilities
# ------------------------------------------------------------

test_patient_ids = sorted(
    test_patients
)

probability_matrix = pd.DataFrame(
    index=test_patient_ids
)

for task in TASKS:

    if task not in task_test_probabilities:
        continue

    series = task_test_probabilities[
        task
    ]

    # If a patient has multiple entries for a task,
    # average them.
    series = series.groupby(
        level=0
    ).mean()

    probability_matrix[
        task
    ] = series

print(
    "\nPatient probability matrix:",
    probability_matrix.shape
)

# ============================================================
# TRUE PATIENT LABELS
# ============================================================

patient_label_map = (
    patients
    .set_index("patient_id")["label"]
    .to_dict()
)

y_patient = np.array(
    [
        patient_label_map[p]
        for p in probability_matrix.index
    ]
)

# ============================================================
# HANDLE MISSING TASKS
# ============================================================

probability_matrix = probability_matrix.fillna(
    probability_matrix.mean()
)

# ============================================================
# SIMPLE FUSION METHODS
# ============================================================

fusion_results = {}

# Equal mean
equal_probability = (
    probability_matrix.mean(
        axis=1
    ).values
)

fusion_results[
    "Equal Mean"
] = equal_probability

# Median
median_probability = (
    probability_matrix.median(
        axis=1
    ).values
)

fusion_results[
    "Median"
] = median_probability

# ------------------------------------------------------------
# Performance of each fusion method
# ------------------------------------------------------------

for name, probabilities in fusion_results.items():

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    accuracy = accuracy_score(
        y_patient,
        predictions
    )

    precision = precision_score(
        y_patient,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_patient,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_patient,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        y_patient,
        probabilities
    )

    print(
        f"\n{name}"
    )

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
        f"F1 Score : {f1 * 100:.2f}%"
    )

    print(
        f"ROC-AUC  : {auc * 100:.2f}%"
    )


# ============================================================
# LEARN TASK WEIGHTS ON TRAINING PATIENTS ONLY
# ============================================================

print("\n" + "=" * 70)
print("LEARNED TASK WEIGHT FUSION")
print("=" * 70)

# Build training task probability matrix.
#
# We train the individual task models on the training patients,
# then obtain their TRAINING probabilities.
#
# These probabilities are used ONLY to learn fusion weights.

train_probability_matrix = pd.DataFrame(
    index=sorted(train_patients)
)

for task in TASKS:

    task_df = df[
        df["task"] == task
    ].copy()

    train_task = task_df[
        task_df["patient_id"].isin(
            train_patients
        )
    ].copy()

    if len(train_task) == 0:
        continue

    X_train = train_task[
        feature_columns
    ]

    y_train = train_task[
        "label"
    ].astype(int)

    # Refit an independent model for obtaining
    # training probabilities.
    #
    # This is only for constructing a fusion signal;
    # the final evaluation remains on the untouched test patients.

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train_imp = imputer.fit_transform(
        X_train
    )

    fusion_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=SEED,
        n_jobs=-1
    )

    fusion_model.fit(
        X_train_imp,
        y_train
    )

    train_probability = (
        fusion_model
        .predict_proba(
            X_train_imp
        )[:, 1]
    )

    series = pd.Series(
        train_probability,
        index=train_task[
            "patient_id"
        ].values
    )

    series = series.groupby(
        level=0
    ).mean()

    train_probability_matrix[
        task
    ] = series

train_probability_matrix = (
    train_probability_matrix
    .reindex(
        sorted(train_patients)
    )
)

train_probability_matrix = (
    train_probability_matrix.fillna(
        train_probability_matrix.mean()
    )
)

y_train_patient = np.array(
    [
        patient_label_map[p]
        for p in train_probability_matrix.index
    ]
)

# ============================================================
# WEIGHT SEARCH
# ============================================================

available_tasks = [
    t
    for t in TASKS
    if t in train_probability_matrix.columns
    and t in probability_matrix.columns
]

print(
    "\nAvailable tasks:",
    available_tasks
)

# Correlation-based initial weights.
# We only use training data here.

weights = {}

for task in available_tasks:

    x = train_probability_matrix[
        task
    ].values

    y = y_train_patient

    if np.std(x) == 0:
        score = 0.0
    else:
        score = abs(
            np.corrcoef(
                x,
                y
            )[0, 1]
        )

        if not np.isfinite(score):
            score = 0.0

    weights[task] = score

weight_sum = sum(
    weights.values()
)

if weight_sum == 0:

    weights = {
        task: 1.0
        for task in available_tasks
    }

    weight_sum = sum(
        weights.values()
    )

weights = {
    task: weights[task] / weight_sum
    for task in available_tasks
}

print("\nLearned task weights:")

for task in available_tasks:

    print(
        f"{task}: {weights[task]:.4f}"
    )

# ============================================================
# WEIGHTED TEST FUSION
# ============================================================

weighted_test_probability = np.zeros(
    len(probability_matrix)
)

for task in available_tasks:

    weighted_test_probability += (
        probability_matrix[task].values
        * weights[task]
    )

# ============================================================
# THRESHOLD SEARCH USING TRAINING DATA
# ============================================================

weighted_train_probability = np.zeros(
    len(train_probability_matrix)
)

for task in available_tasks:

    weighted_train_probability += (
        train_probability_matrix[
            task
        ].values
        * weights[task]
    )

best_threshold = 0.50
best_train_f1 = -1

for threshold in np.arange(
    0.30,
    0.71,
    0.01
):

    train_pred = (
        weighted_train_probability
        >= threshold
    ).astype(int)

    score = f1_score(
        y_train_patient,
        train_pred,
        zero_division=0
    )

    if score > best_train_f1:

        best_train_f1 = score
        best_threshold = float(
            threshold
        )

print(
    "\nSelected threshold:",
    round(best_threshold, 2)
)

print(
    "Training fusion F1:",
    round(
        best_train_f1 * 100,
        2
    ),
    "%"
)

# ============================================================
# FINAL TEST
# ============================================================

final_prediction = (
    weighted_test_probability
    >= best_threshold
).astype(int)

accuracy = accuracy_score(
    y_patient,
    final_prediction
)

precision = precision_score(
    y_patient,
    final_prediction,
    zero_division=0
)

recall = recall_score(
    y_patient,
    final_prediction,
    zero_division=0
)

f1 = f1_score(
    y_patient,
    final_prediction,
    zero_division=0
)

auc = roc_auc_score(
    y_patient,
    weighted_test_probability
)

print("\n" + "=" * 70)
print("FINAL 13-TASK RF FUSION RESULTS")
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
    f"F1 Score : {f1 * 100:.2f}%"
)

print(
    f"ROC-AUC  : {auc * 100:.2f}%"
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_patient,
        final_prediction
    )
)

print("=" * 70)

print(
    "\nExperiment completed."
)   