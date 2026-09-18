import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

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
    roc_auc_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")


# ============================================================
# ENHANCED AVHFM V4
# MULTI-TASK AVHFM FUSION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.path.join(
    BASE_DIR,
    "..",
    "vocal_features.csv"
)

print("=" * 78)
print("ENHANCED AVHFM V4 - MULTI-TASK AVHFM FUSION")
print("=" * 78)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n[1/9] Loading dataset...")

df = pd.read_csv(CSV_PATH)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# 2. IDENTIFY VOCAL TASK
# ============================================================

print("\n[2/9] Identifying vocal tasks...")


def get_task(filename):

    name = str(filename).lower()

    name = os.path.splitext(name)[0]

    parts = name.split("-")

    if len(parts) >= 2:
        return parts[1]

    return "unknown"


df["task"] = df["file_name"].apply(get_task)


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
    "u_n"
]


print("\nTask counts:")

for task in TASKS:

    count = (
        df["task"] == task
    ).sum()

    print(
        f"  {task:6s}: {count:,}"
    )


# ============================================================
# 3. CREATE AVHFM FEATURES
# ============================================================

print("\n[3/9] Creating AVHFM features...")

EPS = 1e-8


# ------------------------------------------------------------
# ORIGINAL AVHFM FEATURES
# ------------------------------------------------------------

df["vocal_stability_index"] = (
    df["hnr"]
    /
    (
        df["pitch_std"]
        + df["jitter"]
        + df["shimmer"]
        + EPS
    )
)


df["perturbation_index"] = (
    df["jitter"]
    + df["shimmer"]
)


df["phonation_energy_index"] = (
    df["rms_mean"]
    /
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
    df["pitch_std"]
    * df["jitter"]
)


df["pitch_shimmer_interaction"] = (
    df["pitch_std"]
    * df["shimmer"]
)


# ------------------------------------------------------------
# ENHANCED AVHFM FEATURES
# ------------------------------------------------------------

df["hnr_jitter_ratio"] = (
    df["hnr"]
    / (df["jitter"] + EPS)
)


df["hnr_shimmer_ratio"] = (
    df["hnr"]
    / (df["shimmer"] + EPS)
)


df["pitch_variability_ratio"] = (
    df["pitch_std"]
    /
    (
        df["pitch_mean"].abs()
        + EPS
    )
)


df["spectral_variability_ratio"] = (
    df["spectral_irregularity_index"]
    /
    (
        df["centroid_mean"].abs()
        + EPS
    )
)


df["mfcc_variability_ratio"] = (
    df["mfcc_irregularity_index"]
    /
    (
        df["rms_mean"].abs()
        + EPS
    )
)


df["jitter_shimmer_ratio"] = (
    df["jitter"]
    /
    (
        df["shimmer"]
        + EPS
    )
)


df["voice_quality_index"] = (
    df["hnr"]
    /
    (
        1
        + df["jitter"]
        + df["shimmer"]
        + df["pitch_std"]
    )
)


df["perturbation_pitch_ratio"] = (
    df["perturbation_index"]
    /
    (
        df["pitch_mean"].abs()
        + EPS
    )
)


# ============================================================
# 4. DEFINE AVHFM FEATURE SET
# ============================================================

DROP_COLUMNS = [
    "file_path",
    "disease",
    "patient_id",
    "file_name",
    "label",
    "task"
]


FEATURE_COLUMNS = [
    column
    for column in df.columns
    if column not in DROP_COLUMNS
]


df[FEATURE_COLUMNS] = (
    df[FEATURE_COLUMNS]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
)


print(
    f"\nAVHFM feature count: "
    f"{len(FEATURE_COLUMNS)}"
)


# ============================================================
# 5. PATIENT/TASK MATRIX
# ============================================================

print(
    "\n[4/9] Constructing patient-level "
    "multi-task AVHFM matrix..."
)


patient_labels = (
    df.groupby("patient_id")["label"]
    .first()
)


patients = (
    patient_labels.index
    .to_numpy()
)


print(
    f"Patients available: "
    f"{len(patients):,}"
)


patient_task_counts = (
    df.groupby("patient_id")["task"]
    .nunique()
)


complete_patients = (
    patient_task_counts[
        patient_task_counts == len(TASKS)
    ]
    .index
    .to_numpy()
)


print(
    f"Patients with all 13 tasks: "
    f"{len(complete_patients):,}"
)


patient_labels_complete = (
    patient_labels.loc[
        complete_patients
    ]
)


# ============================================================
# PATIENT-LEVEL TRAIN/TEST SPLIT
# ============================================================

train_patients, test_patients = train_test_split(

    complete_patients,

    test_size=0.20,

    stratify=patient_labels_complete.values,

    random_state=42
)


print(
    f"Training patients: "
    f"{len(train_patients):,}"
)


print(
    f"Testing patients : "
    f"{len(test_patients):,}"
)


overlap = (
    set(train_patients)
    &
    set(test_patients)
)


print(
    f"Patient overlap   : "
    f"{len(overlap)}"
)


# ============================================================
# 6. BUILD PATIENT-LEVEL AVHFM MATRIX
# ============================================================

print(
    "\n[5/9] Aggregating AVHFM features "
    "for every vocal task..."
)


def build_patient_matrix(patient_list):

    patient_set = set(
        patient_list
    )

    subset = df[
        df["patient_id"].isin(
            patient_set
        )
    ].copy()

    output = {}

    for patient_id, patient_group in (
        subset.groupby("patient_id")
    ):

        row = {}

        for task in TASKS:

            task_data = patient_group[
                patient_group["task"] == task
            ]

            if len(task_data) == 0:
                continue

            for feature in FEATURE_COLUMNS:

                values = pd.to_numeric(
                    task_data[feature],
                    errors="coerce"
                )

                values = values.replace(
                    [np.inf, -np.inf],
                    np.nan
                )

                mean_value = values.mean()

                std_value = values.std(
                    ddof=0
                )

                row[
                    f"{task}__{feature}__mean"
                ] = mean_value

                row[
                    f"{task}__{feature}__std"
                ] = std_value

        output[patient_id] = row

    result = pd.DataFrame.from_dict(
        output,
        orient="index"
    )

    result.index.name = "patient_id"

    return result


train_matrix = build_patient_matrix(
    train_patients
)


test_matrix = build_patient_matrix(
    test_patients
)


print(
    f"Training matrix shape: "
    f"{train_matrix.shape}"
)


print(
    f"Testing matrix shape : "
    f"{test_matrix.shape}"
)


# ============================================================
# 7. CROSS-TASK AVHFM FEATURES
# ============================================================

print(
    "\n[6/9] Creating cross-task "
    "AVHFM features..."
)


def add_cross_task_features(matrix):

    matrix = matrix.copy()

    vowel_groups = {

        "a": [
            "a_h",
            "a_l",
            "a_lhl",
            "a_n"
        ],

        "i": [
            "i_h",
            "i_l",
            "i_lhl",
            "i_n"
        ],

        "u": [
            "u_h",
            "u_l",
            "u_lhl",
            "u_n"
        ]
    }


    # --------------------------------------------------------
    # VOWEL-GROUP DIFFERENCES
    # --------------------------------------------------------

    for feature in FEATURE_COLUMNS:

        group_values = {}

        for group_name, tasks in (
            vowel_groups.items()
        ):

            columns = [
                f"{task}__{feature}__mean"
                for task in tasks
                if (
                    f"{task}__{feature}__mean"
                    in matrix.columns
                )
            ]

            if len(columns) > 0:

                group_values[group_name] = (
                    matrix[columns]
                    .mean(axis=1)
                )


        if (
            "a" in group_values
            and "i" in group_values
        ):

            matrix[
                f"cross_ai__{feature}"
            ] = (
                group_values["a"]
                -
                group_values["i"]
            )


        if (
            "a" in group_values
            and "u" in group_values
        ):

            matrix[
                f"cross_au__{feature}"
            ] = (
                group_values["a"]
                -
                group_values["u"]
            )


        if (
            "i" in group_values
            and "u" in group_values
        ):

            matrix[
                f"cross_iu__{feature}"
            ] = (
                group_values["i"]
                -
                group_values["u"]
            )


    # --------------------------------------------------------
    # IMPORTANT TASK CONTRASTS
    # --------------------------------------------------------

    important_features = [

        "jitter",
        "shimmer",
        "hnr",
        "pitch_mean",
        "pitch_std",
        "rms_mean",
        "centroid_mean",
        "bandwidth_mean",
        "rolloff_mean",
        "perturbation_index",
        "vocal_stability_index",
        "avhi",
        "pitch_jitter_interaction",
        "pitch_shimmer_interaction"

    ]


    important_comparisons = [

        (
            "a_lhl",
            "u_lhl",
            "a_lhl_vs_u_lhl"
        ),

        (
            "a_n",
            "u_n",
            "a_n_vs_u_n"
        ),

        (
            "i_lhl",
            "u_lhl",
            "i_lhl_vs_u_lhl"
        ),

        (
            "a_h",
            "i_h",
            "a_h_vs_i_h"
        ),

        (
            "a_l",
            "i_l",
            "a_l_vs_i_l"
        )

    ]


    for feature in important_features:

        for task1, task2, comparison_name in (
            important_comparisons
        ):

            column1 = (
                f"{task1}__{feature}__mean"
            )

            column2 = (
                f"{task2}__{feature}__mean"
            )


            if (
                column1 in matrix.columns
                and
                column2 in matrix.columns
            ):

                matrix[
                    f"{comparison_name}__{feature}"
                ] = (
                    matrix[column1]
                    -
                    matrix[column2]
                )


    return matrix


train_matrix = add_cross_task_features(
    train_matrix
)


test_matrix = add_cross_task_features(
    test_matrix
)


# Ensure exactly the same columns
test_matrix = test_matrix.reindex(
    columns=train_matrix.columns
)


print(
    f"Final multi-task AVHFM features: "
    f"{train_matrix.shape[1]:,}"
)


# ============================================================
# 8. PREPARE MODEL DATA
# ============================================================

print(
    "\n[7/9] Preparing model data..."
)


X_train = train_matrix.copy()

X_test = test_matrix.copy()


y_train = (
    patient_labels.loc[
        train_patients
    ]
    .astype(int)
)


y_test = (
    patient_labels.loc[
        test_patients
    ]
    .astype(int)
)


# Make sure indexes align
X_train = X_train.reindex(
    index=train_patients
)

X_test = X_test.reindex(
    index=test_patients
)


imputer = SimpleImputer(
    strategy="median"
)


X_train_imp = (
    imputer.fit_transform(
        X_train
    )
)


X_test_imp = (
    imputer.transform(
        X_test
    )
)


print(
    f"Training samples: "
    f"{X_train_imp.shape[0]:,}"
)


print(
    f"Training features: "
    f"{X_train_imp.shape[1]:,}"
)


# ============================================================
# 9. TRAIN MULTIPLE AVHFM MODELS
# ============================================================

print(
    "\n[8/9] Training multi-task "
    "AVHFM models..."
)


models = {

    "ExtraTrees":
        ExtraTreesClassifier(

            n_estimators=1000,

            max_features="sqrt",

            min_samples_split=2,

            min_samples_leaf=1,

            class_weight="balanced",

            random_state=42,

            n_jobs=-1
        ),


    "RandomForest":
        RandomForestClassifier(

            n_estimators=1000,

            max_features="sqrt",

            min_samples_split=2,

            min_samples_leaf=1,

            class_weight="balanced",

            random_state=42,

            n_jobs=-1
        ),


    "HistGradientBoosting":
        HistGradientBoostingClassifier(

            max_iter=500,

            learning_rate=0.035,

            max_leaf_nodes=31,

            min_samples_leaf=15,

            l2_regularization=1.0,

            random_state=42
        )

}


model_probabilities = {}

model_results = []


for name, model in models.items():

    print(
        f"\nTraining {name}..."
    )


    model.fit(
        X_train_imp,
        y_train
    )


    prediction = model.predict(
        X_test_imp
    )


    probability = model.predict_proba(
        X_test_imp
    )[:, 1]


    model_probabilities[name] = (
        probability
    )


    accuracy = accuracy_score(
        y_test,
        prediction
    )


    precision = precision_score(
        y_test,
        prediction,
        zero_division=0
    )


    recall = recall_score(
        y_test,
        prediction,
        zero_division=0
    )


    f1 = f1_score(
        y_test,
        prediction,
        zero_division=0
    )


    auc = roc_auc_score(
        y_test,
        probability
    )


    model_results.append({

        "Model": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1": f1,

        "ROC_AUC": auc

    })


    print(
        f"Accuracy : "
        f"{accuracy * 100:.2f}%"
    )


    print(
        f"Precision: "
        f"{precision * 100:.2f}%"
    )


    print(
        f"Recall   : "
        f"{recall * 100:.2f}%"
    )


    print(
        f"F1       : "
        f"{f1 * 100:.2f}%"
    )


    print(
        f"ROC-AUC  : "
        f"{auc * 100:.2f}%"
    )


# ============================================================
# AVHFM WEIGHTED MULTI-TASK ENSEMBLE
# ============================================================

print(
    "\n[9/9] Building Multi-Task "
    "AVHFM ensemble..."
)


ensemble_probability = (

    0.40
    * model_probabilities[
        "ExtraTrees"
    ]

    +

    0.35
    * model_probabilities[
        "RandomForest"
    ]

    +

    0.25
    * model_probabilities[
        "HistGradientBoosting"
    ]

)


ensemble_prediction = (
    ensemble_probability >= 0.50
).astype(int)


ensemble_accuracy = (
    accuracy_score(
        y_test,
        ensemble_prediction
    )
)


ensemble_precision = (
    precision_score(
        y_test,
        ensemble_prediction,
        zero_division=0
    )
)


ensemble_recall = (
    recall_score(
        y_test,
        ensemble_prediction,
        zero_division=0
    )
)


ensemble_f1 = (
    f1_score(
        y_test,
        ensemble_prediction,
        zero_division=0
    )
)


ensemble_auc = (
    roc_auc_score(
        y_test,
        ensemble_probability
    )
)


model_results.append({

    "Model":
        "AVHFM V4 Multi-Task Ensemble",

    "Accuracy":
        ensemble_accuracy,

    "Precision":
        ensemble_precision,

    "Recall":
        ensemble_recall,

    "F1":
        ensemble_f1,

    "ROC_AUC":
        ensemble_auc

})


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 78)
print("FINAL AVHFM V4 MULTI-TASK RESULTS")
print("=" * 78)


print(
    f"Accuracy : "
    f"{ensemble_accuracy * 100:.2f}%"
)


print(
    f"Precision: "
    f"{ensemble_precision * 100:.2f}%"
)


print(
    f"Recall   : "
    f"{ensemble_recall * 100:.2f}%"
)


print(
    f"F1       : "
    f"{ensemble_f1 * 100:.2f}%"
)


print(
    f"ROC-AUC  : "
    f"{ensemble_auc * 100:.2f}%"
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


# ============================================================
# MODEL COMPARISON
# ============================================================

print("\nModel comparison:")


results_df = pd.DataFrame(
    model_results
)


print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_path = os.path.join(
    BASE_DIR,
    "avhfm_v4_results.csv"
)


results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# SAVE MODEL ARTIFACT
# ============================================================

artifact = {

    "models":
        models,

    "imputer":
        imputer,

    "feature_columns":
        list(
            train_matrix.columns
        ),

    "tasks":
        TASKS,

    "ensemble_weights": {

        "ExtraTrees":
            0.40,

        "RandomForest":
            0.35,

        "HistGradientBoosting":
            0.25

    },

    "threshold":
        0.50,

    "description":
        "Enhanced AVHFM multi-task patient-level model"

}


artifact_path = os.path.join(
    BASE_DIR,
    "avhfm_v4_multitask.pkl"
)


joblib.dump(
    artifact,
    artifact_path
)


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 78)
print("FILES SAVED")
print("=" * 78)

print(
    results_path
)

print(
    artifact_path
)

print(
    "\nAVHFM V4 COMPLETE"
)

print("=" * 78)