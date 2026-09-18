import pandas as pd
import numpy as np

from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold
)

from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("../vocal_features.csv")

print("Original rows:", len(df))


# ============================================================
# 2. IDENTIFY RECORDING TYPE
# ============================================================

df["recording_type"] = df["file_name"].str.extract(
    r"-(a_h|a_l|a_lhl|a_n|i_h|i_l|i_lhl|i_n|iau|u_h|u_l|u_lhl|u_n)-"
)[0]


# ============================================================
# 3. REMOVE DIAGNOSIS-FOLDER DUPLICATES
# ============================================================

df = df.drop_duplicates(
    subset=["patient_id", "file_name"]
).copy()

print("Rows after deduplication:", len(df))


# ============================================================
# 4. CREATE AVHFM FEATURES
# ============================================================

eps = 1e-8

df["vocal_stability_index"] = (
    df["hnr"]
    / (
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
    / (
        df["zcr_mean"]
        + eps
    )
)

df["spectral_irregularity_index"] = (
    df["centroid_std"]
    + df["bandwidth_std"]
    + df["rolloff_std"]
)

mfcc_std_columns = [
    c
    for c in df.columns
    if c.startswith("mfcc_")
    and c.endswith("_std")
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


# ============================================================
# 5. DEFINE FEATURES
# ============================================================

features = [
    "pitch_mean",
    "pitch_std",
    "pitch_min",
    "pitch_max",
    "jitter",
    "shimmer",
    "hnr",
    "zcr_mean",
    "zcr_std",
    "rms_mean",
    "rms_std",
    "centroid_mean",
    "centroid_std",
    "bandwidth_mean",
    "bandwidth_std",
    "rolloff_mean",
    "rolloff_std",
]


# MFCC features that actually exist
for i in range(1, 14):

    for suffix in [
        "mean",
        "std",
        "delta_mean",
        "delta_std"
    ]:

        column = f"mfcc_{i}_{suffix}"

        if column in df.columns:
            features.append(column)


# AVHFM features
features += [
    "vocal_stability_index",
    "perturbation_index",
    "phonation_energy_index",
    "spectral_irregularity_index",
    "mfcc_irregularity_index",
    "avhi",
    "pitch_jitter_interaction",
    "pitch_shimmer_interaction",
]


missing_features = [
    f for f in features
    if f not in df.columns
]

if missing_features:

    print("ERROR - Missing features:")
    print(missing_features)

    raise SystemExit


print("Number of features per recording:", len(features))


# ============================================================
# 6. KEEP COMPLETE PATIENTS
# ============================================================

complete_counts = (
    df.groupby("patient_id")["recording_type"]
    .nunique()
)

complete_ids = complete_counts[
    complete_counts == 13
].index

df = df[
    df["patient_id"].isin(complete_ids)
].copy()

print("Complete patients:", df["patient_id"].nunique())


# ============================================================
# 7. RECORDING TYPES
# ============================================================

recording_types = [
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
# 8. BUILD PATIENT-LEVEL FEATURE VECTORS
# ============================================================

X_list = []
y_list = []
patient_ids = []


for patient_id, group in df.groupby("patient_id"):

    group = group.set_index("recording_type")

    patient_features = []

    for recording_type in recording_types:

        values = group.loc[
            recording_type,
            features
        ].to_numpy(dtype=float)

        patient_features.extend(values)

    X_list.append(patient_features)

    y_list.append(
        int(group["label"].iloc[0])
    )

    patient_ids.append(patient_id)


X = np.array(
    X_list,
    dtype=float
)

y = np.array(y_list)

patient_ids = np.array(patient_ids)


print("Patient feature shape:", X.shape)


# ============================================================
# 9. FINAL PATIENT-LEVEL TRAIN / TEST SPLIT
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(
        X,
        y,
        groups=patient_ids
    )
)


X_train_full = X[train_idx]
y_train_full = y[train_idx]
groups_train = patient_ids[train_idx]

X_test = X[test_idx]
y_test = y[test_idx]


print("Train patients:", len(train_idx))
print("Test patients:", len(test_idx))


# ============================================================
# 10. OUT-OF-FOLD TRAINING PREDICTIONS
# ============================================================

print()
print("Generating 5-fold patient-level OOF predictions...")


oof_prob = np.zeros(
    len(train_idx)
)

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


for fold, (cv_train, cv_valid) in enumerate(
    cv.split(
        X_train_full,
        y_train_full,
        groups_train
    ),
    start=1
):

    print("Fold", fold)

    imputer = SimpleImputer(
        strategy="median"
    )

    X_cv_train = imputer.fit_transform(
        X_train_full[cv_train]
    )

    X_cv_valid = imputer.transform(
        X_train_full[cv_valid]
    )

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_cv_train,
        y_train_full[cv_train]
    )

    oof_prob[cv_valid] = (
        model.predict_proba(
            X_cv_valid
        )[:, 1]
    )


# ============================================================
# 11. FIND BEST THRESHOLD USING ONLY OOF PREDICTIONS
# ============================================================

print()
print("Searching for best threshold...")


best_threshold = 0.50
best_accuracy = 0.0
best_f1 = 0.0


for threshold in np.arange(
    0.20,
    0.81,
    0.01
):

    oof_pred = (
        oof_prob >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_train_full,
        oof_pred
    )

    f1 = f1_score(
        y_train_full,
        oof_pred
    )

    # Primary objective: accuracy
    # Secondary objective: F1
    if (
        accuracy > best_accuracy
        or (
            np.isclose(
                accuracy,
                best_accuracy
            )
            and f1 > best_f1
        )
    ):

        best_accuracy = accuracy
        best_f1 = f1
        best_threshold = threshold


print(
    "Best threshold:",
    round(best_threshold, 2)
)

print(
    "OOF accuracy:",
    round(best_accuracy * 100, 2),
    "%"
)

print(
    "OOF F1:",
    round(best_f1 * 100, 2),
    "%"
)


# ============================================================
# 12. TRAIN FINAL MODEL ON ALL TRAINING PATIENTS
# ============================================================

print()
print("Training final Random Forest...")


imputer = SimpleImputer(
    strategy="median"
)

X_train = imputer.fit_transform(
    X_train_full
)

X_test = imputer.transform(
    X_test
)


final_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


final_model.fit(
    X_train,
    y_train_full
)


# ============================================================
# 13. TEST PROBABILITIES
# ============================================================

test_prob = final_model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 14. APPLY OOF-SELECTED THRESHOLD
# ============================================================

test_pred = (
    test_prob >= best_threshold
).astype(int)


# ============================================================
# 15. FINAL TEST RESULTS
# ============================================================

print()
print("=" * 60)
print("13-RECORDING FUSION + OOF THRESHOLD")
print("=" * 60)

print(
    "Threshold:",
    round(best_threshold, 2)
)

print(
    "Accuracy:",
    round(
        accuracy_score(
            y_test,
            test_pred
        ) * 100,
        2
    ),
    "%"
)

print(
    "Precision:",
    round(
        precision_score(
            y_test,
            test_pred
        ) * 100,
        2
    ),
    "%"
)

print(
    "Recall:",
    round(
        recall_score(
            y_test,
            test_pred
        ) * 100,
        2
    ),
    "%"
)

print(
    "F1:",
    round(
        f1_score(
            y_test,
            test_pred
        ) * 100,
        2
    ),
    "%"
)

print(
    "ROC-AUC:",
    round(
        roc_auc_score(
            y_test,
            test_prob
        ),
        4
    )
)

print("=" * 60)


# ============================================================
# 16. TEST PROBABILITY DIAGNOSTICS
# ============================================================

print()
print("TEST PROBABILITY DIAGNOSTICS")
print("-" * 60)

healthy_prob = test_prob[
    y_test == 0
]

disorder_prob = test_prob[
    y_test == 1
]

print(
    "Probability range:",
    round(test_prob.min(), 4),
    "to",
    round(test_prob.max(), 4)
)

print(
    "Mean probability healthy:",
    round(healthy_prob.mean(), 4)
)

print(
    "Mean probability disorder:",
    round(disorder_prob.mean(), 4)
)

print(
    "Median probability healthy:",
    round(np.median(healthy_prob), 4)
)

print(
    "Median probability disorder:",
    round(np.median(disorder_prob), 4)
)

print("-" * 60)