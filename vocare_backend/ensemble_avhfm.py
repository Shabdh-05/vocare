import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# ============================================================
# AVHFM ENSEMBLE TRAINING
# SVM + RANDOM FOREST + GRADIENT BOOSTING
# Patient-level evaluation
# ============================================================

RANDOM_STATE = 42


def create_avhfm_features(df):
    """Create the exact 77 AVHFM features used by the existing models."""

    df = df.copy()

    eps = 1e-6

    # --------------------------------------------------------
    # Derived AVHFM features
    # --------------------------------------------------------

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
        df["jitter"] + df["shimmer"]
    )

    df["phonation_energy_index"] = (
        df["rms_mean"]
        / (df["zcr_mean"] + eps)
    )

    df["spectral_irregularity_index"] = (
        df["centroid_std"]
        + df["bandwidth_std"]
        + df["rolloff_std"]
    )

    mfcc_std_cols = [
        f"mfcc_{i + 1}_std"
        for i in range(13)
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
        df["pitch_std"] * df["jitter"]
    )

    df["pitch_shimmer_interaction"] = (
        df["pitch_std"] * df["shimmer"]
    )

    # Remove metadata columns.
    exclude_cols = [
        "file_path",
        "disease",
        "patient_id",
        "file_name",
        "label",
    ]

    X = df.drop(columns=exclude_cols)

    # Replace infinite values.
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    y = df["label"].astype(int)

    groups = df["patient_id"]

    return X, y, groups


def build_svm():
    """Tuned SVM."""

    return Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            SVC(
                C=5,
                kernel="rbf",
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=RANDOM_STATE,
            )
        ),
    ])


def build_rf():
    """Tuned Random Forest."""

    return Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                max_features="log2",
                min_samples_split=2,
                min_samples_leaf=2,
                class_weight=None,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
    ])


def build_gb():
    """Tuned Gradient Boosting."""

    return Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "classifier",
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=250,
                max_leaf_nodes=31,
                max_depth=None,
                min_samples_leaf=40,
                l2_regularization=1.0,
                random_state=RANDOM_STATE,
            )
        ),
    ])


def find_best_weights(y_true, svm_prob, rf_prob, gb_prob):
    """
    Search ensemble weights using ONLY out-of-fold
    training predictions.

    The final test set is never used here.
    """

    best_f1 = -1
    best_weights = None
    best_threshold = 0.50

    # --------------------------------------------------------
    # Search weights.
    # Weights sum to 1.
    # --------------------------------------------------------

    weight_values = np.arange(
        0.0,
        1.01,
        0.05
    )

    for svm_w in weight_values:

        for rf_w in weight_values:

            gb_w = 1.0 - svm_w - rf_w

            if gb_w < 0 or gb_w > 1:
                continue

            ensemble_prob = (
                svm_w * svm_prob
                + rf_w * rf_prob
                + gb_w * gb_prob
            )

            # ------------------------------------------------
            # Search classification threshold.
            # ------------------------------------------------

            for threshold in np.arange(
                0.30,
                0.71,
                0.01
            ):

                pred = (
                    ensemble_prob >= threshold
                ).astype(int)

                score = f1_score(
                    y_true,
                    pred
                )

                if score > best_f1:

                    best_f1 = score

                    best_weights = (
                        svm_w,
                        rf_w,
                        gb_w
                    )

                    best_threshold = threshold

    return (
        best_weights,
        best_threshold,
        best_f1
    )


def main():

    base_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    project_root = os.path.dirname(
        base_dir
    )

    csv_path = os.path.join(
        project_root,
        "vocal_features.csv"
    )

    print("=" * 70)
    print("AVHFM ENSEMBLE TRAINING")
    print("SVM + RANDOM FOREST + GRADIENT BOOSTING")
    print("=" * 70)

    print(
        f"\nLoading dataset:\n{csv_path}"
    )

    df = pd.read_csv(csv_path)

    print(
        f"Loaded {len(df)} recordings."
    )

    # --------------------------------------------------------
    # Create 77 AVHFM features.
    # --------------------------------------------------------

    X, y, groups = create_avhfm_features(df)

    print(
        f"AVHFM feature count: {X.shape[1]}"
    )

    if X.shape[1] != 77:

        raise ValueError(
            f"Expected 77 features, "
            f"but found {X.shape[1]}"
        )

    print(
        f"Unique patients: "
        f"{groups.nunique()}"
    )

    # ========================================================
    # STEP 1
    # Patient-level train/test split
    # ========================================================

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    train_idx, test_idx = next(
        splitter.split(
            X,
            y,
            groups
        )
    )

    X_train = X.iloc[train_idx].copy()
    y_train = y.iloc[train_idx].copy()
    groups_train = groups.iloc[train_idx].copy()

    X_test = X.iloc[test_idx].copy()
    y_test = y.iloc[test_idx].copy()

    print("\n" + "=" * 70)
    print("PATIENT-LEVEL DATA SPLIT")
    print("=" * 70)

    print(
        f"Training recordings: {len(X_train)}"
    )

    print(
        f"Testing recordings:  {len(X_test)}"
    )

    print(
        f"Training patients: "
        f"{groups_train.nunique()}"
    )

    print(
        f"Testing patients: "
        f"{groups.iloc[test_idx].nunique()}"
    )

    overlap = set(
        groups_train.unique()
    ).intersection(
        set(groups.iloc[test_idx].unique())
    )

    print(
        f"Patient overlap: {len(overlap)}"
    )

    if len(overlap) != 0:

        raise ValueError(
            "Patient leakage detected!"
        )

    # ========================================================
    # STEP 2
    # Out-of-fold predictions
    # ========================================================

    print("\n" + "=" * 70)
    print("GENERATING OUT-OF-FOLD PREDICTIONS")
    print("=" * 70)

    n_splits = 3

    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    svm_oof = np.zeros(
        len(X_train)
    )

    rf_oof = np.zeros(
        len(X_train)
    )

    gb_oof = np.zeros(
        len(X_train)
    )

    for fold, (fold_train, fold_valid) in enumerate(
        cv.split(
            X_train,
            y_train,
            groups_train
        ),
        start=1
    ):

        print(
            f"\n--- Fold {fold}/{n_splits} ---"
        )

        X_fold_train = X_train.iloc[
            fold_train
        ]

        y_fold_train = y_train.iloc[
            fold_train
        ]

        X_fold_valid = X_train.iloc[
            fold_valid
        ]

        # ----------------------------------------------------
        # SVM
        # ----------------------------------------------------

        print("Training SVM...")

        svm = build_svm()

        svm.fit(
            X_fold_train,
            y_fold_train
        )

        svm_oof[fold_valid] = (
            svm.predict_proba(
                X_fold_valid
            )[:, 1]
        )

        # ----------------------------------------------------
        # Random Forest
        # ----------------------------------------------------

        print("Training Random Forest...")

        rf = build_rf()

        rf.fit(
            X_fold_train,
            y_fold_train
        )

        rf_oof[fold_valid] = (
            rf.predict_proba(
                X_fold_valid
            )[:, 1]
        )

        # ----------------------------------------------------
        # Gradient Boosting
        # ----------------------------------------------------

        print("Training Gradient Boosting...")

        gb = build_gb()

        gb.fit(
            X_fold_train,
            y_fold_train
        )

        gb_oof[fold_valid] = (
            gb.predict_proba(
                X_fold_valid
            )[:, 1]
        )

        print(
            f"Fold {fold} completed."
        )

    # ========================================================
    # STEP 3
    # Individual OOF performance
    # ========================================================

    print("\n" + "=" * 70)
    print("OUT-OF-FOLD PERFORMANCE")
    print("=" * 70)

    for name, probs in [
        ("SVM", svm_oof),
        ("RF", rf_oof),
        ("GB", gb_oof),
    ]:

        pred = (
            probs >= 0.50
        ).astype(int)

        print(
            f"\n{name}"
        )

        print(
            f"Accuracy : "
            f"{accuracy_score(y_train, pred) * 100:.2f}%"
        )

        print(
            f"Precision: "
            f"{precision_score(y_train, pred) * 100:.2f}%"
        )

        print(
            f"Recall   : "
            f"{recall_score(y_train, pred) * 100:.2f}%"
        )

        print(
            f"F1       : "
            f"{f1_score(y_train, pred) * 100:.2f}%"
        )

        print(
            f"ROC-AUC  : "
            f"{roc_auc_score(y_train, probs):.4f}"
        )

    # ========================================================
    # STEP 4
    # Optimize ensemble weights using OOF only
    # ========================================================

    print("\n" + "=" * 70)
    print("OPTIMIZING ENSEMBLE")
    print("=" * 70)

    (
        best_weights,
        best_threshold,
        best_oof_f1
    ) = find_best_weights(
        y_train,
        svm_oof,
        rf_oof,
        gb_oof
    )

    svm_w, rf_w, gb_w = best_weights

    print(
        f"\nBest SVM weight: "
        f"{svm_w:.2f}"
    )

    print(
        f"Best RF weight : "
        f"{rf_w:.2f}"
    )

    print(
        f"Best GB weight : "
        f"{gb_w:.2f}"
    )

    print(
        f"Best threshold : "
        f"{best_threshold:.2f}"
    )

    print(
        f"OOF Ensemble F1: "
        f"{best_oof_f1 * 100:.2f}%"
    )

    # ========================================================
    # STEP 5
    # Train final models on ALL training patients
    # ========================================================

    print("\n" + "=" * 70)
    print("TRAINING FINAL ENSEMBLE COMPONENTS")
    print("=" * 70)

    print("\nTraining final SVM...")

    final_svm = build_svm()

    final_svm.fit(
        X_train,
        y_train
    )

    print("Training final Random Forest...")

    final_rf = build_rf()

    final_rf.fit(
        X_train,
        y_train
    )

    print("Training final Gradient Boosting...")

    final_gb = build_gb()

    final_gb.fit(
        X_train,
        y_train
    )

    # ========================================================
    # STEP 6
    # FINAL TEST
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL PATIENT-LEVEL TEST")
    print("=" * 70)

    svm_test_prob = (
        final_svm.predict_proba(
            X_test
        )[:, 1]
    )

    rf_test_prob = (
        final_rf.predict_proba(
            X_test
        )[:, 1]
    )

    gb_test_prob = (
        final_gb.predict_proba(
            X_test
        )[:, 1]
    )

    ensemble_test_prob = (
        svm_w * svm_test_prob
        + rf_w * rf_test_prob
        + gb_w * gb_test_prob
    )

    ensemble_test_pred = (
        ensemble_test_prob >= best_threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        ensemble_test_pred
    )

    precision = precision_score(
        y_test,
        ensemble_test_pred
    )

    recall = recall_score(
        y_test,
        ensemble_test_pred
    )

    f1 = f1_score(
        y_test,
        ensemble_test_pred
    )

    roc_auc = roc_auc_score(
        y_test,
        ensemble_test_prob
    )

    print(
        f"\nAVHFM ENSEMBLE RESULTS"
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
        f"ROC-AUC  : {roc_auc:.4f}"
    )

    # ========================================================
    # STEP 7
    # Save models and configuration
    # ========================================================

    print("\n" + "=" * 70)
    print("SAVING AVHFM ENSEMBLE")
    print("=" * 70)

    ensemble_path = os.path.join(
        base_dir,
        "avhfm_ensemble.pkl"
    )

    ensemble_data = {
        "svm": final_svm,
        "rf": final_rf,
        "gb": final_gb,
        "weights": {
            "svm": svm_w,
            "rf": rf_w,
            "gb": gb_w,
        },
        "threshold": best_threshold,
        "feature_columns": list(X.columns),
    }

    joblib.dump(
        ensemble_data,
        ensemble_path
    )

    print(
        f"Saved ensemble to:\n"
        f"{ensemble_path}"
    )

    # Save feature columns.
    feature_columns_path = os.path.join(
        base_dir,
        "avhfm_feature_columns.pkl"
    )

    joblib.dump(
        list(X.columns),
        feature_columns_path
    )

    # Save threshold.
    threshold_path = os.path.join(
        base_dir,
        "avhfm_threshold.pkl"
    )

    joblib.dump(
        best_threshold,
        threshold_path
    )

    print(
        "\nAVHFM ensemble training complete."
    )


if __name__ == "__main__":
    main()