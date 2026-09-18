import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def create_avhfm_features(df):
    df = df.copy()
    eps = 1e-6

    df["vocal_stability_index"] = (
        df["hnr"] /
        (df["pitch_std"] + df["jitter"] + df["shimmer"] + eps)
    )

    df["perturbation_index"] = (
        df["jitter"] + df["shimmer"]
    )

    df["phonation_energy_index"] = (
        df["rms_mean"] /
        (df["zcr_mean"] + eps)
    )

    df["spectral_irregularity_index"] = (
        df["centroid_std"] +
        df["bandwidth_std"] +
        df["rolloff_std"]
    )

    mfcc_std_cols = [
        f"mfcc_{i+1}_std"
        for i in range(13)
    ]

    df["mfcc_irregularity_index"] = (
        df[mfcc_std_cols].mean(axis=1)
    )

    df["avhi"] = (
        0.30 * df["perturbation_index"] +
        0.25 * df["pitch_std"] +
        0.20 * df["spectral_irregularity_index"] +
        0.15 * df["mfcc_irregularity_index"] -
        0.10 * df["hnr"]
    )

    df["pitch_jitter_interaction"] = (
        df["pitch_std"] * df["jitter"]
    )

    df["pitch_shimmer_interaction"] = (
        df["pitch_std"] * df["shimmer"]
    )

    return df


def main():

    base_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    project_root = os.path.dirname(base_dir)

    csv_path = os.path.join(
        project_root,
        "vocal_features.csv"
    )

    print("Loading dataset...")

    df = pd.read_csv(csv_path)

    print("Records:", len(df))
    print(
        "Patients:",
        df["patient_id"].nunique()
    )

    df = create_avhfm_features(df)

    exclude_cols = {
        "file_path",
        "disease",
        "patient_id",
        "file_name",
        "label"
    }

    feature_cols = [
        col
        for col in df.columns
        if col not in exclude_cols
    ]

    X = df[feature_cols].copy()

    y = df["label"].astype(int)

    groups = df["patient_id"]

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # =========================================================
    # PATIENT-LEVEL TRAIN / TEST SPLIT
    # =========================================================

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42
    )

    train_idx, test_idx = next(
        splitter.split(
            X,
            y,
            groups
        )
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    groups_train = groups.iloc[train_idx]

    print(
        "\nTrain records:",
        len(X_train)
    )

    print(
        "Test records:",
        len(X_test)
    )

    print(
        "Train patients:",
        groups_train.nunique()
    )

    print(
        "Test patients:",
        groups.iloc[test_idx].nunique()
    )

    print(
        "Patient overlap:",
        len(
            set(groups_train.unique())
            &
            set(groups.iloc[test_idx].unique())
        )
    )

    # =========================================================
    # HISTOGRAM GRADIENT BOOSTING
    # =========================================================

    pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "classifier",
            HistGradientBoostingClassifier(
                random_state=42
            )
        )
    ])

    # ---------------------------------------------------------
    # PARAMETER SEARCH
    # ---------------------------------------------------------

    param_grid = {
        "classifier__learning_rate": [
            0.03,
            0.05,
            0.10
        ],

        "classifier__max_iter": [
            150,
            250
        ],

        "classifier__max_leaf_nodes": [
            15,
            31,
            63
        ],

        "classifier__max_depth": [
            None,
            8
        ],

        "classifier__min_samples_leaf": [
            20,
            40
        ],

        "classifier__l2_regularization": [
            0.0,
            1.0
        ]
    }

    total_combinations = (
        3 * 2 * 3 * 2 * 2 * 2
    )

    total_fits = (
        total_combinations * 3
    )

    cv = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )

    print(
        "\n================================"
    )

    print(
        "STARTING GRADIENT BOOSTING TUNING"
    )

    print(
        "================================"
    )

    print(
        "Parameter combinations:",
        total_combinations
    )

    print(
        "Cross-validation folds: 3"
    )

    print(
        "Total fits:",
        total_fits
    )

    print(
        "\nThis may take several minutes."
    )

    # =========================================================
    # GRID SEARCH
    # =========================================================

    grid = GridSearchCV(
        pipeline,
        param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=2,
        refit=True
    )

    grid.fit(
        X_train,
        y_train,
        groups=groups_train
    )

    # =========================================================
    # BEST MODEL
    # =========================================================

    print(
        "\n================================"
    )

    print(
        "BEST GRADIENT BOOSTING"
    )

    print(
        "================================"
    )

    print(
        "Best parameters:"
    )

    print(
        grid.best_params_
    )

    print(
        "Best CV F1:",
        round(
            grid.best_score_ * 100,
            2
        ),
        "%"
    )

    # =========================================================
    # TEST
    # =========================================================

    best_model = grid.best_estimator_

    y_pred = best_model.predict(
        X_test
    )

    y_prob = best_model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    print(
        "\n================================"
    )

    print(
        "PATIENT-LEVEL TEST RESULTS"
    )

    print(
        "================================"
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

    # =========================================================
    # SAVE MODEL
    # =========================================================

    model_path = os.path.join(
        base_dir,
        "tuned_gb_model.pkl"
    )

    joblib.dump(
        best_model,
        model_path
    )

    print(
        "\nSaved model:"
    )

    print(
        model_path
    )

    print(
        "\nGradient Boosting tuning completed successfully."
    )


if __name__ == "__main__":
    main()