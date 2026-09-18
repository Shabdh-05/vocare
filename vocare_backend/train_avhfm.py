import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, cross_val_score
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report
)


def create_avhfm_features(df):
    """
    Create the AVHFM-derived acoustic features.
    These calculations must remain synchronized with feature_utils.py.
    """

    df = df.copy()

    eps = 1e-6

    # ---------------------------------------------------------
    # AVHFM derived acoustic features
    # ---------------------------------------------------------

    df["vocal_stability_index"] = (
        df["hnr"] /
        (
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
        df["rms_mean"] /
        (df["zcr_mean"] + eps)
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

    # ---------------------------------------------------------
    # Acoustic Voice Health Index
    # ---------------------------------------------------------

    df["avhi"] = (
        0.30 * df["perturbation_index"]
        + 0.25 * df["pitch_std"]
        + 0.20 * df["spectral_irregularity_index"]
        + 0.15 * df["mfcc_irregularity_index"]
        - 0.10 * df["hnr"]
    )

    # ---------------------------------------------------------
    # Interaction features
    # ---------------------------------------------------------

    df["pitch_jitter_interaction"] = (
        df["pitch_std"] * df["jitter"]
    )

    df["pitch_shimmer_interaction"] = (
        df["pitch_std"] * df["shimmer"]
    )

    return df


def main():

    # =========================================================
    # 1. Paths
    # =========================================================

    base_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    project_root = os.path.dirname(base_dir)

    csv_path = os.path.join(
        project_root,
        "vocal_features.csv"
    )

    print("=" * 70)
    print("AVHFM MODEL TRAINING")
    print("=" * 70)

    print(f"\nLoading dataset from:")
    print(csv_path)

    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} records.")
    print(f"Unique patients: {df['patient_id'].nunique()}")

    # =========================================================
    # 2. Create AVHFM features
    # =========================================================

    print("\nCreating AVHFM acoustic features...")

    df = create_avhfm_features(df)

    # =========================================================
    # 3. Select features
    # =========================================================

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

    print(
        f"Total training features: {len(feature_cols)}"
    )

    X = df[feature_cols].copy()

    y = df["label"].astype(int)

    groups = df["patient_id"]

    # Replace invalid numerical values
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # =========================================================
    # 4. Patient-level train/test split
    # =========================================================

    print("\nCreating patient-level train/test split...")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42
    )

    train_idx, test_idx = next(
        splitter.split(
            X,
            y,
            groups=groups
        )
    )

    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()

    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()

    groups_train = groups.iloc[train_idx].copy()
    groups_test = groups.iloc[test_idx].copy()

    print(
        f"Training records: {len(X_train)}"
    )

    print(
        f"Testing records: {len(X_test)}"
    )

    print(
        f"Training patients: {groups_train.nunique()}"
    )

    print(
        f"Testing patients: {groups_test.nunique()}"
    )

    overlap = set(
        groups_train
    ).intersection(
        set(groups_test)
    )

    print(
        f"Patient overlap: {len(overlap)}"
    )

    if len(overlap) != 0:
        raise RuntimeError(
            "Patient leakage detected!"
        )

    # =========================================================
    # 5. Candidate models
    # =========================================================

    print("\nPreparing candidate models...")

    models = {

        "Random Forest": Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=600,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features="sqrt",
                    class_weight="balanced",
                    bootstrap=True,
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]),

        "Extra Trees": Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "classifier",
                ExtraTreesClassifier(
                    n_estimators=600,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features="sqrt",
                    class_weight="balanced",
                    bootstrap=False,
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]),

        "Hist Gradient Boosting": Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    max_iter=400,
                    learning_rate=0.05,
                    max_leaf_nodes=31,
                    min_samples_leaf=15,
                    l2_regularization=1.0,
                    random_state=42
                )
            )
        ]),

        "Logistic Regression": Pipeline([
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
                LogisticRegression(
                    C=1.0,
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=42
                )
            )
        ])
    }

    # =========================================================
    # 6. Patient-grouped cross-validation
    # =========================================================

    print("\nEvaluating candidate models...")
    print(
        "Using 5-fold patient-grouped cross-validation."
    )

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    results = []

    for name, model in models.items():

        print("\n" + "-" * 70)
        print(f"Testing: {name}")

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            groups=groups_train,
            cv=cv,
            scoring="f1",
            n_jobs=1
        )

        mean_score = scores.mean()
        std_score = scores.std()

        print(
            f"CV F1: {mean_score:.4f} "
            f"+/- {std_score:.4f}"
        )

        results.append(
            {
                "name": name,
                "model": model,
                "cv_f1": mean_score
            }
        )

    # =========================================================
    # 7. Select best model
    # =========================================================

    results.sort(
        key=lambda x: x["cv_f1"],
        reverse=True
    )

    best_name = results[0]["name"]
    best_model = results[0]["model"]

    print("\n" + "=" * 70)
    print("BEST MODEL")
    print("=" * 70)

    print(
        f"Selected: {best_name}"
    )

    print(
        f"Cross-validation F1: "
        f"{results[0]['cv_f1'] * 100:.2f}%"
    )

    # =========================================================
    # 8. Train best model
    # =========================================================

    print(
        f"\nTraining final {best_name} model..."
    )

    best_model.fit(
        X_train,
        y_train
    )

    # =========================================================
    # 9. Find best classification threshold
    # =========================================================

    print(
        "\nOptimizing classification threshold..."
    )

    train_prob = best_model.predict_proba(
        X_train
    )[:, 1]

    thresholds = np.arange(
        0.30,
        0.71,
        0.01
    )

    best_threshold = 0.50
    best_threshold_f1 = 0.0

    for threshold in thresholds:

        train_pred = (
            train_prob >= threshold
        ).astype(int)

        score = f1_score(
            y_train,
            train_pred
        )

        if score > best_threshold_f1:
            best_threshold_f1 = score
            best_threshold = threshold

    print(
        f"Best training threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Training F1 at threshold: "
        f"{best_threshold_f1 * 100:.2f}%"
    )

    # =========================================================
    # 10. Final untouched test evaluation
    # =========================================================

    print("\n" + "=" * 70)
    print("FINAL PATIENT-LEVEL TEST EVALUATION")
    print("=" * 70)

    y_prob = best_model.predict_proba(
        X_test
    )[:, 1]

    y_pred = (
        y_prob >= best_threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    print(
        f"\nAccuracy  : {accuracy * 100:.2f}%"
    )

    print(
        f"Precision : {precision * 100:.2f}%"
    )

    print(
        f"Recall    : {recall * 100:.2f}%"
    )

    print(
        f"F1 Score  : {f1 * 100:.2f}%"
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )

    print(
        "\nClassification Report:\n"
    )

    print(
        classification_report(
            y_test,
            y_pred,
            digits=4,
            zero_division=0
        )
    )

    # =========================================================
    # 11. Save model
    # =========================================================

    model_path = os.path.join(
        base_dir,
        "avhfm_model.pkl"
    )

    cols_path = os.path.join(
        base_dir,
        "avhfm_feature_columns.pkl"
    )

    threshold_path = os.path.join(
        base_dir,
        "avhfm_threshold.pkl"
    )

    print(
        f"\nSaving model to:\n{model_path}"
    )

    joblib.dump(
        best_model,
        model_path
    )

    print(
        f"Saving feature columns to:\n{cols_path}"
    )

    joblib.dump(
        feature_cols,
        cols_path
    )

    print(
        f"Saving threshold to:\n{threshold_path}"
    )

    joblib.dump(
        best_threshold,
        threshold_path
    )

    # =========================================================
    # 12. Final summary
    # =========================================================

    print("\n" + "=" * 70)
    print("AVHFM TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best model          : {best_name}"
    )

    print(
        f"Accuracy            : {accuracy * 100:.2f}%"
    )

    print(
        f"Precision           : {precision * 100:.2f}%"
    )

    print(
        f"Recall              : {recall * 100:.2f}%"
    )

    print(
        f"F1 Score            : {f1 * 100:.2f}%"
    )

    print(
        f"ROC-AUC             : {roc_auc:.4f}"
    )

    print(
        f"Decision threshold  : {best_threshold:.2f}"
    )

    print("\nModel files saved successfully.")


if __name__ == "__main__":
    main()