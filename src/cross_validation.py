"""
Stratified K-Fold cross-validation for the pan-cancer classifier.

Why a sklearn Pipeline here: to avoid leakage, the variance filter and the
scaler must be refit on each fold's training split, not just once on the
overall train set. Wrapping them in a Pipeline and handing that Pipeline to
cross_validate() makes scikit-learn do this automatically and correctly for
every fold.

VarianceThreshold(threshold=T) keeps features with variance > T, which is
the same rule as the manual filter_variance() in preprocessing.py.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from config import RANDOM_STATE, VARIANCE_THRESHOLD, CV_FOLDS


def build_pipeline():
    return Pipeline([
        ("variance_filter", VarianceThreshold(threshold=VARIANCE_THRESHOLD)),
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )),
    ])


def run_cross_validation(X, y, n_splits=CV_FOLDS):
    pipeline = build_pipeline()

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
    }

    results = cross_validate(
        pipeline, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1,
    )

    print(f"\n{n_splits}-fold Stratified Cross-Validation results\n" + "-" * 45)
    for metric in scoring:
        scores = results[f"test_{metric}"]
        print(f"{metric:12s}: {scores.mean():.4f} +/- {scores.std():.4f}  "
              f"(per fold: {np.round(scores, 4).tolist()})")

    return results


if __name__ == "__main__":
    from data_loading import fetch_and_cache
    from preprocessing import log_transform

    X, y = fetch_and_cache()
    label_col = y.columns[0]
    y = y[label_col]

    # log2(x+1) is a fixed, per-sample transform -- safe to apply before CV,
    # same reasoning as in classification.py.
    X = log_transform(X)

    run_cross_validation(X, y)
