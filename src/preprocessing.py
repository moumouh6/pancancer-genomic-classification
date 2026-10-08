import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from config import VARIANCE_THRESHOLD


def log_transform(X):
    return np.log2(X + 1)


def fit_variance_filter(X, threshold=VARIANCE_THRESHOLD):
    """Fit the variance filter on TRAIN data only. Returns filtered X and the
    list of selected genes, to be reused (not refit) on the test set."""
    variances = X.var(axis=0)
    keep_cols = variances[variances > threshold].index
    X_filtered = X[keep_cols]
    print(f"Variance filter (fit on train): {X.shape[1]} genes -> {X_filtered.shape[1]} genes")
    return X_filtered, list(keep_cols)


def apply_variance_filter(X, selected_genes):
    """Apply a variance filter already fit on train to new data (e.g. test)."""
    return X[selected_genes]


def fit_standardize(X):
    """Fit the scaler on TRAIN data only. Returns scaled X and the fitted
    scaler, to be reused (not refit) on the test set."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    return X_scaled, scaler


def apply_standardize(X, scaler):
    """Apply a scaler already fit on train to new data (e.g. test)."""
    X_scaled = scaler.transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)


# Backwards-compatible aliases (old names) kept out on purpose: filter_variance
# and standardize used to fit_transform on the full dataset, which leaked
# test information into the train-time choices (genes kept, mean/std used).
# Use fit_variance_filter/apply_variance_filter and fit_standardize/apply_standardize instead.

if __name__ == "__main__":
    from data_loading import fetch_and_cache

    X, y = fetch_and_cache()
    print(f"Avant preprocessing: {X.shape}")

    X = log_transform(X)
    X, selected_genes = fit_variance_filter(X)
    X, scaler = fit_standardize(X)

    print(f"Après preprocessing: {X.shape}")