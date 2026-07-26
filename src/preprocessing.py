import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from config import VARIANCE_THRESHOLD


def log_transform(X):
    return np.log2(X + 1)


def filter_variance(X, threshold=VARIANCE_THRESHOLD):
    variances = X.var(axis=0)
    keep_cols = variances[variances > threshold].index
    X_filtered = X[keep_cols]
    print(f"Variance filter: {X.shape[1]} genes -> {X_filtered.shape[1]} genes")
    return X_filtered, list(keep_cols)


def standardize(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    return X_scaled, scaler

if __name__ == "__main__":
    from data_loading import fetch_and_cache

    X, y = fetch_and_cache()
    print(f"Avant preprocessing: {X.shape}")

    X = log_transform(X)
    X = filter_variance(X)
    X = standardize(X)

    print(f"Après preprocessing: {X.shape}")