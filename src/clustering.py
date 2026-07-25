import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from config import RANDOM_STATE


def run_pca(X, n_components=2):
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    print(f"Variance expliquée par composante: {pca.explained_variance_ratio_}")
    print(f"Variance totale expliquée: {pca.explained_variance_ratio_.sum():.2%}")
    return X_pca


def run_kmeans(X_pca, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = kmeans.fit_predict(X_pca)
    return cluster_labels


def plot_clusters(X_pca, cluster_labels, true_labels):
    label_col = true_labels.columns[0]
    true_classes = true_labels[label_col]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Gauche : ce que K-Means a trouvé, sans connaître les vraies classes
    scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap="tab10")
    axes[0].set_title("Clusters trouvés par K-Means")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    # Droite : les vraies étiquettes de cancer
    unique_classes = true_classes.unique()
    class_to_num = {cls: i for i, cls in enumerate(unique_classes)}
    numeric_true = true_classes.map(class_to_num)

    scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=numeric_true, cmap="tab10")
    axes[1].set_title("Vraies classes de cancer")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")

    plt.tight_layout()
    plt.savefig("results/figures/pca_clusters_comparison.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    from data_loading import fetch_and_cache
    from preprocessing import log_transform, filter_variance, standardize

    X, y = fetch_and_cache()
    X = log_transform(X)
    X = filter_variance(X)
    X = standardize(X)

    X_pca = run_pca(X, n_components=2)
    cluster_labels = run_kmeans(X_pca)
    plot_clusters(X_pca, cluster_labels, y)
