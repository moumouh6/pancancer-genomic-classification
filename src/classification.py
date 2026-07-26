import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from config import RANDOM_STATE


def split_data(X, y, test_size=0.3):
    label_col = y.columns[0]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y[label_col],
        test_size=test_size,
        stratify=y[label_col],
        random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    return y_pred, cm


def plot_confusion_matrix(cm, class_labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Prédiction")
    plt.ylabel("Vraie classe")
    plt.title("Matrice de confusion")
    plt.tight_layout()
    plt.savefig("results/figures/confusion_matrix.png", dpi=150)
    plt.show()


def save_model(model, scaler, selected_genes, path="models/random_forest.joblib"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "scaler": scaler,
        "selected_genes": selected_genes
    }, path)
    print(f"Modèle sauvegardé: {path}")


if __name__ == "__main__":
    from data_loading import fetch_and_cache
    from preprocessing import log_transform, filter_variance, standardize

    X, y = fetch_and_cache()
    X = log_transform(X)
    X, selected_genes = filter_variance(X)
    X, scaler = standardize(X)

    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_random_forest(X_train, y_train)

    y_pred, cm = evaluate_model(model, X_test, y_test)
    plot_confusion_matrix(cm, model.classes_)

    save_model(model, scaler, selected_genes)