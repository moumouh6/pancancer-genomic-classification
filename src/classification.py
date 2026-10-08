import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from config import RANDOM_STATE, RESULTS_FIGURES_DIR, ROOT_DIR
from cross_validation import build_pipeline


def split_data(X, y, test_size=0.3):
    label_col = y.columns[0]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y[label_col],
        test_size=test_size,
        stratify=y[label_col],
        random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


def train_pipeline(X_train, y_train):
    """Fit the unified Pipeline (variance filter + scaler + Random Forest)
    on train data only. This is the single object later used by the API:
    one .fit() here, one .predict() on new raw gene-expression rows."""
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    return y_pred, cm


def plot_confusion_matrix(cm, class_labels):
    RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_FIGURES_DIR / "confusion_matrix.png"

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Prédiction")
    plt.ylabel("Vraie classe")
    plt.title("Matrice de confusion")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Matrice de confusion sauvegardée: {out_path}")


def save_pipeline(pipeline, path=None):
    """Save the whole fitted Pipeline as ONE object: variance filter, scaler
    and Random Forest together. The API only needs to load this one file and
    call .predict() / .predict_proba() on a raw (log-transformed) gene vector
    -- no separate gene list or scaler to manage."""
    if path is None:
        path = ROOT_DIR / "models" / "pancancer_pipeline.joblib"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"Pipeline sauvegardé: {path}")


if __name__ == "__main__":
    from data_loading import fetch_and_cache
    from preprocessing import log_transform

    X, y = fetch_and_cache()

    # log2(x+1) is a fixed, per-sample transform (no statistics estimated
    # from the data), so applying it before the split is NOT leakage.
    X = log_transform(X)

    # Split FIRST, on raw (log-transformed) data. The Pipeline below is
    # fit only on X_train -- variance filter and scaler never see X_test.
    X_train, X_test, y_train, y_test = split_data(X, y)

    pipeline = train_pipeline(X_train, y_train)

    y_pred, cm = evaluate_model(pipeline, X_test, y_test)
    plot_confusion_matrix(cm, pipeline.classes_)

    save_pipeline(pipeline)