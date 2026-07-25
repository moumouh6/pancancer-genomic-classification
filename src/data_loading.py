import pandas as pd
from config import DATA_RAW_DIR, RAW_EXPRESSION_FILE, RAW_LABELS_FILE, UCI_DATASET_ID
import requests
import zipfile
import tarfile


def fetch_and_cache():
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_EXPRESSION_FILE.exists() and RAW_LABELS_FILE.exists():
        X = pd.read_csv(RAW_EXPRESSION_FILE, index_col=0)
        y = pd.read_csv(RAW_LABELS_FILE, index_col=0)
        return X, y

    zip_url = "https://archive.ics.uci.edu/static/public/401/gene+expression+cancer+rna+seq.zip"
    zip_path = DATA_RAW_DIR / "dataset.zip"

    # 1. Télécharger le zip
    response = requests.get(zip_url)
    response.raise_for_status()  # plante bruyamment si le téléchargement échoue (404, 500, etc.)
    zip_path.write_bytes(response.content)

    # 2. Dézipper — ça révèle le .tar.gz à l'intérieur
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA_RAW_DIR)

    # 3. Trouver le .tar.gz extrait automatiquement (pas de nom en dur)
    tar_path = next(DATA_RAW_DIR.glob("*.tar.gz"))

    # 4. Extraire le tar.gz — il contient un sous-dossier avec data.csv et labels.csv dedans
    with tarfile.open(tar_path) as tar:
        tar.extractall(DATA_RAW_DIR, filter="data")

    # 5. Chercher data.csv et labels.csv où qu'ils soient dans l'arborescence extraite
    data_csv = next(DATA_RAW_DIR.rglob("data.csv"))
    labels_csv = next(DATA_RAW_DIR.rglob("labels.csv"))

    X = pd.read_csv(data_csv, index_col=0)
    y = pd.read_csv(labels_csv, index_col=0)

    # 6. Sauvegarder sous tes noms de cache habituels
    X.to_csv(RAW_EXPRESSION_FILE)
    y.to_csv(RAW_LABELS_FILE)

    return X, y


def summarize(X, y):
    label_col = y.columns[0]
    print(f"Shape: {X.shape[0]} samples, {X.shape[1]} genes")
    print(f"Missing values: {X.isna().sum().sum()}")
    print(y[label_col].value_counts())


if __name__ == "__main__":
    X, y = fetch_and_cache()
    summarize(X, y)