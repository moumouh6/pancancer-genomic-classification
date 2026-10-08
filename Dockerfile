# Image légère pour servir l'API FastAPI (pas pour ré-entraîner le modèle).
FROM python:3.11-slim

WORKDIR /app

# Dépendances Python d'abord, pour profiter du cache Docker tant que
# requirements.txt ne change pas (reconstruction plus rapide).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code source : garde la structure src/ attendue par config.py
# (ROOT_DIR = parent du dossier contenant config.py, donc /app ici).
COPY src/ ./src/

# Modèle déjà entraîné (pancancer_pipeline.joblib) -- pas d'entraînement
# au démarrage du conteneur, juste le chargement.
COPY models/ ./models/

# Dataset mis en cache (requis par /sample et par fetch_and_cache() au
# démarrage de l'API). Seuls les CSV sont nécessaires, pas les archives
# brutes -- voir .dockerignore.
COPY data/ ./data/

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]
