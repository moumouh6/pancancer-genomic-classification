from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="Pan-Cancer Classifier API")

artifact = joblib.load("models/random_forest.joblib")
model = artifact["model"]
scaler = artifact["scaler"]
selected_genes = artifact["selected_genes"]


class PredictionRequest(BaseModel):
    expression: dict  # {"gene_name": valeur_expression, ...}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}



@app.post("/predict")
def predict(request: PredictionRequest):
    # 1. Vérifier que tous les gènes attendus sont présents
    missing_genes = set(selected_genes) - set(request.expression.keys())
    if missing_genes:
        raise HTTPException(
            status_code=400,
            detail=f"{len(missing_genes)} gènes manquants sur {len(selected_genes)} attendus. "
                   f"Exemples manquants: {list(missing_genes)[:5]}"
        )

    # 2. Construire un DataFrame, dans le bon ordre de colonnes
    X = pd.DataFrame([request.expression])
    X = X[selected_genes]  # force l'ordre exact attendu par le scaler/modèle

    # 3. Log-transform — même formule que preprocessing.py
    X = np.log2(X + 1)

    # 4. Standardisation avec le scaler déjà entraîné — jamais fit_transform ici
    X_scaled = scaler.transform(X)

    # 5. Prédiction + probabilités
    prediction = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]

    # 6. Formatter les probabilités par classe
    proba_dict = {
        cls: round(float(prob), 4)
        for cls, prob in zip(model.classes_, probabilities)
    }

    return {
        "predicted_class": prediction,
        "confidence": proba_dict[prediction],
        "probabilities": proba_dict
    }