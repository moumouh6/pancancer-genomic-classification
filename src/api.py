from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import random
from fastapi.middleware.cors import CORSMiddleware
from data_loading import fetch_and_cache
from config import ROOT_DIR

app = FastAPI(title="Pan-Cancer Classifier API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # pour la démo locale ; à restreindre en production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chargé une seule fois au démarrage, comme le modèle
X_full, y_full = fetch_and_cache()

MODEL_PATH = ROOT_DIR / "models" / "random_forest.joblib"
artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
scaler = artifact["scaler"]
selected_genes = artifact["selected_genes"]


class PredictionRequest(BaseModel):
    expression: dict  # {"gene_name": valeur_expression, ...}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/sample")
def get_sample():
    idx = random.randint(0, len(X_full) - 1)
    sample_expression = X_full.iloc[idx][selected_genes].to_dict()
    true_label = y_full.iloc[idx, 0]
    return {
        "expression": sample_expression,
        "true_label": true_label
    }

@app.post("/predict")
def predict(request: PredictionRequest):
    missing_genes = set(selected_genes) - set(request.expression.keys())
    if missing_genes:
        raise HTTPException(
            status_code=400,
            detail=f"{len(missing_genes)} gènes manquants sur {len(selected_genes)} attendus. "
                   f"Exemples manquants: {list(missing_genes)[:5]}"
        )

    X = pd.DataFrame([request.expression])
    X = X[selected_genes]

    X = np.log2(X + 1)

    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]

    proba_dict = {
        cls: round(float(prob), 4)
        for cls, prob in zip(model.classes_, probabilities)
    }

    return {
        "predicted_class": prediction,
        "confidence": proba_dict[prediction],
        "probabilities": proba_dict
    }