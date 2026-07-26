import requests
import json
from data_loading import fetch_and_cache
from preprocessing import log_transform, filter_variance

X, y = fetch_and_cache()

# Attention : pas de log_transform ni de standardize ici !
# L'API attend les valeurs BRUTES du patient (avant log-transform),
# c'est elle qui applique log-transform + scaler en interne, comme un vrai patient le ferait.

_, selected_genes = filter_variance(log_transform(X))

# On prend un seul patient (index 0) et on garde uniquement les gènes attendus par le modèle
sample_patient = X.iloc[0][selected_genes]
true_label = y.iloc[0, 0]

payload = {"expression": sample_patient.to_dict()}

response = requests.post("http://localhost:8000/predict", json=payload)



def test_missing_genes():
    incomplete_payload = {"expression": {"gene_1": 5.2, "gene_2": 3.1}}  # volontairement incomplet
    response = requests.post("http://localhost:8000/predict", json=incomplete_payload)
    print(f"Status code: {response.status_code}")
    print(f"Réponse: {response.json()}")


if __name__ == "__main__":
    print(f"Vraie classe: {true_label}")
    print(f"Réponse API: {json.dumps(response.json(), indent=2)}")
    # ... ton test existant avec sample_patient ...
    print("\n--- Test avec gènes manquants ---")
    test_missing_genes()