import joblib
import pandas as pd
import numpy as np

def test_inference_range():
    # 1. Charger les assets
    model = joblib.load('xgb_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    # 2. Créer une donnée de test (Profil standard)
    feature_names = ['loan_amt_outstanding', 'total_debt_outstanding', 'income', 'years_employed', 'fico_score']
    sample_data = pd.DataFrame([[5000, 15000, 45000, 5, 650]], columns=feature_names)
    
    # 3. Exécuter l'inférence
    input_scaled = scaler.transform(sample_data)
    prob = model.predict_proba(input_scaled)[0][1]
    pred = model.predict(input_scaled)[0]
    
    # 4. LES VÉRIFICATIONS (Assertions)
    # Vérifie que la probabilité est bien entre 0 et 1
    assert 0 <= prob <= 1, f"La probabilité {prob} est hors limites !"
    
    # Vérifie que la prédiction est soit 0 soit 1
    assert pred in [0, 1], f"La classe prédite {pred} est invalide !"