import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 1. CONFIGURATION
st.set_page_config(page_title="Credit Score Pro", page_icon="🛡️", layout="wide")

# STYLE CSS MIS À JOUR (Harmonisation des tailles et suppression des espaces inutiles)
st.markdown("""
    <style>
    .result-box {
        height: 100px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 10px;
        font-size: 25px; /* Taille identique pour lettres et chiffres */
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .main-box {
        border: 1px solid #e6e9ef;
        padding: 15px;
        border-radius: 12px;
        background-color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. CHARGEMENT DES ASSETS
@st.cache_resource
def load_assets():
    model = joblib.load('xgb_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()
feature_names = ['loan_amt_outstanding', 'total_debt_outstanding', 'income', 'years_employed', 'fico_score']
labels_fr = ['Prêt restant', 'Dette totale', 'Revenu annuel', 'Années emploi', 'Score FICO']

# 3. ONGLETS
tab1, tab2, tab3 = st.tabs(["🚀 Prédiction", "📊 Analyse Modèle", "📖 Glossaire"])

with tab1:
    st.title("🛡️ Analyse de Risque")
    
    with st.sidebar:
        st.header("📋 Profil du Client")
        l_amt = st.number_input("Prêt restant ($)", value=5000, step=500)
        t_debt = st.number_input("Dette totale ($)", value=15000, step=500)
        inc = st.number_input("Revenu annuel ($)", value=45000, step=1000)
        y_emp = st.slider("Années d'emploi", 0, 40, 5)
        fico_s = st.slider("Score FICO", 300, 850, 650)
        predict_btn = st.button("Lancer l'Analyse")

    if predict_btn:
        # Calculs
        input_df = pd.DataFrame([[l_amt, t_debt, inc, y_emp, fico_s]], columns=feature_names)
        input_scaled = scaler.transform(input_df)
        prob = model.predict_proba(input_scaled)[0][1]
        pred = model.predict(input_scaled)[0]

        col_left, col_right = st.columns([1, 1.2])

        with col_left:
            # BLOC 1 : Verdict (Texte)
            if pred == 1:
                st.markdown(f'<div class="result-box" style="background-color: #ffebee; color: #c62828;">❌ CRÉDIT REFUSÉ</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-box" style="background-color: #e8f5e9; color: #2e7d32;">✅ CRÉDIT ACCORDÉ</div>', unsafe_allow_html=True)
            
            # BLOC 2 : Probabilité (Chiffres - Même taille que le verdict)
            st.markdown(f'<div class="result-box" style="background-color: #f1f3f4; color: #3c4043;">Risque : {prob:.2%}</div>', unsafe_allow_html=True)
            
            # Note de synthèse (Plus compacte sous les deux blocs)
            if prob < 0.2: stat, rec, col = "Faible", "Approuver sans réserve.", "#2e7d32"
            elif prob < 0.5: stat, rec, col = "Modéré", "Vérifier les garanties.", "#ef6c00"
            else: stat, rec, col = "Élevé", "Risque critique détecté.", "#c62828"

            # Récupération du facteur déterminant
            importances_g = model.feature_importances_
            contribution_l = input_scaled[0] * importances_g
            facteur_cle = labels_fr[np.argmax(np.abs(contribution_l))]

            st.markdown(f"""
                <div style="border-left: 5px solid {col}; padding: 15px; background-color: #ffffff; border-radius: 8px; box-shadow: 0px 2px 5px rgba(0,0,0,0.05); color: #000000;">
                    <p style="margin:0; font-size:16px; color: #000000;"><strong>Diagnostic :</strong> Risque {stat}</p>
                    <p style="margin:0; font-size:16px; color: #000000;"><strong>Facteur Clé :</strong> {facteur_cle}</p>
                    <p style="margin:0; font-size:16px; color: #000000;"><strong>Conseil :</strong> {rec}</p>
                </div>
                """, unsafe_allow_html=True)

        with col_right:
            st.write("**Analyse des facteurs de risque :**")
            
            # 1. Récupération des importances globales
            importances_g = model.feature_importances_
            
            # 2. Calcul de la contribution brute (Valeur normalisée * Poids)
            contribution_brute = input_scaled[0] * importances_g
            
            # 3. Renversement de la logique pour les variables "positives"
            # On veut que : Positif (Rouge) = Augmente le risque | Négatif (Bleu) = Diminue le risque
            display_contributions = []
            for i, feat in enumerate(feature_names):
                val = contribution_brute[i]
                
                # Pour ces variables, une valeur élevée est une BONNE chose.
                # Donc si la valeur est haute (positive), on l'inverse pour qu'elle apparaisse en BLEU (négatif).
                if feat in ['income', 'years_employed', 'fico_score']:
                    display_contributions.append(-val)
                else:
                    # Pour la dette et le prêt, une valeur haute est une MAUVAISE chose.
                    # On garde donc le signe positif pour que ce soit en ROUGE.
                    display_contributions.append(val)

            # 4. Création du graphique
            fig, ax = plt.subplots(figsize=(7, 4.5))
            
            # Code couleur : Rouge pour le risque, Bleu pour la sécurité
            colors = ['#FF4B4B' if x > 0 else '#1C83E1' for x in display_contributions]
            
            # On utilise les labels en français pour l'affichage
            ax.barh(labels_fr, display_contributions, color=colors)
            
            # Esthétique du graphique
            ax.axvline(0, color='black', linewidth=0.8) # Ligne centrale à zéro
            ax.set_xlabel("← Diminue le Risque | Augmente le Risque →", fontsize=10)
            plt.tight_layout()
            
            st.pyplot(fig)
    else:
        st.info("Ajustez les paramètres à gauche pour démarrer l'analyse.")

# --- ONGLET 2 : ANALYSE DU MODÈLE ---
with tab2:
    st.header("⚙️ Performance et Importance Globale")
    importances = model.feature_importances_
    feat_imp_df = pd.DataFrame({'Variable': labels_fr, 'Importance': importances}).sort_values(by='Importance', ascending=True)
    
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.barplot(x='Importance', y='Variable', data=feat_imp_df, palette='viridis', ax=ax2)
    st.pyplot(fig2)
    st.info("La variable 'Dette totale' reste statistiquement le prédicteur le plus puissant du jeu de données.")

# --- ONGLET 3 : GLOSSAIRE ---
with tab3:
    st.header("📖 Description des Variables")
    st.write("Pour bien comprendre l'analyse, voici la définition de chaque paramètre utilisé :")
    
    st.markdown("""
    - **Prêt restant (`loan_amt_outstanding`)** : Le montant total du prêt actuel qu'il reste à rembourser par le client.
    - **Dette totale (`total_debt_outstanding`)** : Cumul de toutes les dettes en cours du client (crédits conso, immobilier, revolving, etc.).
    - **Revenu annuel (`income`)** : Salaire et autres revenus déclarés sur une année. Un revenu élevé compense souvent une dette importante.
    - **Années d'emploi (`years_employed`)** : Ancienneté dans le poste actuel ou stabilité professionnelle. Plus elle est élevée, plus le risque diminue.
    - **Score FICO (`fico_score`)** : Note de solvabilité (entre 300 et 850). C'est l'indicateur standard de la fiabilité d'un emprunteur.
    ---
    *Note : Toutes les variables sont normalisées avant d'être envoyées au modèle XGBoost.*
    """)