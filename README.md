<p align="center">
  <img src="docs/banner" alt="Bannière du projet" width="1000" height="250"/>
</p> 



# 🏦 Bank Default Prediction : End-to-End MLOps Project
> **Project developed for the "Sorbonne Data Analytics" Professional Graduate Degree at Université Paris 1 Panthéon-Sorbonne.**


### 🛠️ Tech Stack & MLOps Pipeline

![ML Focus](https://img.shields.io/badge/Focus-Bank_Default_Prediction-E11922?style=for-the-badge&logo=target&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

![ML Models](https://img.shields.io/badge/Models-XGBoost_%7C_Random_Forest_%7C_MLflow-blueviolet?style=for-the-badge&logo=scikit-learn&logoColor=white)
![UI](https://img.shields.io/badge/Interface-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

![Docker](https://img.shields.io/badge/Container-Docker_Desktop_%7C_Hub-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![CI/CD](https://img.shields.io/badge/Pipeline-GitHub_Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)

![AWS](https://img.shields.io/badge/Cloud-AWS_ECS_Fargate_%7C_ECR-FF9900?style=for-the-badge&logo=amazon-aws&logoColor=white)
![Version](https://img.shields.io/badge/VCS-Git_%7C_GitHub-F05033?style=for-the-badge&logo=git&logoColor=white)

---

## 🚀 Présentation du Projet
Ce projet implémente une démarche **End-to-End (E2E)** pour prédire le risque de défaut de paiement. L'application est conçue comme un outil d'aide à la décision pour les conseillers bancaires, permettant d'évaluer la solvabilité via le score FICO et d'autres indicateurs financiers.

* **Lien de l'application :** `http://35.181.169.159:8501/`

## 📊 Phase d'Expérimentation (MLflow)
Avant le déploiement, une phase comparative a été menée sur 4 modèles :
* **Modèles testés :** Régression Logistique, Arbre de Décision, Random Forest, XGBoost.
* **Tracking :** Utilisation de **MLflow** pour suivre les hyperparamètres et les performances.
* **Sélection :** **XGBoost** a été retenu pour ses performances supérieures en termes de **Recall** et d'**AUC-ROC**.

## 🏗️ Architecture Cloud & CI/CD
Le déploiement est entièrement automatisé pour garantir une mise en production continue :

1. **Dockerisation :** L'application Streamlit est encapsulée dans un container Docker.
2. **CI/CD (GitHub Actions) :** - Build automatique de l'image lors d'un `git push`.
   - Stockage sur **Amazon ECR**.
   - Déploiement "Rolling Update" sur **Amazon ECS (Fargate)**.
3. **Monitoring :** Gestion des logs **

## 💻 Paramètres de l'Application
L'interface Streamlit permet de simuler un profil client avec les variables suivantes :
* **Dette totale**, **Revenu annuel**, **Années d'ancienneté**, **Score FICO**.
* **Output :** Probabilité de défaut, analyse des variables déterminantes et commentaire automatisé.


Ce projet consiste en la mise en place d'une solution complète de Machine Learning (E2E) pour prédire le risque de défaut de paiement des clients bancaires. L'objectif est de fournir un outil d'aide à la décision intuitif, déployé via une architecture cloud robuste.

## 🚀 Aperçu du Projet
L'application permet aux conseillers bancaires de saisir les informations financières d'un client pour obtenir instantanément une probabilité de défaut, accompagnée d'une explication des variables déterminantes.

* **Interface en ligne :** [LIEN_VERS_TON_APP_AWS]
* **Stack Technique :** Python, XGBoost, Streamlit, Docker, AWS (ECS/ECR), GitHub Actions.

---

## 📊 Pipeline de Machine Learning

### 1. Données & Preprocessing
Le dataset contient des informations financières critiques (Score FICO, revenus, dettes).
* **Features :** `credit_lines_outstanding`, `loan_amt_outstanding`, `total_debt_outstanding`, `income`, `years_employed`, `fico_score`.
* **Preprocessing :** Normalisation des données et gestion des scores FICO.

### 2. Expérimentations & Choix du Modèle
Quatre modèles ont été mis en compétition et suivis via **MLflow** :
* Logistic Regression
* Decision Tree
* Random Forest
* **XGBoost** (Sélectionné pour ses performances supérieures)

**Métriques clés :** Focus sur le **Recall** (pour minimiser les faux négatifs) et l'**AUC-ROC**.

---

## 🏗️ Architecture MLOps & Déploiement

Le projet implémente une démarche CI/CD automatisée :

1.  **Développement :** Code écrit sous VS Code, versionné avec Git.
2.  **Containerisation :** Création d'une image Docker optimisée.
3.  **CI/CD Pipeline :** À chaque `push` sur GitHub, une **GitHub Action** :
    * Build l'image Docker.
    * La pousse sur **Amazon ECR**.
    * Déploie automatiquement la nouvelle version sur **Amazon ECS (Fargate)**.
4.  **Hébergement :** L'application est servie sur le port 8501 via AWS.



1. **Cloner le projet :**
   ```bash
   git clone [https://github.com/TON-PSEUDO/TON-REPO.git](https://github.com/TON-PSEUDO/TON-REPO.git)
   cd TON-REPO
