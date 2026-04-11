FROM python:3.12.5-slim

# 1. Dossier de travail
WORKDIR /app

# 2. Installation des dépendances système pour XGBoost
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 3. Étape stratégique : On installe les libs AVANT de copier le code
# Cela permet d'utiliser le cache de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 4. Maintenant on copie le reste du projet
COPY . .

# 5. On expose le port par défaut de Streamlit
EXPOSE 8501

# 6. La bonne commande pour Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]