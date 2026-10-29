
FROM python:3.12-slim

WORKDIR /app

# Installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code et les artefacts nécessaires
COPY src/ ./src/
COPY mlartifacts/ ./mlartifacts/
COPY Notebooks/mlruns/ ./Notebooks/mlruns/

# Exposer le port FastAPI
EXPOSE 8000

# Démarrer l'API
CMD ["uvicorn", "src.app.app:app", "--host", "0.0.0.0", "--port", "8000"]
