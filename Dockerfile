# Stage 1: Build stage
FROM python:3.10-slim AS builder

# Variables d’environnement pour désactiver le cache et améliorer la performance
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Définir le répertoire de travail
WORKDIR /app

# Installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copier le code de l’application
COPY app.py /app
COPY models /app/models

# Stage 2: Runtime stage
FROM python:3.10-slim

# Variables d’environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Répertoire de travail
WORKDIR /app


# Copier uniquement les fichiers nécessaires depuis le builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

# Créer les répertoires nécessaires et ajuster les permissions
# RUN mkdir -p /app/data/Healthy /app/data/Blight /app/data/Common_Rust /app/data/Gray_Leaf_Spot && \
#     mkdir /app/temp && \
#     chmod -R 777 /app/temp /app/data

# # Ajouter un utilisateur non-root et lui attribuer les permissions nécessaires
# RUN adduser --disabled-password appuser && \
#     chown -R appuser:appuser /app

# # Changer d’utilisateur pour éviter d’exécuter en tant que root
# USER appuser

# Exposer le port de l’application
EXPOSE 8000

# Healthcheck pour surveiller l’état de l’application
HEALTHCHECK CMD curl --fail http://localhost:7860/health || exit 1

# Commande pour lancer l’application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]