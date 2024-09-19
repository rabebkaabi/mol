

# Étape 1 : Utiliser une image Python de base
FROM python:3.9-slim

# Étape 2 : Définir le répertoire de travail dans le conteneur
C:/Users/rabeb.kaabi/Desktop/Molecule/MoleculePrediction-master/myflaskapp/app

# Étape 3 : Copier le fichier requirements.txt dans le conteneur
COPY requirements.txt requirements.txt

# Étape 4 : Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Étape 5 : Copier tout le contenu de l'application dans le conteneur
COPY . .

# Étape 6 : Exposer le port sur lequel l'application va fonctionner (5000 pour Flask)
EXPOSE 5000

# Étape 7 : Commande d'exécution pour démarrer l'application
CMD ["python", "app.py"]
