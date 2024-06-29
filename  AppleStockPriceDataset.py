#Importer les bibliothèques nécessaires

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from google.colab import drive


####################


#Charger le jeu de données

file_path = 'Apple_Dataset.csv' # Chemin vers le fichier CSV 
data = pd.read_csv(file_path)

data.head() # Aperçu des données


####################


#Nettoyer le jeu de données

data.isnull().sum() # Vérifier les valeurs manquantes
data.dropna(inplace=True) # Supprimer les lignes avec des valeurs manquantes
data['Date'] = pd.to_datetime(data['Date']) # Convertir la colonne 'Date' en datetime
data.sort_values('Date', inplace=True) # Trier les données par date


####################

#Analyse exploratoire des données (EDA)

# Visualiser les prix de clôture au fil du temps
plt.figure(figsize=(14, 7)) # Définit la taille de la figure
plt.plot(data['Date'], data['Close']) # Trace les données de prix de clôture par date
plt.title('Apple Stock Price Over Time') # Ajoute un titre au graphique
plt.xlabel('Date') # Ajoute une étiquette à l'axe des x
plt.ylabel('Close Price') # Ajoute une étiquette à l'axe des y
plt.grid(True) # Ajoute une grille pour faciliter la lecture
plt.show() # Affiche le graphique

# Visualiser la corrélation entre les variables
plt.figure(figsize=(10, 8)) #  Crée une nouvelle figure avec une taille spécifique de 10 pouces de large et 8 pouces de haut
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
# Utilisation de la bibliothèque seaborn, créer une carte de chaleur (heatmap) représentant la matrice de corrélation des variables dans le DataFrame data
plt.show() # Affiche le graphique


####################


#Préparation des données pour la modélisation

y = data['Close'] # Utiliser 'Close' comme variable cible
X = data.drop(['Date', 'Close'], axis=1) # Utiliser les autres colonnes comme caractéristiques
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Division des données en ensembles d'entraînement et de test


####################


# Entraînement et évaluation du modèle

model = RandomForestRegressor(n_estimators=100, random_state=42) # Modélisation avec un RandomForestRegressor
model.fit(X_train, y_train)

y_pred = model.predict(X_test)# Prédiction sur l'ensemble de test

# Évaluation des performances du modèle
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"R^2 Score: {r2_score(y_test, y_pred)}")


####################


#Optimisation et amélioration du modèle
from sklearn.model_selection import GridSearchCV
# Définir les paramètres à tester
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# GridSearchCV pour trouver les meilleurs hyperparamètres
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Meilleurs hyperparamètres
print(f"Best parameters: {grid_search.best_params_}")
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

# Évaluation des performances du meilleur modèle
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_best)}")
print(f"R^2 Score: {r2_score(y_test, y_pred_best)}")