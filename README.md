# OC_P8
Participation à la compétition Kaggle Predict Future Sales.

Les données sont téléchargeables via l'onglet Data de la compétition: https://www.kaggle.com/c/competitive-data-science-predict-future-sales/data.
Elles sont également récupérables via l'API Kaggle:

# from kaggle.api.kaggle_api_extended import KaggleApi
# api = KaggleApi()
# api.authenticate()
# files = api.competition_download_files("competitive-data-science-predict-future-sales")

Le script preprocessing.py est commun à toutes les branches.
Les scripts model_*.py permettent de générer les fichiers CSV à soumettre.

Dépendances pour les modèles:
sklearn
lightgbm
xgboost
catboost
tensorflow
