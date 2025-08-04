# Rakuten Product Classification – DataScientest x Mines Paris


## Presentation and Installation


Ce projet s’inscrit dans le cadre de la formation **DataScientest – Mines Paris** et du challenge proposé par **Rakuten Institute of Technology** via la plateforme Challenge Data en partenariat avec le **Collège de France**. Il vise à **automatiser la classification des produits vendus sur la marketplace Rakuten** France en s’appuyant à la fois sur des **données textuelles** (titres, descriptions) et **visuelles** (images produits).


## Objectifs

- Construire un modèle de classification supervisée multimodal (texte + image) pour prédire la catégorie prdtypecode des produits.

- Traiter les défis liés au déséquilibre des classes, à la diversité linguistique et à l’hétérogénéité des visuels.

- Proposer une structuration sémantique des catégories pour faciliter l'expérience utilisateur et optimiser la navigation.

## Methodologie

### Exploration et Préprocessing

- Fusion des données texte et visuelles via imageid et productid

- Traitement des valeurs manquantes (35% de description manquantes)

- Construction d’un dictionnaire de mots vagues multilingues pour améliorer les visualisations

- Création de colonnes auxiliaires (ex. image_name) pour faciliter les jointures

#### Visualisation

- Nuages de mots bruts et nettoyés

- Affichage d’images par catégorie

- Diagrammes de distribution des classes (déséquilibrées)

- Arborescence thématique (ex. : Jeux & gaming > Accessoires gaming)

#### Observations et résultats de l'étape EDA

- 27 catégories identifiées dans Y_train, très déséquilibrées (de 764 à 10 209 produits)

- Définition d’un nom standardisé des catégories inspiré des marketplaces

- Structuration de 8 thématiques principales (Jeux & Gaming, Livres & Presse, Maison & Jardin, etc.)

- Mise en place d’un pipeline pour visualiser, nettoyer, et interpréter les données

#### Prochaines étapes

- Entraînement d’un modèle de classification multimodale (texte + image)

- Mise en place d’un modèle en cascade : d’abord prédiction de la thématique, puis de la catégorie

- Évaluation via F1-score pondéré pour gérer le déséquilibre des classes

- Déploiement en démonstration Streamlit

### Feature Engineering (prochaine étapes dont voici les orientations principales)

- Vectorisation des textes (designation et description)

- Traitement d’images (normalisation, redimensionnement, encodage)

- Réduction de dimension (PCA envisagé)

- Approche multimodale : fusion des embeddings texte et image

## Installation

### Cloner ce dépôt :

git clone https://github.com/ghjulia01/Rakuten.git
cd rakuten_product_classification

### Créer un environnement virtuel

python -m venv env
source env/bin/activate

### Installer les dépendances :

pip install -r requirements.txt

### Télécharger les données (fournies dans le cadre du challenge Rakuten) :

Il est obligatoire de s'enregistrer au challenge pour pouvoir accéder aux données.

- X_train.csv, Y_train.csv, X_test.csv

- images.zip à extraire dans ./data/images/


## Licence

Ce projet utilise des données propriétaires de Rakuten, mises à disposition uniquement à des fins de formation et de compétition. Toute réutilisation est interdite sans autorisation.

**Streamlit Application**

Une application Streamlit est proposée pour visualiser les résultats :

## Presentation and Installation

Fonctionnalités qui seront disponibles :

- Visualisation d’images par catégorie

- Analyse des mots-clés par catégorie

- Nuages de mots (avec ou sans nettoyage)

- Statistiques descriptives

- Arborescence thématique des catégories


## Streamlit App

**Add explanations on how to use the app.**

To run the app (be careful with the paths of the files in the app):

```shell
conda create --name my-awesome-streamlit python=3.9
conda activate my-awesome-streamlit
pip install -r requirements.txt
streamlit run app.py
```

The app should then be available at [localhost:8501](http://localhost:8501).
