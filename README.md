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

#### A- Traitement du texte – Pipeline Rakuten

##### 1. Objectif

Le texte produit (désignation + description) est une source d’information clé pour prédire la catégorie des produits.
Notre objectif est de :
    - nettoyer les données textuelles (supprimer le bruit, harmoniser la langue, gérer les formats)
    - représenter le texte sous forme vectorielle exploitable par les modèles
    - enrichir avec des signaux simples mais informatifs (présence d’une description)

##### 2. Sources utilisées

- designation : nom du produit
- description : texte descriptif du produit

Remarque : l’existence même d’une description est un signal potentiel (produits plus formalisés → catégories spécifiques).
Nous l’avons donc conservée sous forme d’une variable binaire : has_description.

##### 3. Étapes de traitement

###### 3.1. Fusion & indicateur has_description

On fusionne designation et description pour obtenir un texte unique (full_text).
On ajoute has_description = 1 si description non vide, sinon 0.

Raison :
La fusion maximise la quantité d’informations lexicales.
L’indicateur binaire capture la valeur informative de la présence/absence d’une description.

###### 3.2. Nettoyage avec TextCleaner

Le TextCleaner applique, dans cet ordre :

- Suppression HTML
- Retire les balises <...> et entités (&amp;, &#39;…)
- Évite que le modèle apprenne sur du bruit syntaxique.
- Passage en minuscules
- Uniformise le vocabulaire (iPhone → iphone).
- Suppression de la ponctuation & caractères non-alphanumériques
- Nettoie pour se concentrer sur les tokens utiles.
- Suppression des stopwords
- Liste FR enrichie pour retirer les “mots vides” (ex. “blanc”, “actuel”, “moderne”)
- Traduction ciblée (translate_map)
    - Utilise un dictionnaire EN/DE → FR généré via le script
    - make_cleaned_frequencies_and_map.py :
- Extraction des tokens fréquents non français après nettoyage.
- Construction d’un mapping vers leur traduction française.
- Ce mapping est chargé automatiquement depuis config/.
- Stemming (Snowball, FR)
    - Réduction des mots à leur racine (voitures → voitur).
    - Choix du stemming plutôt que de la lemmatisation car :
        - Corpus multilingue → moins dépendant d’un modèle linguistique FR pur.
        - Plus rapide et robuste aux variantes morphologiques.
Nous avons préféré garder à ce stade les caractères numériques qui pouvaient nous aider 
à caractériser des produits (exemple PS3).

###### 3.3. Vectorisation avec TextTfidfVectorizer

Après nettoyage, les textes passent par une vectorisation TF-IDF :
    - max_features = 5 000 (paramétrable)
    - ngram_range = (1, 2) → unigrams + bigrams
        - Unigram = un seul mot
            - Ex. : “chaussure”, “cuir”, “homme”
        - Bigram = séquence de 2 mots consécutifs
            - Ex. : “chaussure cuir”, “cuir homme”
        - Les unigrams capturent le vocabulaire général.
        - Les bigrams capturent des expressions significatives qui ont un sens particulier dans un contexte produit.
            - Par ex., “acier inox” ≠ “acier” + “inox” séparés.
            - “coque iphone” ≠ “coque” + “iphone” de manière indépendante.
    - min_df=2 / max_df=0.95 : suppression des termes trop rares ou trop fréquents
    - sublinear_tf=True : pondération logarithmique pour limiter l’impact des répétitions 
        - Par défaut, TF-IDF calcule la TF (Term Frequency) = nombre d’occurrences du mot dans le document.
            - Exemple :
                “chaussure” apparaît 1 fois → TF = 1
                “chaussure” apparaît 20 fois → TF = 20
        - Avec sublinear_tf=True, on applique la transformation logarithmique:
            - 1 occurrence → 1 + log ( 1 ) = 1 
            - 20 occurrenceS → 1 + log ( 20 ) = 4.3
            L’écart entre 1 et 20 occurrences est fortement réduit.
            Cela évite que des mots répétés de manière artificielle dans un texte (ou spam) dominent le score TF-IDF.
            Dans les descriptions produit, certains mots (“neuf”, “promotion”, “livraison”) peuvent être répétés sans apporter d’info nouvelle, cette option réduit leur poids relatif.
    - strip_accents='unicode' : harmonise les variantes accentuées
    - lowercase=False : déjà fait en amont
    - dtype='float32' : mémoire optimisée

#### B-  Traitement d’images (normalisation, redimensionnement, encodage)

#### C-  Réduction de dimension (PCA envisagé)

#### D-  Approche multimodale : fusion des embeddings texte et image

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

**Streamlit Application** (prochaines étapes)

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
