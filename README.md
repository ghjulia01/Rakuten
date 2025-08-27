# Rakuten Product Classification – DataScientest x Mines Paris


## Presentation  


Ce projet s’inscrit dans le cadre de la formation **DataScientest – Mines Paris** et du challenge proposé par **Rakuten Institute of Technology** via la plateforme Challenge Data en partenariat avec le **Collège de France**. Il vise à **automatiser la classification des produits vendus sur la marketplace Rakuten** France en s’appuyant à la fois sur des **données textuelles** (titres, descriptions) et **visuelles** (images produits).


## Objectifs

- Construire un modèle de classification supervisée multimodal (texte + image) pour prédire la catégorie prdtypecode des produits.

- Traiter les défis liés au déséquilibre des classes, à la diversité linguistique et à l’hétérogénéité des visuels.

- Proposer une structuration sémantique des catégories pour faciliter l'expérience utilisateur et optimiser la navigation.

-L’approche est multimodale : une branche texte (nettoyage + TF-IDF) et une branche image (chargement, normalisation, PCA), fusionnées puis apprises par un classifieur linéaire, avec rééquilibrage des classes (undersampling/oversampling).

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

### Diagramme

```mermaid
flowchart TB
  %% ===================== STYLES =====================
  classDef phase fill:#f2f6ff,stroke:#5271ff,stroke-width:1px,color:#1f2a44;
  classDef step  fill:#fff,stroke:#999,stroke-width:1px,color:#111;
  classDef io    fill:#fffbe6,stroke:#c9a227,color:#4d3d00;
  classDef tool  fill:#eefaf3,stroke:#2ca46c,color:#083b2c;
  classDef warn  fill:#fff0f0,stroke:#d9534f,color:#7a0b0b;

  subgraph P0[0) Installation & Configuration]:::phase
    P0a[Créer .venv311 & pip install -r requirements.txt]:::step
    P0b[Configurer features/config.toml\n(paths, text, images, sampling, cv, model)]:::step
  end

  subgraph P1[1) Préparation des données]:::phase
    P1a[Verifier CSV:\nX_train_update.csv,\nY_train_CVw08PX.csv,\nX_test_update.csv]:::io
    P1b[Images:\nimage_train/, image_test/]:::io
    P1c[Optionnel – Générer vocabulaire & map:\nfeatures/make_cleaned_frequencies_and_map.py\n→ token_frequencies_cleaned_stem.csv\n→ translate_map_starter_from_cleaned.json]:::tool
  end

  subgraph P2[2) Baselines (références)]:::phase
    B0[Baseline B0\nDummy - most_frequent]:::step
    B1[Baseline B1\nDummy - stratified]:::step
    B2[Baseline B2\nTexte seul:\nTextCleaner → TF-IDF → LR]:::step
    B3[Baseline B3\nImage seule:\nImageLoader → Resize → Flatten → PCA → LR]:::step
    noteB2["Sans resampling, class_weight='balanced'"]:::warn
  end

  subgraph P3[3) Pipeline multimodal (B4)]:::phase
    subgraph TXT[Branche Texte]:::phase
      T1[TextCleaner:\nconcat titre+desc, nettoyage, accents,\nmap EN/DE→FR (option), stemming (option)]:::step
      T2[TF-IDF (1-2g; min_df/max_df/max_features)]:::step
      T3[Features additionnelles:\nHasDescriptionFlag, DesignationLength]:::step
    end
    subgraph IMG[Branche Image]:::phase
      I1[ImageLoader (RGB, resize size=[32|64])\nvaleurs [0,1], fallback image noire]:::step
      I2[Flatten (H×W×C → vecteur)]:::step
      I3[PCA dense (n_components) \n(ou SVD sparse si configuré)]:::step
    end
    FU[FeatureUnion (texte + image + stats)]:::step
    SAMP[Sampling:\nundersampling + oversampling]:::step
    CLF[Classifier: LR ou LinearSVC]:::step
  end

  subgraph P4[4) Entraînement & Prédiction]:::phase
    F1[Fit pipeline sur X_train,y_train]:::step
    F2[Mise à jour du chemin ImageLoader\n→ image_test/ (sans recréer la branche)]:::step
    F3[Predict sur X_test]:::step
  end

  subgraph P5[5) Évaluation & Reporting]:::phase
    R1[CV (stratified K-fold):\nF1 macro & F1 pondéré]:::step
    R2[Export CSV cumulé:\nresults/baseline_results_summary.csv]:::io
    R3[Rapports par baseline:\nresults/report_b*_cv.txt]:::io
    V1[Figures:\nplot_baselines.py → barres F1\nplot_baseline_bars.py → macro vs weighted\nplot_confusion_matrix.py → CM top-K]:::tool
  end

  subgraph P6[6) Sorties & Livraison]:::phase
    O1[models/text_image_classifier.joblib]:::io
    O2[models/y_test_pred.csv]:::io
    O3[models/compare_cv_results.csv (option --compare)]:::io
    G1[README + Diagrams Mermaid]:::tool
    G2[Sync vers dépôt professeur\nvia git subtree --prefix=Julie ...]:::tool
  end

  P0 --> P1 --> P2 --> P3 --> P4 --> P5 --> P6
  T1 --> T2 --> T3 --> FU
  I1 --> I2 --> I3 --> FU
  FU --> SAMP --> CLF

### Feature Engineering 

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
                - “chaussure” apparaît 1 fois → TF = 1
                - “chaussure” apparaît 20 fois → TF = 20
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

- Chargement des images par `productid` via un pipeline scikit-learn dédié (`ImageLoader`).
- Conversion en RGB.
- Redimensionnement homogène (TOML images.size, ex. 32×32 ou 64×64).
- Normalisation des pixels dans [0,1].
- Encodage en vecteurs aplatis exploitables par un modèle linéaire.
- Gestion des images manquantes: vecteur nul (image noire).
- Extraction optionnelle de 3 features simples (width, height, occupancy) pour enrichir les signaux visuels.

#### C-  Réduction de dimension (SVD/PCA)

- Aplatissement des images 
- Les vecteurs aplatis sont denses → privilégier PCA dense (mémoire maîtrisée).
- Réduction optionnelle configurable :
  - **TruncatedSVD** sur données sparse .
  - **PCA dense** (plus adapté).
- Paramètres configurables dans `config.toml` (`enabled`, `method`, `n_components`).
- Conseil RAM : size = [32,32] + n_components ≈ 50~150 pour un bon compromis.
- Objectif : compresser les données visuelles en conservant l’essentiel de l’information et limiter le surapprentissage.

#### D-  Approche multimodale : fusion des embeddings texte et image

Voici un résumé expliquant le flow :

- Branche texte : TextCleaner → TF-IDF → HasDescriptionFlag → DesignationLength
- Branche image : ImageLoader → Flatten → SVD/PCA optionnel
- Fusion : FeatureUnion texte + image_pixels + image_stats
- Rééchantillonnage : RandomUnderSampler + RandomOverSampler
- Modèle final : LogisticRegression ou LinearSVC (choix dans config.toml)



#### E- Hyperparamètres dans config.toml

Points clés (éditables sans toucher au code) :

- Chemins : [paths] (csv) et [images] (répertoires train/test, taille size = [32,32] ou [64,64]).
- Texte : [text] (max_features, min_df, max_df, use_stem, translate_map_path).
- Images : [images.dim_reduction] (enabled, method="pca", n_components, random_state).
- CV : [cv].splits (3 ou 5).
- Équilibrage : [sampling] (major_class, major_cap, tail_min).
- Modèle : [model] (type lr/svc, use_class_weight, etc.).

#### F- Architecture du projet

Architecture du projet
.
- ├── data/
- │   ├── X_train_update.csv
- │   ├── Y_train_CVw08PX.csv
- │   └── X_test_update.csv
- │
- ├── data/images/images/
- │   ├── image_train/   # images d'entraînement
- │   └── image_test/    # images de test
- │
- ├── features/
- │   ├── text_cleaner.py
- │   ├── text_vectorizer.py
- │   ├── image_loader.py
- │   ├── image_stats.py
- │   ├── make_cleaned_frequencies_and_map.py
- │   └── config.toml          # configuration centrale du projet
- │
- ├── main/
- │   └── train_model.py       # orchestration (baselines, CV, pipeline complet)
- │
- ├── models/                  # pipelines pour texte et image
- ├── results/                 # métriques & figures (sorties baselines)
- ├── tools/                   # scripts de reporting (figures)
- │   ├── plot_baselines.py
- │   ├── plot_baseline_bars.py
- │   └── plot_confusion_matrix.py
- └── README.md

#### G- Baselines & Protocole d’évaluation

Nous évaluons 5 références avant/après le multimodal :

- Code	    Baseline	        Description
- B0	        Naïf (majoritaire)	DummyClassifier(strategy="most_frequent")
- B1	        Naïf (stratifié)	DummyClassifier(strategy="stratified")
- B2	        Texte seul	        TextCleaner + TF-IDF → LR (sans rééchantillonnage)
- B3	        Image seule	        ImageLoader → flatten → PCA → LR (sans texte)
- B4	        Multimodal complet	Texte + Image (+ image_stats) + under/over-sampling (pipeline principal)

- Métriques : F1 macro (équité inter-classes) & F1 pondéré.
- Validation : K-fold stratifié (paramétré via TOML).
- Reproductibilité : random_state fixés; config centralisée.

```mermaid
flowchart LR
  B0[**B0** Dummy most_frequent] --> COMP[Comparaison F1]
  B1[**B1** Dummy stratified]   --> COMP
  B2[**B2** Texte seul: TF-IDF→LR] --> COMP
  B3[**B3** Image seule: PCA→LR]   --> COMP
  B4[**B4** Multimodal: Texte+Image+Sampling→LR/SVC] --> COMP

#### H- Comment exécuter le projet

##### Génération du dictionnaire et des fréquences (optionnel si déjà fait)
        python features/make_cleaned_frequencies_and_map.py 
        --x_train_csv data/X_train_update.csv 
        --out_freq features/token_frequencies_cleaned_stem.csv 
        --out_map features/translate_map_starter_from_cleaned.json 
        --config features/config.toml


##### Lancer les baselines

###### B0 / B1
python -m main.train_model --config features/config.toml --baseline b0

python -m main.train_model --config features/config.toml --baseline b1

###### B2 (texte seul)
python -m main.train_model --config features/config.toml --baseline b2

###### B3 (image seule)
python -m main.train_model --config features/config.toml --baseline b3

##### Lancer le modèle multimodal (B4)
python -m main.train_model --config features/config.toml

##### (Option) Comparer LR vs SVC (CV)
python -m main.train_model --config features/config.toml --compare

#### I- Visualisations & Rapports

##### 1) Barres (une métrique) – F1 macro par défaut
python tools/plot_baselines.py
###### autre métrique & sortie
python tools/plot_baselines.py --metric f1_weighted --out results/figures/baseline_f1_weighted.png

##### 2) Barres groupées – F1 macro vs F1 pondéré
python tools/plot_baseline_bars.py
###### imposer l’ordre : B0→B4
python tools/plot_baseline_bars.py --order B0 B1 B2 B3 B4

##### 3) Matrice de confusion (top-K classes)
###### Texte seul (B2), 3 folds, normalisée, top 25 classes
python tools/plot_confusion_matrix.py --config features/config.toml --baseline b2

###### Multimodal (B4), 5 folds, non normalisée, top 30
python tools/plot_confusion_matrix.py --config features/config.toml --baseline b4 --splits 5 --normalize false --topk 30


##### Sorties & Organisation des résultats

###### Baselines

- results/baseline_results_summary.csv – cumul des runs (append)
- results/report_b0_cv.txt, … – classification_report par baseline (CV)
- results/figures/*.png – bar charts & matrices de confusion (+ CSV agrégés)
- results/figures/baseline_f1_macro.png, baseline_f1_bars.png, …
- results/figures/cm_<baseline>.png + cm_<baseline>_full.csv + report_<baseline>_cv.txt

###### Modèle complet (B4)

- models/text_image_classifier.joblib – pipeline entraîné
- models/y_test_pred.csv – prédictions sur X_test
- models/compare_cv_results.csv – si --compare activé


#### J- Bonnes pratiques & Dépannage

##### Import models introuvable : 
lancer depuis la racine avec python -m main.train_model ... et vérifier les __init__.py.

##### Ajustement test images sans recréer la branche : 
On met à jour le chemin du ImageLoader dans la pipeline fit (pas de recréation) → dimension identique train/test.

##### Convergence LR : 
augmenter max_iter (ex. 5000) ou relâcher tol.

##### Mémoire images :

Éviter SVD sparse sur pixels denses,
Préférer PCA dense,
Réduire images.size si nécessaire.

##### Équilibrage : 
Toujours comparer B4 à B2 pour mesurer l’apport de l’image.

#### K- Changelog (Rendu 2)

- Ajout des baselines B0–B3 + comparaison B4 (multimodal).
- Export automatique CSV + TXT + figures (barres, matrice de confusion).
- Refactor prédiction images : repointage du ImageLoader test sans recréer la branche (stabilité dimensionnelle).
- Paramétrage centralisé (TOML) et scripts de visualisation (tools/…).

## Installation

### Cloner ce dépôt :

git clone https://github.com/ghjulia01/Rakuten.git
cd jul25_bootcamp_ds_classification-de-produits-e--commerce-rakuten-main

### Créer un environnement virtuel (Windows)

python -m venv .venv311

### Activer un environnement virtuel (Windows)

#### Sous PowerShell :
.venv311\Scripts\Activate.ps1

#### Ou sous CMD :
.venv311\Scripts\activate.bat

#### (Sous Linux/Mac, utilisez `source .venv311/bin/activate`)

### Installer les dépendances :

#### Si requirements.txt existe déjà :
pip install -r requirements.txt

#### Sinon, générer d'abord le fichier requirements.txt avec le script fourni :
python tools/generate_requirements.py
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
