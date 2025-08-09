
# Test suite — Branche textuelle (TF-IDF + features)

## Arborescence proposée
Placez ce dossier `tests/` à la racine de votre projet (là où résident `features/` et `models/`).

```
your-project/
  features/
    text_cleaner.py
    text_vectorizer.py
    text_pipeline.py
  tests/
    conftest.py
    test_designation_length_and_has_desc.py
    test_text_cleaner.py
    test_text_tfidf_vectorizer.py
    test_text_pipeline_integration.py
    test_translate_map_loader.py
    test_serialization_roundtrip.py
    test_edge_cases.py
  ...
```

## Pré-requis
- `pytest`
- `nltk` + stopwords (le téléchargement est automatisé dans `conftest.py`)
- `scikit-learn`, `pandas`, `numpy`

## Lancer les tests
```bash
pytest -q
```

## Notes
- Si votre layout de modules diffère (ex. `from models.text_cleaner import ...`), adaptez les imports au début des tests.
- Les tests vérifient la cohérence train/test, la sérialisation, et des cas limites (NaN, champs manquants).
- La feature `DesignationLength` doit être intégrée dans `FeatureUnion` de `create_text_pipeline` sous le nom `("title_len", DesignationLength())`.
