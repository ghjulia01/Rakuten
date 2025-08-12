
# Test suite — Branche image (chargement + stats + pipeline)

## Contenu
- `test_image_loader.py` : tests unitaire d'ImageLoader (forme, normalisation, missing)
- `test_image_stats.py`  : tests d'ImageStatsFeaturizer (colonnes, dtypes, thresholds, set_image_dir)
- `test_image_pipeline_integration.py` : intégration de `create_image_pipeline` (sparse CSR, dimensions, invariance à l'ordre)

## Pré-requis
- `pytest`, `Pillow` (PIL), `numpy`, `scipy`

## Lancer les tests
Placez `tests_images/` à la racine du projet (même niveau que `features/` et `models/`) puis :
```bash
pytest -q tests_images
```
