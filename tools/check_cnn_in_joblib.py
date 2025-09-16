# tools/check_cnn_in_joblib.py
import argparse, sys
from pathlib import Path
import importlib
import joblib

# 1) S'assurer que le repo racine est sur le path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 2) IMPORTER le module qui contient les classes picklées,
#    puis les exposer dans __main__ (là où le pickle les cherche)
try:
    tm = importlib.import_module("main.train_model")
    import __main__ as MAIN
    # Classes custom susceptibles d'être dans le pickle
    for name in ("ToFloat32", "AdaptiveUnderSampler"):
        if hasattr(tm, name):
            setattr(MAIN, name, getattr(tm, name))
except Exception as e:
    print(f"[WARN] Impossible de préparer les alias __main__ → {e}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="artifacts/b4.joblib",
                   help="Chemin vers le modèle .joblib")
    args = p.parse_args()

    pipe = joblib.load(args.model)  # <= maintenant le pickle retrouve les classes
    print(f"Loaded: {type(pipe).__name__}")

    # Récupération de la FeatureUnion globale
    try:
        features = pipe.named_steps["features"]
    except KeyError:
        names = list(getattr(pipe, "named_steps", {}).keys())
        raise SystemExit(f"Étape 'features' introuvable. Étapes présentes: {names}")

    names = [name for name, _ in features.transformer_list]
    print("Branches présentes dans FeatureUnion:", names)
    weights = getattr(features, "transformer_weights", None)
    print("Poids (si définis) :", weights)
    print(f"\nCNN présente ? {'OUI' if 'image_cnn' in names else 'NON'}")

if __name__ == "__main__":
    main()