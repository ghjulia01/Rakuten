from pathlib import Path
import sys, traceback
import numpy as np
import pandas as pd


import types, sys
# python tools/check_export.py


# Cherche ToFloat32 dans plusieurs modules possibles (selon versions)
ToFloat32 = None
for modpath in ("models.cnn_features", "tools.cnn_features", "features.transforms"):
    try:
        mod = __import__(modpath, fromlist=["ToFloat32"])
        if hasattr(mod, "ToFloat32"):
            ToFloat32 = getattr(mod, "ToFloat32")
            break
    except Exception:
        pass

# Fallback: définit une version basique de ToFloat32
if ToFloat32 is None:
    class ToFloat32:
        def __call__(self, x):
            try:
                import numpy as np
                if hasattr(x, "numpy"):  # torch Tensor
                    return x.float()
                if hasattr(x, "astype"): # numpy array
                    return x.astype("float32")
                # PIL.Image -> numpy then float32
                try:
                    import numpy as np
                    return np.asarray(x, dtype="float32")
                except Exception:
                    return x
            except Exception:
                return x

# Injecte dans __main__ pour que joblib puisse le retrouver
sys.modules["__main__"].ToFloat32 = ToFloat32

import joblib
# python tools/check_export.py
# --- Fonction utilitaire pour remonter au repo root
def find_repo_root(start: Path) -> Path:
    """Remonte jusqu'à trouver un dossier contenant 'artifacts' (max 6 niveaux)."""
    cur = start.resolve()
    for _ in range(6):
        if (cur / "artifacts").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    # fallback: parent direct
    return start.resolve().parents[1]

# --- Localisation robuste du repo root (marche depuis tools/, scripts/, etc.)
HERE = Path(__file__).resolve()
REPO_ROOT = find_repo_root(HERE)
ARTIFACT = REPO_ROOT / "artifacts" / "b4.joblib"

print(f"[INFO] __file__:    {HERE}")
print(f"[INFO] Repo root:  {REPO_ROOT}")
print(f"[INFO] Artifact:   {ARTIFACT}")


# --- S'assurer que les modules custom du repo sont importables pour joblib.load
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
    print(f"[INFO] sys.path[0] = {sys.path[0]}")

# --- Vérifs de base
if not ARTIFACT.exists():
    print("[ERR] Artifact introuvable :", ARTIFACT)
    sys.exit(1)

from main.train_model import ToFloat32, LabelEncodingClassifier  
import sys
sys.modules["__main__"].ToFloat32 = ToFloat32
sys.modules["__main__"].LabelEncodingClassifier = LabelEncodingClassifier

# --- Charge le pipeline exporté
try:
    pipe = joblib.load(ARTIFACT)
    print("[OK] joblib.load réussi.")
except ModuleNotFoundError as e:
    print("[ERR] Module manquant lors du chargement du .joblib :", e)
    print("      -> Lance le script depuis la racine du projet OU garde ce patch sys.path")
    sys.exit(1)
except Exception as e:
    print("[ERR] Échec joblib.load :", e)
    traceback.print_exc()
    sys.exit(1)

# --- Fix prédiction: garantir que chaque sous-transformeur de l'Union a transform()
from sklearn.base import BaseEstimator, TransformerMixin

class _AsTransformer(BaseEstimator, TransformerMixin):
    """Adapter: expose transform() en appelant fit_transform() si transform() n'existe pas."""
    def __init__(self, inner):
        self.inner = inner
    def fit(self, X, y=None):
        # l'objet inner est déjà fitted (exporté), on renvoie self
        return self
    def transform(self, X):
        if hasattr(self.inner, "transform"):
            return self.inner.transform(X)
        if hasattr(self.inner, "fit_transform"):
            # safe fallback si l'objet ne fournit pas transform()
            return self.inner.fit_transform(X, None) if "y" in self.inner.fit_transform.__code__.co_varnames else self.inner.fit_transform(X)
        raise AttributeError(f"{self.inner.__class__.__name__} n'expose ni transform() ni fit_transform().")

def _ensure_union_transform(union_like):
    """Remplace dans transformer_list tout sous-pipeline sans transform() par _AsTransformer(...)."""
    try:
        tlist = list(getattr(union_like, "transformer_list", []))
        if not tlist:
            return
        new_list = []
        for name, trans in tlist:
            if hasattr(trans, "transform"):
                new_list.append((name, trans))
            elif hasattr(trans, "fit_transform"):
                new_list.append((name, _AsTransformer(trans)))
            else:
                print(f"[WARN] Sous-transformeur '{name}' ignoré (ni transform, ni fit_transform): {type(trans)}")
        union_like.transformer_list = new_list
    except Exception as e:
        print(f"[WARN] Patch union transform impossible: {e}")

# Récupérer l'union à l'intérieur de pipe.named_steps["features"]
features_step = pipe.named_steps.get("features")
if features_step is not None:
    union = features_step.named_steps["union"] if hasattr(features_step, "named_steps") and "union" in features_step.named_steps else features_step
    _ensure_union_transform(union)

# --- Choix du CSV d'entrée (X + y)
CANDIDATES = [
    REPO_ROOT / "notebooks" / "df.csv",        # ← ajoute ceci
    REPO_ROOT / "data" / "X_valid.csv",
    REPO_ROOT / "data" / "X_train_update.csv",
]
csv_path = next((p for p in CANDIDATES if p.exists()), None)
if csv_path is None:
    print("[ERR] Introuvable : df.csv / X_valid.csv / X_train_update.csv")
    sys.exit(1)

print(f"[INFO] Chargement échantillon depuis: {csv_path}")
DF = pd.read_csv(csv_path)

# Détection de la cible
TARGET_COLS = ["prdtypecode", "target", "label"]
target_col = next((c for c in TARGET_COLS if c in DF.columns), None)

if target_col is None:
    print("[WARN] Pas de colonne cible trouvée; métriques désactivées.")
    X_df = DF; y_series = None
else:
    y_series = DF[target_col].copy()
    X_df = DF.drop(columns=[target_col])

# Échantillon cohérent X/y
n = min(8000, len(X_df))
idx = X_df.sample(n, random_state=42).index
X_valid = X_df.loc[idx].reset_index(drop=True)
y_valid = y_series.loc[idx].reset_index(drop=True) if y_series is not None else None
print(f"[INFO] X_valid shape: {X_valid.shape} | y_valid: {('OK' if y_valid is not None else 'absent')}")

# --- Prédictions + métriques
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
proba = pipe.predict_proba(X_valid)  # colonnes en indices encodés
le = getattr(pipe.named_steps.get("model", None), "le_", None)  # nom de step: "model"
if le is not None:
    import pandas as pd
    proba_df = pd.DataFrame(proba, columns=le.classes_)  # ← vrais prdtypecode en colonnes
    print("Confidence mean:", proba_df.max(axis=1).mean())
else:
    print("[WARN] Impossible de retrouver le mapping; colonnes=indices encodés")
# récupérer les labels si dispo
labels = getattr(pipe.named_steps.get("clf", pipe), "classes_", None)

y_pred = pipe.predict(X_valid)  # ← labels = vrais prdtypecode
acc = accuracy_score(y_valid, y_pred)
f1m = f1_score(y_valid, y_pred, average="macro")
print(f"[METRICS] Acc={acc:.4f} | F1-macro={f1m:.4f}")

if y_valid is not None:
    acc = accuracy_score(y_valid, y_pred)
    f1m = f1_score(y_valid, y_pred, average="macro")
    print(f"[METRICS] Acc={acc:.4f} | F1-macro={f1m:.4f}")

from sklearn.model_selection import train_test_split
X_tmp, _, y_tmp, _ = train_test_split(X_df, y_series, train_size=8000, stratify=y_series, random_state=42)
cm = confusion_matrix(y_tmp, pipe.predict(X_tmp))
print(f"[METRICS] Confusion matrix (sur 8000 échantillons):")
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_valid, y_pred, digits=3))
print(confusion_matrix(y_valid, y_pred))
