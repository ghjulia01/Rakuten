# tools/train_image_only_demo.py
import json, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

NPZ = Path("data/demo_images_embeddings.npz")
INDEX = Path("data/demo_images_index.json")
LABEL_MAP_JSON = Path("data/label_map.json")  # optionnel si tu as une map
OUT_JOBLIB = Path("artifacts/demo_image_only.joblib")

# Ici, on suppose que le label est déduit du nom de dossier parent, ex :
# streamlit_app/demo_images/<label>/image_001.jpg
def derive_label_from_path(p: str) -> str:
    from pathlib import Path
    return Path(p).parent.name

def main():
    X = np.load(NPZ)["X"].astype("float32")  # float32 côté sklearn = plus stable
    paths = json.loads(INDEX.read_text(encoding="utf-8"))["paths"]
    y = pd.Series(paths).map(derive_label_from_path).values

    # Encode labels (str) -> int
    classes, y_int = np.unique(y, return_inverse=True)

    Xtr, Xva, ytr, yva = train_test_split(X, y_int, test_size=0.2, stratify=y_int, random_state=42)

    clf = LogisticRegression(max_iter=2000, n_jobs=1, verbose=0, C=2.0)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xva)
    print("F1-macro (val):", f1_score(yva, pred, average="macro"))

    payload = {
        "clf": clf,
        "classes_": classes,    # pour la restitution des libellés
        "embeddings_shape": X.shape[1],
        "backbone": "resnet50-imagenet",
    }
    OUT_JOBLIB.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, OUT_JOBLIB, compress=3)
    print("Saved ->", OUT_JOBLIB)

if __name__ == "__main__":
    main()