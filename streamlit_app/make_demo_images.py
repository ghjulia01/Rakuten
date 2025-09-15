# streamlit_app/make_demo_images.py
import os, shutil, argparse
from pathlib import Path
import pandas as pd

def infer_image_name(row, img_col):
    # 1) si la colonne image existe
    if img_col and img_col in row and pd.notna(row[img_col]):
        return str(row[img_col])
    # 2) fallback: construit le nom à partir de imageid/productid
    if {"imageid", "productid"} <= set(row.index):
        return f"image_{int(row['imageid'])}_product_{int(row['productid'])}.jpg"
    return None

def main():
    ap = argparse.ArgumentParser(description="Créer un mini jeu d'images de démo pour Streamlit.")
    ap.add_argument("--csv", help="CSV source (par défaut: notebooks/df.csv)")
    ap.add_argument("--src", help="Dossier source des images (défaut: data/images/images/image_train)")
    ap.add_argument("--out", help="Dossier de sortie (défaut: streamlit_app/demo_images)")
    ap.add_argument("--img-col", default="image_name", help="Colonne image (def: image_name, fallback image_path)")
    ap.add_argument("--label-col", default="prdtypecode", help="Colonne label (def: prdtypecode)")
    ap.add_argument("--n-per-class", type=int, default=10, help="Nb d'images par classe")
    args = ap.parse_args()

    APP_DIR  = Path(__file__).resolve().parent
    REPO_DIR = APP_DIR.parent

    csv_in  = Path(args.csv) if args.csv else (REPO_DIR / "notebooks" / "df.csv")
    src_dir = Path(args.src) if args.src else (REPO_DIR / "data" / "images" / "images" / "image_train")
    out_dir = Path(args.out) if args.out else (APP_DIR / "demo_images")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_in.exists():
        raise FileNotFoundError(f"CSV introuvable: {csv_in}")
    if not src_dir.exists():
        raise FileNotFoundError(f"Dossier images introuvable: {src_dir}")

    df = pd.read_csv(csv_in)
    label_col = args.label_col
    if label_col not in df.columns:
        raise ValueError(f"Colonne label '{label_col}' introuvable. Colonnes: {list(df.columns)}")

    img_col = args.img_col if args.img_col in df.columns else ("image_path" if "image_path" in df.columns else None)

    rows = []
    for cat, grp in df.groupby(label_col):
        sample = grp.sample(min(args.n_per_class, len(grp)), random_state=42)
        for _, r in sample.iterrows():
            name = infer_image_name(r, img_col)
            if not name:
                continue
            cand = Path(str(name))
            # absolu → direct ; sinon on tente dans src_dir (par nom de fichier, puis sous-chemin)
            if cand.is_absolute() and cand.exists():
                src = cand
            elif (src_dir / cand.name).exists():
                src = src_dir / cand.name
            elif (src_dir / cand).exists():
                src = src_dir / cand
            else:
                continue

            dest = out_dir / src.name
            try:
                shutil.copy2(src, dest)
                rows.append({label_col: cat, "image_rel": dest.name})
            except Exception:
                pass

    out_csv = out_dir / "demo_images.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"✔ Copié {len(rows)} images dans {out_dir}")
    print(f"✔ CSV: {out_csv}")
    print("Dans Streamlit : colonne image = 'image_rel' et Base des images = 'streamlit_app/demo_images'.")

if __name__ == "__main__":
    main()