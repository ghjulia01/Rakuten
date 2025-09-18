# tools/smoke_head_topk.py
import sys
import argparse
from models.cnn_features import CNNFeaturizer
# python tools/smoke_head_topk.py --images_dir data/images/images --head_path models/head_ft.pth --rows 3 --csv notebooks/df.csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True, help="Dossier images (Rakuten)")
    ap.add_argument("--head_path", required=True, help="Chemin du fichier tête FT (torch.save)")
    ap.add_argument("--rows", type=int, default=3, help="Nb d'exemples à tester")
    ap.add_argument("--csv", help="CSV avec colonnes imageid,productid (pour _path_from_row)")
    args = ap.parse_args()

    fe = CNNFeaturizer(image_dir=args.images_dir, arch="resnet50", device="auto")
    fe.load_head(args.head_path)

    paths = []
    if args.csv:
        import pandas as pd
        df = pd.read_csv(args.csv)
        for i in range(min(args.rows, len(df))):
            paths.append(fe._path_from_row(df.iloc[i]))
    else:
        print("Sans CSV, fournir des chemins complets à la main dans le code.")
        return

    print("→ Test top-k (k=5) sur", len(paths), "images…")
    topk = fe.topk_from_paths(paths, k=5)
    for i, items in enumerate(topk):
        print(f"[{i}]")
        for idx, lbl, logit, proba in items:
            print(f"  - {idx:5d} | {lbl:20s} | logit={logit:8.4f} | p={proba:7.4f}")

if __name__ == "__main__":
    main()