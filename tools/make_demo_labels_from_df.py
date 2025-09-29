# tools/make_demo_labels_from_df.py
# python tools/make_demo_labels_from_df.py
import pandas as pd, json
from pathlib import Path

DF_PATH = "notebooks/df.csv"                       # ton df complet
INDEX = "data/demo_images_index.json"         # liste des images démo
OUT = "data/demo_labels.csv"

def fname_from_ids(imgid, prodid):
    return f"image_{int(imgid)}_product_{int(prodid)}.jpg"

def main():
    df = pd.read_csv(DF_PATH)                  # contient imageid/productid/...
    idx = json.loads(Path(INDEX).read_text())["paths"]
    # extraire ids depuis les noms des fichiers démo
    import re
    m = [re.search(r'image_(\d+)_product_(\d+)\.', p) for p in idx]
    pairs = [(int(mm.group(1)), int(mm.group(2))) for mm in m if mm]
    demo = pd.DataFrame(pairs, columns=["imageid","productid"])
    demo["fname"] = demo.apply(lambda r: fname_from_ids(r.imageid, r.productid), axis=1)
    # join
    
    df_demo = df.merge(
        demo[["imageid", "productid", "fname"]],
        on=["imageid", "productid"],
        how="inner",
        validate="m:1",
        suffixes=("", "_demo")
    )

    # Choisir la bonne colonne 'fname' après le merge
    fname_col = "fname"
    if fname_col not in df_demo.columns:
        # si collision ailleurs, Pandas aura mis fname_x / fname_y
        if "fname_y" in df_demo.columns:
            fname_col = "fname_y"
        elif "fname_x" in df_demo.columns:
            fname_col = "fname_x"
        else:
            # dernier recours: recalculer le nom de fichier
            df_demo["fname"] = df_demo.apply(lambda r: fname_from_ids(r.imageid, r.productid), axis=1)
            fname_col = "fname"

    # construit image_rel absolu (pointe vers demo_images)
    base = Path("streamlit_app/demo_images").resolve()
    df_demo["image_rel"] = df_demo[fname_col].apply(lambda f: str(base / f))

    # sauvegarde
    out_csv = "data/demo_df_for_predict.csv"
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df_demo.to_csv(out_csv, index=False)
    print("Écrit", out_csv, "avec", len(df_demo), "lignes")
    print("Colonnes =", list(df_demo.columns))

if __name__ == "__main__":
    main()