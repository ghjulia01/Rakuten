# tools/compare_b2_b4.py
# -*- coding: utf-8 -*-
"""
Comparer B2 (texte) vs B4 (multimodal) par classe :
- métriques par classe pour B2 et B4 (precision, recall, f1, support)
- deltas (B4 - B2) par classe
- bascules d’erreurs au niveau échantillon (B2 correct / B4 faux, et inverse)
- figures top gains / top pertes (delta F1)

NOUVEAU :
- --key-column pour aligner les deux CSV sur une clé (ex: productid)
- auto-détection de clé parmi: row_id, id, productid, product_id, index
- alignement par intersection (inner join). Si aucune clé utilisable :
  - par défaut, on aligne par position (min longueur) avec avertissement
  - option --strict pour lever une erreur si tailles différentes

Exemple :
  python -m tools.compare_b2_b4 ^
    --b2 results/preds_b2.csv ^
    --b4 results/preds_b4.csv ^
    --labels-map tools/labels_map.json ^
    --out-prefix results/reports/b2_vs_b4 ^
    --topK 15 ^
    --key-column productid
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score


# -------------------- utilitaires --------------------
def charger_mapping_labels(path):
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Mapping labels introuvable: {p}")
    if p.suffix.lower() == ".json":
        with open(p, "r", encoding="utf-8") as f:
            m = json.load(f)
        return {str(k): v for k, v in m.items()}
    elif p.suffix.lower() in {".csv", ".tsv"}:
        df = pd.read_csv(p)
        if not {"id", "name"}.issubset(df.columns):
            raise ValueError("Le CSV de mapping doit avoir colonnes: id,name")
        return {str(r["id"]): r["name"] for _, r in df.iterrows()}
    else:
        raise ValueError("Format mapping non supporté (json/csv)")


def metr_par_classe(y_true, y_pred, labels=None):
    rep = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    rows = []
    # Si labels est fourni, on respecte l'ordre + inclut toutes les classes
    keys = labels if labels is not None else [k for k in rep.keys() if k not in {"accuracy","macro avg","weighted avg"}]
    for lab in keys:
        r = rep.get(lab, {})
        rows.append(
            dict(
                classe_id=str(lab),
                precision=float(r.get("precision", 0.0)),
                recall=float(r.get("recall", 0.0)),
                f1=float(r.get("f1-score", 0.0)),
                support=int(r.get("support", 0)),
            )
        )
    return pd.DataFrame(rows)


def add_noms_readables(df, mapping):
    if not mapping:
        df["classe_nom"] = df["classe_id"]
    else:
        df["classe_nom"] = df["classe_id"].map(lambda x: mapping.get(str(x), str(x)))
    return df


def bar_top_delta(df_delta, col_delta, topK, out_png, titre):
    d = df_delta.copy()
    d[col_delta] = d[col_delta].fillna(0.0)
    d = d.sort_values(col_delta, ascending=False)

    top_pos = d.head(topK)      # top gains
    top_neg = d.tail(topK)      # top pertes

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    # Gains
    axes[0].barh(top_pos["classe_nom"][::-1], top_pos[col_delta][::-1])
    axes[0].set_title(f"Top gains {col_delta} (B4 - B2)")
    axes[0].set_xlabel(col_delta)
    # Pertes
    axes[1].barh(top_neg["classe_nom"], top_neg[col_delta])
    axes[1].set_title(f"Top pertes {col_delta} (B4 - B2)")
    axes[1].set_xlabel(col_delta)

    fig.suptitle(titre)
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Figure → {out_png}")


def choisir_cle(df, prefer=None):
    """Retourne une clé présente dans df (ordre de préférence)."""
    candidats = []
    if prefer and prefer in df.columns:
        return prefer
    # ordre par défaut
    for c in ["row_id", "id", "productid", "product_id", "index"]:
        if c in df.columns:
            candidats.append(c)
    return candidats[0] if candidats else None


def aligner_predictions(b2, b4, key=None, strict=False):
    """
    Aligne B2 et B4 pour comparaison.
    - Si 'key' (ou clé auto) existe dans les deux → inner join
    - Sinon :
        * strict=True : lève une erreur si tailles différentes
        * strict=False : aligne par position sur la longueur min, avec avertissement
    Retourne: y_true, yb2, yb4 (np.array de str)
    """
    # Tenter une clé explcite, sinon auto
    k2 = key or choisir_cle(b2)
    k4 = key or choisir_cle(b4)

    if k2 and k4 and (k2 in b2.columns) and (k4 in b4.columns):
        # Renommer la colonne clé de B4 si différent pour fusion simple
        if k2 != k4:
            b4 = b4.rename(columns={k4: k2})
            k4 = k2

        # On garde minimal: clé + y_true + y_pred
        need = [k2, "y_true", "y_pred"]
        for col in need:
            if col not in b2.columns:
                raise ValueError(f"B2: colonne manquante '{col}'")
            if col not in b4.columns:
                raise ValueError(f"B4: colonne manquante '{col}'")

        b2_s = b2[need].copy()
        b4_s = b4[need].copy()
        # Cast str pour sécurité
        b2_s[k2] = b2_s[k2].astype(str)
        b4_s[k2] = b4_s[k2].astype(str)
        b2_s["y_true"] = b2_s["y_true"].astype(str)
        b2_s["y_pred"] = b2_s["y_pred"].astype(str)
        b4_s["y_true"] = b4_s["y_true"].astype(str)
        b4_s["y_pred"] = b4_s["y_pred"].astype(str)

        merged = b2_s.merge(b4_s, on=[k2, "y_true"], how="inner", suffixes=("_b2", "_b4"))
        if merged.empty:
            raise ValueError("Après alignement par clé, l’intersection est vide.")
        if len(merged) < min(len(b2), len(b4)):
            print(f"[WARN] Alignement: intersection={len(merged)} (B2={len(b2)}, B4={len(b4)}). "
                  f"Les lignes manquantes sont exclues de la comparaison.")

        y_true = merged["y_true"].to_numpy()
        yb2 = merged["y_pred_b2"].to_numpy()
        yb4 = merged["y_pred_b4"].to_numpy()
        return y_true.astype(str), yb2.astype(str), yb4.astype(str)

    # Pas de clé exploitable
    if len(b2) != len(b4):
        if strict:
            raise ValueError(
                "B2 et B4 n'ont pas le même nombre de lignes et aucune clé d’alignement n'a été trouvée. "
                "Spécifie --key-column ou fournis une clé (row_id/id/productid/…)."
            )
        m = min(len(b2), len(b4))
        print(f"[WARN] Aucune clé trouvée. Alignement par position sur la longueur minimale m={m}. "
              f"(B2={len(b2)}, B4={len(b4)})")
        b2 = b2.iloc[:m].copy()
        b4 = b4.iloc[:m].copy()

    y_true = b2["y_true"].astype(str).to_numpy()
    yb2 = b2["y_pred"].astype(str).to_numpy()
    yb4 = b4["y_pred"].astype(str).to_numpy()
    return y_true, yb2, yb4


# -------------------- programme principal --------------------
def main():
    ap = argparse.ArgumentParser(description="Comparaison B2 vs B4 par classe")
    ap.add_argument("--b2", required=True, help="CSV prédictions B2 (y_true,y_pred + clé optionnelle)")
    ap.add_argument("--b4", required=True, help="CSV prédictions B4 (y_true,y_pred + clé optionnelle)")
    ap.add_argument("--labels-map", default=None, help="JSON/CSV id->nom lisible")
    ap.add_argument("--out-prefix", default="results/reports/b2_vs_b4",
                    help="Préfixe des sorties (CSV/PNG)")
    ap.add_argument("--topK", type=int, default=15, help="Top K pour figures gains/pertes")
    ap.add_argument("--key-column", default=None,
                    help="Nom de la clé commune pour aligner (ex: productid). Si non fourni: auto-détection.")
    ap.add_argument("--strict", action="store_true",
                    help="Si tailles différentes et pas de clé: lever une erreur (sinon, aligne par position).")
    args = ap.parse_args()

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    mapping = charger_mapping_labels(args.labels_map)

    # --- lecture CSV
    b2 = pd.read_csv(args.b2)
    b4 = pd.read_csv(args.b4)

    for col in ["y_true", "y_pred"]:
        if col not in b2.columns:
            raise ValueError(f"B2: colonne manquante '{col}'")
        if col not in b4.columns:
            raise ValueError(f"B4: colonne manquante '{col}'")

    # --- alignement robuste
    y_true, yb2, yb4 = aligner_predictions(b2, b4, key=args.key_column, strict=args.strict)

    # --- étiquettes
    labels = sorted(list(set(np.unique(y_true)) | set(np.unique(yb2)) | set(np.unique(yb4))), key=lambda x: str(x))

    # --- métriques globales
    def f1s(yhat):
        return dict(
            f1_macro=f1_score(y_true, yhat, average="macro"),
            f1_weighted=f1_score(y_true, yhat, average="weighted"),
        )
    g_b2 = f1s(yb2)
    g_b4 = f1s(yb4)
    glob = pd.DataFrame([dict(modele="B2", **g_b2), dict(modele="B4", **g_b4)])
    glob.to_csv(out_prefix.with_name(out_prefix.stem + "_global.csv"), index=False)
    print(f"[INFO] Global → {out_prefix.stem}_global.csv\n{glob}")

    # --- par classe
    m_b2 = metr_par_classe(y_true, yb2, labels=labels)
    m_b4 = metr_par_classe(y_true, yb4, labels=labels)
    m_b2 = add_noms_readables(m_b2, mapping)
    m_b4 = add_noms_readables(m_b4, mapping)
    m_b2.rename(columns={"precision":"precision_b2","recall":"recall_b2","f1":"f1_b2","support":"support_b2"}, inplace=True)
    m_b4.rename(columns={"precision":"precision_b4","recall":"recall_b4","f1":"f1_b4","support":"support_b4"}, inplace=True)

    merged = m_b2.merge(m_b4[["classe_id","precision_b4","recall_b4","f1_b4","support_b4"]], on="classe_id", how="outer")
    merged["classe_nom"] = merged["classe_nom"].fillna(merged["classe_id"])
    merged["support"] = merged[["support_b2","support_b4"]].max(axis=1).fillna(0).astype(int)

    # deltas
    merged["d_precision"] = merged["precision_b4"] - merged["precision_b2"]
    merged["d_recall"]    = merged["recall_b4"]    - merged["recall_b2"]
    merged["d_f1"]        = merged["f1_b4"]        - merged["f1_b2"]

    merged.sort_values(["d_f1","support"], ascending=[False, False], inplace=True)
    merged.to_csv(out_prefix.with_name(out_prefix.stem + "_per_class.csv"), index=False)
    print(f"[INFO] Par classe → {out_prefix.stem}_per_class.csv")

    # --- bascules d’erreurs au niveau échantillon
    b2_ok = (yb2 == y_true)
    b4_ok = (yb4 == y_true)
    both_ok   = int((b2_ok & b4_ok).sum())
    b2_only   = int((b2_ok & ~b4_ok).sum())
    b4_only   = int((~b2_ok & b4_ok).sum())
    both_bad  = int((~b2_ok & ~b4_ok).sum())

    flips = pd.DataFrame([
        {"type":"B2_correct_B4_wrong", "n": b2_only},
        {"type":"B4_correct_B2_wrong", "n": b4_only},
        {"type":"both_correct", "n": both_ok},
        {"type":"both_wrong", "n": both_bad},
    ])
    flips["pct"] = flips["n"] / len(y_true)
    flips.to_csv(out_prefix.with_name(out_prefix.stem + "_flips_global.csv"), index=False)
    print(f"[INFO] Bascules globales → {out_prefix.stem}_flips_global.csv\n{flips}")

    # par classe (selon y_true)
    df_all = pd.DataFrame({"y_true": y_true, "b2_ok": b2_ok, "b4_ok": b4_ok})
    def agg(g):
        n = len(g)
        return pd.Series({
            "n": n,
            "B2_correct_B4_wrong": int((g["b2_ok"] & ~g["b4_ok"]).sum()),
            "B4_correct_B2_wrong": int((~g["b2_ok"] & g["b4_ok"]).sum()),
            "both_correct":        int((g["b2_ok"] & g["b4_ok"]).sum()),
            "both_wrong":          int((~g["b2_ok"] & ~g["b4_ok"]).sum()),
        })
    flips_cls = df_all.groupby("y_true", as_index=False).apply(agg).rename(columns={"y_true":"classe_id"})
    flips_cls["classe_nom"] = flips_cls["classe_id"].map(lambda x: mapping.get(str(x), str(x)))
    flips_cls.sort_values("n", ascending=False, inplace=True)
    flips_cls.to_csv(out_prefix.with_name(out_prefix.stem + "_flips_per_class.csv"), index=False)
    print(f"[INFO] Bascules par classe → {out_prefix.stem}_flips_per_class.csv")

    # --- Heuristique "image aide / dégrade"
    thr = 0.02
    merged["hypothese_image"] = np.where(
        merged["d_f1"] >  thr, "image_aide",
        np.where(merged["d_f1"] < -thr, "image_degrade", "neutre")
    )
    merged.to_csv(out_prefix.with_name(out_prefix.stem + "_per_class.csv"), index=False)  # réécrit avec la colonne

    # --- Figures
    fig_gains = out_prefix.with_name(out_prefix.stem + "_top_gains_losses_f1.png")
    bar_top_delta(
        merged[["classe_nom","d_f1"]],
        "d_f1", args.topK, fig_gains,
        "Comparaison B4 - B2 : Top gains/pertes (F1)"
    )

    print("[OK] Comparaison B2 vs B4 terminée.")

if __name__ == "__main__":
    main()