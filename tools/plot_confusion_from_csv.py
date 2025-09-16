# tools/plot_confusion_from_csv.py
# -*- coding: utf-8 -*-
"""
Tracer une matrice de confusion à partir d’un CSV de prédictions (y_true, y_pred)
+ Extensions :
  - tableau par classe (précision, rappel, F1, support)
  - rapport des classes problématiques (plus bas F1/Rappel/Précision au-delà d’un seuil de support)
  - top confusions par classe problématique (comptes + taux)
  - mini-heatmap optionnelle focalisée sur les classes problématiques
  - [NOUVEAU] CSV consolidé des problèmes (avec colonne 'top_confusions')
  - [NOUVEAU] Mini-heatmap annotée avec support & F1 dans les labels
  
Le Scirpt est utilisable en ligne de commande.
 python tools/plot_confusion_from_csv.py `
  --csv results/preds_b4.csv `
  --labels-map features/labels_map.json `
  --normalize true `
  --topN 30 `
  --worst-by f1 --min-support 200 --worst-k 6 --top-mis 3 `
  --heatmap-problemes results/figures/confusion_b4_problemes.png `
  --mini-wrap 18 --mini-fontsize 8

"""

import argparse
import json
from pathlib import Path
from textwrap import fill

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

# ---------------------------- fonctions utilitaires ----------------------------
def charger_mapping_labels(path):
    """Charger un fichier de correspondance id -> nom (json ou csv)."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Mapping labels introuvable: {p}")
    if p.suffix.lower() == ".json":
        with open(p, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        return {str(k): v for k, v in mapping.items()}
    elif p.suffix.lower() in {".csv", ".tsv"}:
        df = pd.read_csv(p)
        if not {"id", "name"}.issubset(df.columns):
            raise ValueError("Le CSV de mapping doit avoir les colonnes: id,name")
        return {str(r["id"]): r["name"] for _, r in df.iterrows()}
    else:
        raise ValueError("Format non supporté (utiliser .json ou .csv)")

def trouver_colonnes_proba(df):
    """Détecter les colonnes de probabilités (proba_* ou p_*)."""
    return [c for c in df.columns if c.startswith("proba_") or c.startswith("p_")]

def topN_classes_par_support(y_true, y_pred, N):
    """Sélectionner les N classes avec le plus de support (y_true)."""
    vals, counts = np.unique(y_true, return_counts=True)
    keep = set(vals[np.argsort(-counts)[:N]].tolist())
    keep |= set(np.unique(y_pred).tolist())
    return sorted(list(keep), key=lambda x: str(x))

def normalisation_option(opt):
    if opt is None or str(opt).lower() == "none":
        return None
    opt = str(opt).lower()
    if opt in {"true", "pred", "all"}:
        return opt
    raise ValueError("--normalize doit être: none, true, pred, all")

def taille_fig_selon_nlabels(n):
    base = max(6, min(24, int(n * 0.5)))
    return (base, base)

def _format_top_confusions(rows):
    """
    rows: liste de dicts [{classe_pred_nom, taux, comptes}, ...]
    Retourne une chaîne lisible: "Nom1: 22.3% (n=135); Nom2: 11.2% (n=68)"
    """
    if not rows:
        return ""
    parts = []
    for r in rows:
        parts.append(f"{r['classe_pred_nom']}: {r['taux']*100:.1f}% (n={r['comptes']})")
    return "; ".join(parts)

# ---------------------------- programme principal ------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Tracer une matrice de confusion depuis un CSV contenant y_true,y_pred et produire un diagnostic des classes problématiques."
    )
    ap.add_argument("--csv", required=True, help="CSV avec au minimum: y_true,y_pred")
    ap.add_argument("--labels-map", default=None,
                    help="Mapping optionnel (.json {id:nom} ou .csv id,name)")
    ap.add_argument("--normalize", default="none",
                    help="none|true|pred|all (sémantique sklearn)")
    ap.add_argument("--topN", type=int, default=None,
                    help="Limiter le tracé aux N classes avec le plus de support")
    ap.add_argument("--include-classes", default=None,
                    help="Fichier optionnel listant les classes à inclure (1 par ligne)")
    ap.add_argument("--title", default=None, help="Titre de la figure")
    ap.add_argument("--output", default="results/figures/confusion.png",
                    help="Fichier de sortie PNG")
    ap.add_argument("--dpi", type=int, default=180)
    ap.add_argument("--figsize", default=None, help='ex: "14x12"')
    ap.add_argument("--topk", type=int, default=0,
                    help="Si >0, calcule l’accuracy Top-k à partir des colonnes proba_*")

    # Exports/diagnostics
    ap.add_argument("--export-par-classe", default="results/reports/metrics_par_classe.csv",
                    help="CSV des métriques par classe (précision/rappel/F1/support)")
    ap.add_argument("--problems-prefix", default="results/reports/problemes",
                    help="Préfixe pour les CSV des classes problématiques")
    ap.add_argument("--worst-by", choices=["f1", "recall", "precision"], default="f1",
                    help="Métrique utilisée pour classer les pires classes")
    ap.add_argument("--min-support", type=int, default=100,
                    help="Ignorer les classes avec un support inférieur")
    ap.add_argument("--worst-k", type=int, default=10,
                    help="Nombre de classes à lister comme problématiques")
    ap.add_argument("--top-mis", type=int, default=3,
                    help="Nombre de confusions principales à extraire par classe problématique")
    ap.add_argument("--heatmap-problemes", default="results/figures/confusion_problemes.png",
                    help="Mini-heatmap focalisée sur les classes problématiques (mettre '' pour désactiver)")
    ap.add_argument("--mini-wrap", type=int, default=22,
                    help="Largeur d'enrobage des étiquettes de la mini-heatmap")
    ap.add_argument("--mini-fontsize", type=int, default=8,
                    help="Taille de police des ticks de la mini-heatmap")
    args = ap.parse_args()

    out_png = args.output
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)

    # ---- lecture du CSV
    df = pd.read_csv(args.csv)
    if not {"y_true", "y_pred"}.issubset(df.columns):
        raise ValueError("Le CSV doit contenir les colonnes: y_true, y_pred")

    y_true = df["y_true"].astype(str).to_numpy()
    y_pred = df["y_pred"].astype(str).to_numpy()

    # ---- option Top-k
    if args.topk and args.topk > 0:
        proba_cols = trouver_colonnes_proba(df)
        if proba_cols:
            class_ids = [c.split("_", 1)[1] for c in proba_cols]
            probs = df[proba_cols].to_numpy()
            topk_idx = np.argsort(-probs, axis=1)[:, : args.topk]
            topk_classes = np.array(class_ids)[topk_idx]
            topk_hit = np.array([yt in set(topk_classes[i]) for i, yt in enumerate(y_true)])
            print(f"[INFO] Top-{args.topk} accuracy (global): {topk_hit.mean():.4f}")
        else:
            print("[WARN] --topk demandé mais pas de colonnes de probabilités trouvées.")

    # ---- limiter aux classes choisies
    keep_classes = None
    if args.include_classes:
        incl = pd.read_csv(args.include_classes, header=None).astype(str)[0].tolist()
        keep_classes = set(incl)
    elif args.topN:
        keep_classes = set(topN_classes_par_support(y_true, y_pred, args.topN))

    if keep_classes is not None:
        mask = np.array([t in keep_classes for t in y_true])
        y_true, y_pred = y_true[mask], y_pred[mask]

    labels = sorted(list(set(np.unique(y_true)) | set(np.unique(y_pred))), key=lambda x: str(x))

    # mapping lisible
    mapping = charger_mapping_labels(args.labels_map)
    display_labels = [mapping.get(l, l) for l in labels] if mapping else labels

    # ---- matrices de confusion
    cm_counts = confusion_matrix(y_true, y_pred, labels=labels, normalize=None)
    cm_true = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    # ---- figure principale
    if args.figsize:
        try:
            w, h = args.figsize.lower().split("x")
            figsize = (float(w), float(h))
        except Exception:
            raise ValueError('--figsize doit être comme "14x12"')
    else:
        figsize = taille_fig_selon_nlabels(len(labels))

    fig, ax = plt.subplots(figsize=figsize)
    norm = normalisation_option(args.normalize)
    cm_disp = confusion_matrix(y_true, y_pred, labels=labels, normalize=norm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_disp, display_labels=display_labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format=".2f" if norm else "d")
    ax.set_title(args.title or f"Matrice de confusion ({'normalisée: '+norm if norm else 'comptes'})")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    print(f"[INFO] Figure sauvegardée → {out_png}")

    # ---- métriques par classe
    rapport = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    lignes = []
    for lab in labels:
        r = rapport.get(lab, {})
        lignes.append({
            "classe_id": lab,
            "classe_nom": mapping.get(lab, lab) if mapping else lab,
            "precision": r.get("precision", 0.0),
            "rappel":    r.get("recall", 0.0),
            "f1":        r.get("f1-score", 0.0),
            "support":   int(r.get("support", 0)),
        })
    df_classes = pd.DataFrame(lignes).sort_values("support", ascending=False)
    Path(args.export_par_classe).parent.mkdir(parents=True, exist_ok=True)
    df_classes.to_csv(args.export_par_classe, index=False)
    print(f"[INFO] Métriques par classe → {args.export_par_classe}")

    # ---- classes problématiques (selon métrique choisie)
    col_metrique = {"f1": "f1", "recall": "rappel", "precision": "precision"}[args.worst_by]
    df_pb = df_classes[df_classes["support"] >= int(args.min_support)].copy()
    if df_pb.empty:
        print("[WARN] Pas de classes au-dessus du min-support.")
        return

    df_pb = df_pb.sort_values([col_metrique, "support"], ascending=[True, False]).head(args.worst_k)
    probs_csv = Path(args.problems_prefix + "_problemes.csv")
    probs_csv.parent.mkdir(parents=True, exist_ok=True)
    df_pb.to_csv(probs_csv, index=False)
    print(f"[INFO] Classes problématiques → {probs_csv}")

    # ---- confusions principales par classe problématique
    label_index = {lab: i for i, lab in enumerate(labels)}
    recs = []
    for _, r in df_pb.iterrows():
        lab = r["classe_id"]
        i = label_index.get(lab, None)
        if i is None:
            continue
        row_true = cm_true[i].copy()    # taux (normalisé par vrai)
        row_counts = cm_counts[i].copy()  # comptes bruts
        row_true[i] = 0.0
        row_counts[i] = 0.0
        order = np.argsort(-row_true)
        keep = [j for j in order if row_true[j] > 0][: args.top_mis]
        for j in keep:
            pred_id = labels[j]
            recs.append({
                "classe_vraie_id": lab,
                "classe_vraie_nom": mapping.get(lab, lab) if mapping else lab,
                "classe_pred_id": pred_id,
                "classe_pred_nom": mapping.get(pred_id, pred_id) if mapping else pred_id,
                "taux": float(row_true[j]),
                "comptes": int(row_counts[j]),
            })

    conf_csv = Path(args.problems_prefix + "_top_confusions.csv")
    pd.DataFrame(recs).to_csv(conf_csv, index=False)
    print(f"[INFO] Confusions principales sauvegardées → {conf_csv}")

    # ---- CSV consolidé problèmes + résumé textuel des confusions
    if len(recs) > 0:
        df_recs = pd.DataFrame(recs)
        gb = df_recs.groupby(["classe_vraie_id", "classe_vraie_nom"])
        try:
            # pandas ≥ 2.2 : on exclut explicitement les colonnes de groupage
            grouped = gb.apply(
                lambda g: pd.Series({
                    "top_confusions": _format_top_confusions(
                        g[["classe_pred_nom", "taux", "comptes"]].to_dict("records")
                    )
                }),
                include_groups=False
            ).reset_index()
        except TypeError:
            # pandas < 2.2 : pas de param include_groups
            grouped = gb.apply(
                lambda g: pd.Series({
                    "top_confusions": _format_top_confusions(
                        g[["classe_pred_nom", "taux", "comptes"]].to_dict("records")
                    )
                })
            ).reset_index()
    else:
        grouped = pd.DataFrame(columns=["classe_vraie_id", "classe_vraie_nom", "top_confusions"])

    df_diag = df_pb.merge(grouped,
                          left_on=["classe_id", "classe_nom"],
                          right_on=["classe_vraie_id", "classe_vraie_nom"],
                          how="left").drop(columns=["classe_vraie_id", "classe_vraie_nom"])
    diag_csv = Path(args.problems_prefix + "_diagnostics.csv")
    df_diag.to_csv(diag_csv, index=False)
    print(f"[INFO] Diagnostic consolidé → {diag_csv}")

    # ---- mini-heatmap focalisée sur pires classes, avec labels enrichis
    if args.heatmap_problemes:
        sel = df_pb["classe_id"].tolist()
        idx = [label_index[s] for s in sel if s in label_index]
        if idx:
            cm_sub = cm_true[np.ix_(idx, idx)]
            # Labels enrichis: "Nom (n=SUPPORT, F1=xx.xx)"
            meta = df_pb.set_index("classe_id")[["classe_nom", "support", "f1"]].to_dict(orient="index")
            lab_sub = []
            for lab in sel:
                if lab in meta:
                    it = meta[lab]
                    lab_sub.append(f"{it['classe_nom']} (n={it['support']}, F1={it['f1']:.2f})")
                else:
                    lab_sub.append(mapping.get(lab, lab) if mapping else lab)

            # NEW: wrap + taille dynamique + petite police
            lab_sub_wrapped = [fill(s, width=args.mini_wrap) for s in lab_sub]
            max_len = max(len(s) for s in lab_sub_wrapped)
            h = max((5), 0.6 * len(idx))
            w = max(8, 0.18 * max_len)

            fig2, ax2 = plt.subplots(figsize=(w, h))
            disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_sub, display_labels=lab_sub_wrapped)
            disp2.plot(ax=ax2, cmap="Oranges", colorbar=True, values_format=".2f")
            for t in ax2.get_xticklabels():
                t.set_rotation(90)
                t.set_fontsize(args.mini_fontsize)
            for t in ax2.get_yticklabels():
                t.set_fontsize(args.mini_fontsize)
            ax2.set_title(f"Matrice de confusion (normalisée par vrai) — pires classes ({col_metrique})")
            plt.tight_layout()
            Path(args.heatmap_problemes).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(args.heatmap_problemes, dpi=args.dpi, bbox_inches="tight")
            print(f"[INFO] Mini-heatmap sauvegardée → {args.heatmap_problemes}")

if __name__ == "__main__":
    main()