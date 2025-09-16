# tools/run_b2_reports.py
# Rapports B2 (confusion + top problèmes + rapport complet)
# python tools/run_b2_reports.py --csv results/preds_b2.csv --labels-map features/
# python tools/run_b2_reports.py


import argparse, subprocess, sys
from pathlib import Path

def run(cmd):
    print("\n" + " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser(description="Rapports B2 (confusion + top problèmes + rapport complet)")
    ap.add_argument("--csv", default="results/preds_b2.csv")
    ap.add_argument("--labels-map", default="features/labels_map.json")
    ap.add_argument("--topN", type=int, default=30)
    ap.add_argument("--worst-k", type=int, default=6)
    ap.add_argument("--min-support", type=int, default=200)
    ap.add_argument("--top-mis", type=int, default=3)
    ap.add_argument("--mini-wrap", type=int, default=18)
    ap.add_argument("--mini-fontsize", type=int, default=8)
    args = ap.parse_args()

    if not Path(args.csv).exists():
        sys.exit(f"CSV introuvable: {args.csv} — Lance d'abord: python -m main.train_model --config features/config.toml --baseline b2")
    if not Path(args.labels_map).exists():
        sys.exit(f"labels_map introuvable: {args.labels_map}")

    # 1) Grande matrice + exports
    run([sys.executable, "tools/plot_confusion_from_csv.py",
         "--csv", args.csv,
         "--labels-map", args.labels_map,
         "--normalize", "true",
         "--topN", str(args.topN),
         "--title", f"Matrice de confusion — B2 (texte, top {args.topN})",
         "--output", "results/figures/confusion_b2_topN.png",
         "--export-par-classe", "results/reports/metrics_b2_par_classe.csv",
         "--problems-prefix", "results/reports/b2",
         "--worst-by", "f1",
         "--min-support", str(args.min_support),
         "--worst-k", str(args.worst_k),
         "--top-mis", str(args.top_mis),
         "--heatmap-problemes", "results/figures/confusion_b2_problemes.png",
         "--mini-wrap", str(args.mini_wrap),
         "--mini-fontsize", str(args.mini_fontsize),
    ])

    # 2) Rapport complet
    run([sys.executable, "tools/rapport_complet.py",
         "--preds", args.csv,
         "--labels-map", args.labels_map,
         "--out-md", "results/reports/rapport_b2.md",
         "--out-html", "results/reports/rapport_b2.html",
         "--topK", str(args.topN),
    ])

    print("\n Terminé. Voir results/figures/ et results/reports/")

if __name__ == "__main__":
    main()