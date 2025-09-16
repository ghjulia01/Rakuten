# tools/run_all_b4.py
# Script pour lancer l'entraînement complet B4 (CV + rapports + modèle final)
# Usage: python tools/run_all_b4.py [--config features/config.toml] [--labels features/labels_map.json] [--out-model artifacts/b4.joblib] [--no-compare] [--topN 30]
# (optionnel) --no-compare pour sauter la comparaison LR vs SVC en CV
# (optionnel) --topN pour choisir le nombre de classes dans la matrice de confusion (défaut 30)
# Nécessite que main.train_model fonctionne (dépendances, dataset, etc.)
# --out-model artifacts/b4.joblib pour changer l’emplacement du modèle final.
# (optionnel) s'assurer qu'on n'a pas d'échantillon forcé
# Remove-Item Env:RAKUTEN_MAX_N -ErrorAction SilentlyContinue
# Si on souhaite des échantillons limités (ex: pour tests), définir $env:RAKUTEN_MAX_N=3000
# python tools/run_all_b4.py --config features/config.toml --no-compare --labels features/labels_map.json --out-model artifacts/b4.joblib


import os, sys, subprocess, argparse, shutil, json
from pathlib import Path

def run(cmd: list[str]):
    print("\n" + " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    p = argparse.ArgumentParser(description="Run B4 end-to-end: CV + reports + final model")
    p.add_argument("--config", default="features/config.toml")
    p.add_argument("--labels", default="features/labels_map.json")
    p.add_argument("--out-model", default="artifacts/b4.joblib")
    p.add_argument("--no-compare", action="store_true", help="Skip LR vs SVC compare")
    p.add_argument("--topN", type=int, default=30, help="Top-N classes for confusion plot")
    args = p.parse_args()

    # 0) S'assurer qu'on est sur le dataset complet
    os.environ.pop("RAKUTEN_MAX_N", None)

    repo = Path(__file__).resolve().parents[1]
    cfg  = str((repo / args.config).resolve())
    labels = str((repo / args.labels).resolve())
    out_model = str((repo / args.out_model).resolve())

    # 1) Baseline B4 en CV (produit OOF, rapports, SVD preview…)
    run([sys.executable, "-m", "main.train_model", "--config", cfg, "--baseline", "b4"])

    # 2) (Optionnel) Comparer LR vs SVC en CV
    if not args.no_compare:
        try:
            run([sys.executable, "-m", "main.train_model", "--config", cfg, "--compare"])
        except subprocess.CalledProcessError:
            print("[WARN] --compare a échoué (optionnel), on continue…")
            
    # 3) Rapports & figures
    try:
        preds_csv = str((repo / "results/preds_b4.csv").resolve())
        run([sys.executable, "tools/rapport_complet.py",
            "--preds", preds_csv,
            "--labels-map", labels])
    except subprocess.CalledProcessError:
        print("[WARN] rapport_complet.py a échoué (optionnel), on continue…")

    if Path(preds_csv).exists():
        try:
            run([
                sys.executable, "tools/plot_confusion_from_csv.py",
                "--csv", preds_csv,
                "--labels-map", labels,              # <- flag corrigé
                "--normalize", "true",
                "--topN", str(args.topN),
                "--title", f"Matrice de confusion — B4 (multimodal, top {args.topN})",
                "--output", "results/figures/confusion_b4_topN.png",
                "--export-par-classe", "results/reports/metrics_b4_par_classe.csv",
                "--problems-prefix", "results/reports/b4",
                "--worst-by", "f1",
                "--min-support", "200",
                "--worst-k", "10",
                "--top-mis", "3",
                "--heatmap-problemes", "results/figures/confusion_b4_problemes.png",
            ])
        except subprocess.CalledProcessError:
            print("[WARN] plot_confusion_from_csv.py a échoué (optionnel).")
    else:
        print("[INFO] results/preds_b4.csv introuvable (étape CV a-t-elle bien tourné ?).")

    # Matrice de confusion + exports diagnostics
    if Path(preds_csv).exists():
        try:
            run([
                sys.executable, "tools/plot_confusion_from_csv.py",
                "--csv", preds_csv,
                "--labels-map", labels,
                "--normalize", "true",
                "--topN", str(args.topN),
                "--title", f"Matrice de confusion — B4 (multimodal, top {args.topN})",
                "--output", "results/figures/confusion_b4_topN.png",
                "--export-par-classe", "results/reports/metrics_b4_par_classe.csv",
                "--problems-prefix", "results/reports/b4",
                "--worst-by", "f1",
                "--min-support", "200",
                "--worst-k", "10",
                "--top-mis", "3",
                "--heatmap-problemes", "results/figures/confusion_b4_problemes.png",
            ])
        except subprocess.CalledProcessError:
            print("[WARN] plot_confusion_from_csv.py a échoué (optionnel).")
    else:
        print("[INFO] results/preds_b4.csv introuvable (étape CV a-t-elle bien tourné ?).")

    # 4) Diagnostics ACP/SHAP (optionnel mais utile)
    try:
        run([sys.executable, "tools/diagnostics_acp_shap.py", "--kind", "b4"])
    except subprocess.CalledProcessError:
        print("[WARN] diagnostics_acp_shap.py a échoué (optionnel), on continue…")

    # 5) Entraîner un modèle final sur 100% du train et sauvegarder en .joblib
    run([sys.executable, "tools/train_b4_and_save.py", "--config", cfg, "--out", out_model])

    # 6) Récap rapide
    print("\n=== DONE ===")
    print(f"- Modèle final : {out_model} (fit full train)")
    for p in [
        "results/preds_b4.csv",
        "reports/report_b4_cv.txt",
        "reports/report_b4_cv_readable.txt",
        "results/baseline_results_summary.csv",
        "results/figures/confusion_b4_topN.png",
        "results/reports/metrics_b4_par_classe.csv",
        "results/compare_cv_results.csv",
    ]:
        path = repo / p
        print(f"- {p} {'✅' if path.exists() else '—'}")

if __name__ == "__main__":
    main()