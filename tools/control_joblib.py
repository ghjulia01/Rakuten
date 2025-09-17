import joblib, pandas as pd
from pathlib import Path
import sys
# python tools/control_joblib.py
# Afficher le nombre de colonnes produites par chaque branche du pipeline "features"

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "main"))

try:
    from train_model import load_config, init_seeds, build_baseline_pipeline
except ModuleNotFoundError:
    from main.train_model import load_config, init_seeds, build_baseline_pipeline

pipe = joblib.load("artifacts/b4.joblib")
feat = pipe.named_steps["features"]

tr_text = dict(pipe.named_steps["features"].transformer_list)["text"]
print(type(tr_text), getattr(tr_text, "named_steps", None))

print("Branches dans 'features':", [n for n, _ in feat.transformer_list])

df = pd.read_csv("notebooks/df.csv")
keep = [c for c in ("designation","description","productid","imageid") if c in df.columns]
df_sm = df[keep].head(5) if keep else df.head(5)

for name, tr in feat.transformer_list:
    try:
        n = tr.transform(df_sm).shape[1]
    except Exception as e:
        n = f"ERR: {e}"
    print(f"{name:>20s} -> {n} colonnes")