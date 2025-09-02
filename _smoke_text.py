import tomllib, pandas as pd
from features.text_pipeline import create_text_pipeline_from_cfg

# Charger la config
with open("features/config.toml","rb") as f:
    cfg = tomllib.load(f)

need = ["designation","description"]
Xtrain = pd.read_csv(cfg["paths"]["x_train_csv"], index_col=0)
Xmini  = Xtrain[need].head(50)  # petit lot pour tester

pipe = create_text_pipeline_from_cfg(cfg["text"])
Xt = pipe.fit_transform(Xmini)
print("OK, shape =", getattr(Xt, "shape", None))
