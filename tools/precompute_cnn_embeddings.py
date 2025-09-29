# tools/precompute_cnn_embeddings.py
# python tools/precompute_cnn_embeddings.py
import os, json, numpy as np, torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

IMG_DIR = Path(r"streamlit_app/demo_images")  # adapte si besoin
OUT_NPZ = Path(r"data/demo_images_embeddings.npz")
OUT_JSON = Path(r"data/demo_images_index.json")

# Modèle feature extractor (ResNet50 global avgpool → 2048-dim)
def get_model(device):
    import torchvision.models as models
    import torch.nn as nn
    m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    m.fc = nn.Identity()
    m.eval().to(device)
    return m

def get_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])

def list_images(root):
    ex = {".jpg",".jpeg",".png",".bmp",".webp"}
    return [p for p in Path(root).rglob("*") if p.suffix.lower() in ex]

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(device)
    tfm = get_transform()

    paths = list_images(IMG_DIR)
    feats = []
    idx = []
    for p in tqdm(paths, desc="Embeddings"):
        try:
            img = Image.open(p).convert("RGB")
            x = tfm(img).unsqueeze(0).to(device)
            with torch.no_grad():
                f = model(x).squeeze(0).cpu().numpy().astype(np.float16)
            feats.append(f)
            idx.append(str(p))
        except Exception:
            # on skippe les images illisibles
            continue

    X = np.stack(feats, axis=0)            # (N, 2048) float16
    os.makedirs(OUT_NPZ.parent, exist_ok=True)
    np.savez_compressed(OUT_NPZ, X=X)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"paths": idx}, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {X.shape} -> {OUT_NPZ} and index -> {OUT_JSON}")

if __name__ == "__main__":
    main()