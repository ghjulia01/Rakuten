# tests/test_cnn_features_head.py
# Test unitaire du module cnn_features.py (tête entraînée + logits/top-k)
# python -m pytest -v tests/test_cnn_features.py

import os
import io
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import pytest

from models.cnn_features import CNNFeaturizer

class _TinyBackbone(nn.Module):
    """
    Backbone jouet: retourne un embedding [B, feat_dim] = somme des pixels (canal) projetée.
    Évite torchvision et tout téléchargement de poids pour rendre le test léger/rapide.
    """
    def __init__(self, feat_dim=8):
        super().__init__()
        self.feat_dim = feat_dim
        self.proj = nn.Linear(3, feat_dim, bias=False)  # 3 canaux -> feat_dim
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(feat_dim, 3, dtype=torch.float32))

    def forward(self, x):  # x: [B,3,H,W], valeurs [0,1]
        # Moyenne glob. par canal -> [B,3], puis projection -> [B,feat_dim]
        b, c, h, w = x.shape
        pooled = x.mean(dim=(2,3))        # [B,3]
        out = self.proj(pooled)           # [B,feat_dim]
        return out

@pytest.fixture
def dummy_images(tmp_path):
    # crée 2 images RGB 32x32
    paths = []
    for i, col in enumerate([(255,0,0),(0,255,0)]):
        p = tmp_path / f"img_{i}.png"
        img = Image.new("RGB", (32,32), color=col)
        img.save(p)
        paths.append(str(p))
    missing = str(tmp_path / "missing.png")
    return paths, missing

def _fake_preprocess(pil_img):
    # ToTensor-like, valeurs [0,1]
    import numpy as np
    arr = np.asarray(pil_img).astype("float32") / 255.0  # H W C
    t = torch.from_numpy(arr).permute(2,0,1)             # C H W
    return t

def test_attach_and_logits_topk(tmp_path, dummy_images, monkeypatch):
    paths, missing = dummy_images

    # 1) Instancie le featurizer sans poids externes
    fe = CNNFeaturizer(image_dir=str(tmp_path), arch="resnet50", device="cpu")
    fe._feat_dim = 8  # on fixe manuellement
    # Monkeypatch _build_model pour renvoyer notre backbone et preprocess factice
    def _fake_build_model():
        return _TinyBackbone(feat_dim=fe._feat_dim), _fake_preprocess
    monkeypatch.setattr(fe, "_build_model", _fake_build_model)
    fe._lazy_load()  # initialise _model et _preprocess

    # 2) Attache une tête entraînée (jouet) + classes
    classes = ["A","B","C"]
    head = nn.Linear(fe._feat_dim, len(classes))
    with torch.no_grad():
        head.weight.fill_(0.1); head.bias.zero_()
    fe.attach_head(head=head, classes=classes, normalize_feat=True)

    # 3) Logits sur 2 images existantes + 1 manquante
    logits = fe.predict_logits_from_paths(paths + [missing])
    assert logits.shape == (3, len(classes))
    assert np.isnan(logits[-1]).all()  # ligne NaN pour l'image manquante
    assert not np.isnan(logits[0]).any()

    # 4) Probas + top-k
    proba = fe.predict_proba_from_paths(paths + [missing])
    assert proba.shape == logits.shape
    assert 0.99 <= proba[0].sum() <= 1.01  # ~1 (tolérance flottante)
    topk = fe.topk_from_paths(paths, k=2)
    assert len(topk) == 2
    assert len(topk[0]) == 2
    idx0, lbl0, logit0, p0 = topk[0][0]
    assert lbl0 in classes
    assert 0 <= p0 <= 1

def test_save_and_load_head(tmp_path, monkeypatch):
    # 1) Featurizer + backbone factice
    fe = CNNFeaturizer(image_dir=str(tmp_path), arch="resnet50", device="cpu")
    fe._feat_dim = 6
    def _fake_build_model():
        return _TinyBackbone(feat_dim=fe._feat_dim), _fake_preprocess
    monkeypatch.setattr(fe, "_build_model", _fake_build_model)
    fe._lazy_load()

    # 2) Simule une tête FT + classes et sauvegarde manuelle
    classes = ["X","Y","Z","W"]
    head = nn.Linear(fe._feat_dim, len(classes))
    with torch.no_grad():
        head.weight.uniform_(-0.01, 0.01); head.bias.zero_()

    save_path = tmp_path / "head_ft.pt"
    torch.save({
        "state_dict": head.state_dict(),
        "feat_dim": fe._feat_dim,
        "classes": classes,
        "normalize_feat": True,
    }, save_path)

    # 3) Recharge et vérifie mapping
    fe.load_head(str(save_path))
    assert fe._trained_head is not None
    assert list(fe.label_classes_) == classes
    assert fe.idx_to_label(2) == "Z"