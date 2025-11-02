# src/features/cnn_features.py
# =======================================================
# Extraire un embedding CNN/ViT compatible scikit-learn
# → lit imageid/productid, batch, normalise, renvoie csr_matrix
# + FT optionnel, MixUp/CutMix, Grad-CAM (ResNet), Attention Rollout (ViT)
# =======================================================
from __future__ import annotations
import os
from typing import List, Dict, Optional, Tuple

import numpy as np
from PIL import Image
from scipy import sparse

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

import matplotlib.pyplot as plt
from datetime import datetime
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights,
    resnet101, ResNet101_Weights,
)

from concurrent.futures import ThreadPoolExecutor, as_completed

# Profiling décorateur
try:
    from src.utils.profiling import profile_func
except ImportError:
    def profile_func(func):
        return func

# Hugging Face
from transformers import AutoImageProcessor, AutoModel
# from transformers import ViTModel, ViTImageProcessor, ViTConfig
try:
    from transformers import AutoConfig
except Exception:
    from transformers import ViTConfig as AutoConfig

log = logging.getLogger("models.cnn_features")

ARCH_REGISTRY = {
    "resnet18":  (resnet18,  ResNet18_Weights.IMAGENET1K_V1, 512),
    "resnet50":  (resnet50,  ResNet50_Weights.IMAGENET1K_V2, 2048),
    "resnet101": (resnet101, ResNet101_Weights.IMAGENET1K_V2, 2048),
}


# ---------------------- HF wrapper pour ViT -----------------------------------
class HFBackbone(torch.nn.Module):
    def __init__(self, base, processor, device):
        super().__init__()
        self.base = base
        self.processor = processor
        self.device = device
        self.log = logging.getLogger("models.cnn_features")

    def forward(self, x):   # x: (B,3,H,W) in [0,1]
        imgs = [transforms.ToPILImage()(xi.cpu()) for xi in x]
        inputs = self.processor(images=imgs, return_tensors="pt").to(self.device)
        out = self.base(**inputs)
        # utiliser [CLS], pas le pooler
        if hasattr(out, "last_hidden_state"):
            z = out.last_hidden_state[:, 0]
        else:
            z0 = out[0] if isinstance(out, (tuple, list)) else out
            z = z0.mean(dim=1) if z0.ndim == 3 else z0
        return z


# ---------------------- Featurizer sklearn ------------------------------------
class CNNFeaturizer(BaseEstimator, TransformerMixin):
    """
    Transformer sklearn qui :
      - lit les fichiers images à partir de imageid/productid
      - extrait un embedding CNN pré-entraîné (ResNet) OU ViT (HF)
      - renvoie un csr_matrix (bien compatible avec TF-IDF sparse)
      - peut faire un fine-tuning léger + data augmentation (MixUp/CutMix)
      - peut générer des heatmaps Grad-CAM (ResNet) / Attention Rollout (ViT)
    """
    @profile_func
    def __init__(
        self,
        image_dir: str,
        arch: str = "resnet50",
        batch_size: int = 32,
        device: str = "auto",           # "auto" | "cpu" | "cuda" | "dml"
        use_imagenet_norm: bool = True, # normaliser comme ImageNet
        fallback_zero: bool = True,     # image manquante → vecteur 0
        dtype: str = "float32",         # "float32" conseillé (mémoire)
        num_workers: int = 0,

        # --- paramètres unfreeze / FT / HF ---
        trainable_last_n: int = 0,      # nb de paramètres à défiger (fallback)
        finetune_epochs: int = 0,       # 0 = pas de FT
        finetune_lr: float = 3e-4,
        finetune_weight_decay: float = 0.01,
        finetune_max_n: int = 8000,     # échantillon max pour FT
        trainable_last_layers: int = 1, # nb de blocks Transformer à défiger (ViT)
        hf_model_name: Optional[str] = None,     # ex: "google/vit-base-patch16-224"
        hf_revision: Optional[str] = "main",
        hf_feature_dim: Optional[int] = 768,           # ViT base = 768
        hf_use_fast: bool = True,      

        save_head_path: Optional[str] = None,    # si renseigné, sauvegarde la tête FT + classes
        save_head_normalize: bool = True,        # normalise l'embedding avant la tête (comme en FT)
        foreach: bool = True,                    # torch.optim foreach on/off

        # --- Data augmentation (FT uniquement) ---
        aug_hflip_p: float = 0.2,                # flip horizontal aléatoire
        aug_color_jitter: float = 0.0,           # 0 -> off ; sinon jitter léger
        mixup_alpha: float = 0.0,                # 0 -> off
        cutmix_alpha: float = 0.0,               # 0 -> off
        ft_patience: int = 3,                 # patience early stopping (FT)   
        random_resized_crop_scale: Tuple[float, float] = (0.9, 1.0),
        random_resized_crop_ratio: Tuple[float, float] = (0.95, 1.05),
        label_smoothing: float = 0.0, 
    ):
        self.image_dir = image_dir
        self.arch = arch
        self.batch_size = int(batch_size)
        self.device = device
        self.use_imagenet_norm = use_imagenet_norm
        self.fallback_zero = fallback_zero
        self.dtype = dtype
        self.num_workers = int(num_workers)

        # internes
        self._model = None
        self._preprocess = None
        self._feat_dim = None
        self._device_resolved = None

        # stats diagnostics
        self.n_total = 0
        self.n_loaded = 0
        self.n_missing = 0
        self.n_failed = 0

        # unfreeze / FT / HF
        self.trainable_last_n      = int(trainable_last_n)
        self.finetune_epochs       = int(finetune_epochs)
        self.finetune_lr           = float(finetune_lr)
        self.finetune_weight_decay = float(finetune_weight_decay)
        self.finetune_max_n        = int(finetune_max_n)
        self.trainable_last_layers = int(trainable_last_layers)

        self.hf_model_name  = hf_model_name
        self.hf_revision    = hf_revision
        self.hf_feature_dim = int(hf_feature_dim) if hf_feature_dim is not None else None
        self.save_head_path = save_head_path
        self.save_head_normalize = bool(save_head_normalize)
        self._trained_head: Optional[nn.Module] = None
        self.label_classes_: Optional[np.ndarray] = None
        self.foreach = bool(foreach)
        self.log = logging.getLogger("models.cnn_features")
        self.hf_use_fast    = bool(hf_use_fast)
        self.ft_patience = int(ft_patience)

        # Augmentations
        self.aug_hflip_p = float(aug_hflip_p)
        self.aug_color_jitter = float(aug_color_jitter)
        self.mixup_alpha = float(mixup_alpha)
        self.cutmix_alpha = float(cutmix_alpha)
        self.rrc_scale = tuple(random_resized_crop_scale) if random_resized_crop_scale is not None else None
        self.rrc_ratio = tuple(random_resized_crop_ratio) if random_resized_crop_ratio is not None else None
        self.random_resized_crop_scale = self.rrc_scale
        self.random_resized_crop_ratio = self.rrc_ratio
        self.label_smoothing = float(label_smoothing)

    # -------- Utilitaires -------------------------------------------------------
    def _load_one(self, path: str):
        if not os.path.exists(path):
            return None
        try:
            with Image.open(path).convert("RGB") as im:
                return self._preprocess(im)
        except Exception as e:
            self.log.warning("Image load fail path=%s -> %s", path, e)
            return None

    @profile_func
    def set_image_dir(self, new_dir: str):
        """Mettre à jour le dossier images (utile pour passer TRAIN → TEST)."""
        self.image_dir = new_dir

    @profile_func
    def _resolve_device(self):
        if self._device_resolved is not None:
            return self._device_resolved
        if self.device == "cuda":
            self._device_resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif self.device == "cpu":
            self._device_resolved = torch.device("cpu")
        elif self.device == "dml":
            import torch_directml
            self._device_resolved = torch_directml.device()
        else:  # auto
            if torch.cuda.is_available():
                self._device_resolved = torch.device("cuda")
            else:
                try:
                    import torch_directml
                    self._device_resolved = torch_directml.device()
                except Exception:
                    self._device_resolved = torch.device("cpu")
        return self._device_resolved

    def _hf_from_pretrained(self, name: str, revision: Optional[str]):
        """Chargement robuste HF (réseau → fallback offline), fast processor + no pooler."""
        try:
            cfg = AutoConfig.from_pretrained(name, revision=revision, local_files_only=False)
            if getattr(cfg, "model_type", None) == "vit":
                cfg.add_pooling_layer = False  # <-- évite l’avertissement sur le pooler
            proc = AutoImageProcessor.from_pretrained(
                name, revision=revision, use_fast=self.hf_use_fast, local_files_only=False
            )
            base = AutoModel.from_pretrained(name, revision=revision, config=cfg, local_files_only=False)
            self.log.info("[HF] %s téléchargé.", name)
        except Exception as e:
            self.log.warning("[HF] online KO (%s). Essai offline…", e)
            cfg = AutoConfig.from_pretrained(name, revision=revision, local_files_only=True)
            if getattr(cfg, "model_type", None) == "vit":
                cfg.add_pooling_layer = False
            proc = AutoImageProcessor.from_pretrained(
                name, revision=revision, use_fast=self.hf_use_fast, local_files_only=True
            )
            base = AutoModel.from_pretrained(name, revision=revision, config=cfg, local_files_only=True)
            self.log.info("[HF] %s trouvé en cache.", name)
        return proc, base


    @profile_func
    def _build_model(self):
        # HF branch (ViT & co)
        if self.hf_model_name:
            device = self._resolve_device()
            processor, base = self._hf_from_pretrained(self.hf_model_name, self.hf_revision)
            base = base.to(device).eval()
            model = HFBackbone(base, processor, device).to(device)
            self._feat_dim = int(self.hf_feature_dim or 768)  # ViT-base = 768
            preprocess = transforms.ToTensor()  # HF processor s'occupe du resize/crop/norm
            return model, preprocess

        # Torchvision branch (ResNet)
        arch_key = str(self.arch).lower()
        if arch_key not in ARCH_REGISTRY:
            raise ValueError(f"Architecture inconnue: {self.arch} (supportées: {list(ARCH_REGISTRY)})")
        ctor, weights_enum, feat_dim = ARCH_REGISTRY[arch_key]
        weights = weights_enum
        model = ctor(weights=weights)
        model.fc = nn.Identity()
        model.eval().to(self._resolve_device())
        preprocess = weights.transforms() if self.use_imagenet_norm else transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()
        ])
        self._feat_dim = feat_dim
        return model, preprocess

    def _set_trainable_tail(self, n_params: int):
        """Défige la queue du backbone :
           - HF/ViT : N derniers blocs Transformer (trainable_last_layers)
           - ResNet : layer4 complet
           - Fallback : derniers `n_params` paramètres
        """
        # Cas HF (ViT) : dernier(s) bloc(s) du Transformer
        if hasattr(self._model, "base") and hasattr(self._model.base, "encoder") and hasattr(self._model.base.encoder, "layer"):
            for p in self._model.base.parameters():
                p.requires_grad = False
            n_layers = max(1, int(getattr(self, "trainable_last_layers", 1)))
            for blk in list(self._model.base.encoder.layer)[-n_layers:]:
                for p in blk.parameters():
                    p.requires_grad = True
            self.log.info(f"[INFO] HF unfreeze: last {n_layers} Transformer blocks.")
            return

        # Cas ResNet (torchvision) : défige layer4
        if hasattr(self._model, "layer4"):
            for p in self._model.parameters():
                p.requires_grad = False
            for p in self._model.layer4.parameters():
                p.requires_grad = True
            self.log.info("[INFO] ResNet unfreeze: layer4.")
            return

        # Fallback générique : défige les n derniers paramètres
        for p in self._model.parameters():
            p.requires_grad = False
        if n_params > 0:
            tail = list(self._model.parameters())[-n_params:]
            for p in tail:
                p.requires_grad = True
            print(f"[INFO] Unfreeze last {n_params} parameters (generic)")

    @profile_func
    def _lazy_load(self):
        if self._model is None:
            self._model, self._preprocess = self._build_model()

    @profile_func
    def _path_from_row(self, row) -> str:
        # Nommage Rakuten : image_{imageid}_product_{productid}.jpg
        fname = f"image_{int(row['imageid'])}_product_{int(row['productid'])}.jpg"
        return os.path.join(self.image_dir, fname)

    # ------------------- Data Augmentation utils (MixUp/CutMix) ----------------
    @staticmethod
    def _mixup(x: torch.Tensor, y: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
        idx = torch.randperm(x.size(0), device=x.device)
        mixed_x = lam * x + (1 - lam) * x[idx, :]
        y_a, y_b = y, y[idx]
        return mixed_x, y_a, y_b, float(lam)

    @staticmethod
    def _rand_bbox(W: int, H: int, lam: float):
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        x1 = np.clip(cx - cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y2 = np.clip(cy + cut_h // 2, 0, H)
        return x1, y1, x2, y2

    @staticmethod
    def _cutmix(x: torch.Tensor, y: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
        idx = torch.randperm(x.size(0), device=x.device)
        x1, y1, x2, y2 = CNNFeaturizer._rand_bbox(x.size(3), x.size(2), lam)
        x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
        lam = 1 - ((x2 - x1) * (y2 - y1) / (x.size(-1) * x.size(-2) + 1e-9))
        y_a, y_b = y, y[idx]
        return x, y_a, y_b, float(lam)

    # ------------------------ API sklearn --------------------------------------
    @profile_func
    def fit(self, X, y=None):
        self._lazy_load()
        self.n_total = self.n_loaded = self.n_missing = self.n_failed = 0

        # --- Fine-tuning optionnel ---
        if self.finetune_epochs and y is not None and (self.trainable_last_n > 0 or getattr(self, "trainable_last_layers", 0) > 0):
            from sklearn.preprocessing import LabelEncoder

            # 1) Encode labels
            le = LabelEncoder()
            y_enc = le.fit_transform(np.asarray(y))

            # 2) Sous-échantillon rapide
            n = min(len(X), int(self.finetune_max_n))
            rng = np.random.RandomState(42)
            idx = rng.permutation(len(X))[:n]
            X_ft = X.iloc[idx].reset_index(drop=True)
            y_ft = y_enc[idx]

            # 2bis) Split interne 90/10
            X_tr_ft, X_va_ft, y_tr_ft, y_va_ft = train_test_split(
                X_ft, y_ft, test_size=0.1, random_state=42, stratify=y_ft)

            # 3) Préparer training
            device = self._resolve_device()
            self._set_trainable_tail(self.trainable_last_n)
            self._model.train()

            head = nn.Linear(int(self._feat_dim), int(len(le.classes_))).to(device)

            # Optim
            is_dml = (str(self.device).lower() == "dml") or ("directml" in str(type(device)).lower())
            param_groups = [
                {"params": [p for p in self._model.parameters() if p.requires_grad], "lr": self.finetune_lr},
                {"params": head.parameters(), "lr": self.finetune_lr},
            ]
            adamw_kwargs = dict(weight_decay=self.finetune_weight_decay, betas=(0.9, 0.999), eps=1e-8)
            try:
                opt = torch.optim.AdamW(param_groups, **adamw_kwargs,
                                        foreach=False if is_dml else bool(self.foreach), fused=False)
            except TypeError:
                opt = torch.optim.AdamW(param_groups, **adamw_kwargs)
            try:
                criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
            except TypeError:
                # PyTorch ancien : pas d’arg label_smoothing
                criterion = nn.CrossEntropyLoss()

            # Augmentations simples (hors MixUp/CutMix)
            aug_list = []
            if self.rrc_scale is not None and self.rrc_ratio is not None:
                aug_list.append(transforms.RandomResizedCrop(size=224, scale=self.rrc_scale, ratio=self.rrc_ratio))

            if self.aug_hflip_p > 0:
                aug_list.append(transforms.RandomHorizontalFlip(p=self.aug_hflip_p))
            if self.aug_color_jitter > 0:
                cj = self.aug_color_jitter
                aug_list.append(transforms.ColorJitter(brightness=cj, contrast=cj, saturation=cj, hue=min(0.1, cj)))
            aug = transforms.Compose(aug_list) if aug_list else None

            bs = int(self.batch_size)
            history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}
            best_f1, best_epoch = -1.0, -1
            best_backbone = None
            best_head = None
            patience = int(self.ft_patience) # patience early stopping

            def _iterate_rows_to_batches(X_rows, y_vec, batch_size):
                i = 0
                nloc = len(X_rows)
                while i < nloc:
                    j = min(i + batch_size, nloc)
                    paths_slice = [self._path_from_row(X_rows.iloc[k]) for k in range(i, j)]
                    imgs, ys = [], []
                    for k, p in enumerate(paths_slice, start=i):
                        if os.path.exists(p):
                            try:
                                with Image.open(p).convert("RGB") as im:
                                    t = self._preprocess(im)
                                    if aug is not None:
                                        # repasse par PIL pour les aug torchvision
                                        t = transforms.ToPILImage()(t)
                                        t = aug(t)
                                        t = transforms.ToTensor()(t)
                                    imgs.append(t)
                                ys.append(int(y_vec[k]))
                            except Exception:
                                pass
                    yield imgs, ys
                    i = j

            for epoch in range(int(self.finetune_epochs)):
                # ===== TRAIN =====
                self._model.train()
                running_tr, n_tr = 0.0, 0
                for imgs, ys in _iterate_rows_to_batches(X_tr_ft, y_tr_ft, bs):
                    if not imgs:
                        continue
                    batch = torch.stack(imgs, dim=0).to(device)
                    yb = torch.tensor(ys, dtype=torch.long, device=device)

                    # MixUp / CutMix
                    used_mix = None
                    if self.mixup_alpha > 0 and (self.cutmix_alpha <= 0 or np.random.rand() < 0.5):
                        batch, ya, yb2, lam = self._mixup(batch, yb, self.mixup_alpha)
                        used_mix = ("mixup", lam, ya, yb2)
                    elif self.cutmix_alpha > 0:
                        batch, ya, yb2, lam = self._cutmix(batch, yb, self.cutmix_alpha)
                        used_mix = ("cutmix", lam, ya, yb2)

                    feats = self._model(batch)
                    feats = feats / (feats.norm(dim=1, keepdim=True) + 1e-12)
                    logits = head(feats)

                    if used_mix is None:
                        loss = criterion(logits, yb)
                    else:
                        _, lam, ya, yb2 = used_mix
                        loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb2)

                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(self._model.parameters()) + list(head.parameters()), max_norm=1.0)
                    opt.step()

                    running_tr += float(loss.item()) * (yb.size(0))
                    n_tr += yb.size(0)
                train_loss = running_tr / max(1, n_tr)

                # ===== VAL =====
                self._model.eval()
                running_val, n_val = 0.0, 0
                all_preds, all_true = [], []
                with torch.no_grad():
                    for imgs, ys in _iterate_rows_to_batches(X_va_ft, y_va_ft, bs):
                        if not imgs:
                            continue
                        batch = torch.stack(imgs, dim=0).to(device)
                        yb = torch.tensor(ys, dtype=torch.long, device=device)

                        feats = self._model(batch)
                        feats = feats / (feats.norm(dim=1, keepdim=True) + 1e-12)
                        logits = head(feats)
                        loss = criterion(logits, yb)

                        running_val += float(loss.item()) * yb.size(0)
                        n_val += yb.size(0)

                        preds = logits.argmax(dim=1).detach().cpu().numpy()
                        all_preds.append(preds)
                        all_true.append(yb.detach().cpu().numpy())

                val_loss = running_val / max(1, n_val) if n_val else float("nan")
                y_true = np.concatenate(all_true) if all_true else np.array([])
                y_pred = np.concatenate(all_preds) if all_preds else np.array([])
                val_f1  = f1_score(y_true, y_pred, average="macro") if y_true.size else float("nan")
                val_acc = accuracy_score(y_true, y_pred)            if y_true.size else float("nan")

                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                history["val_f1"].append(val_f1)
                history["val_acc"].append(val_acc)

                self.log.info(f"[FT] epoch {epoch+1}/{self.finetune_epochs}  "
                              f"loss_tr={train_loss:.4f}  loss_val={val_loss:.4f}  "
                              f"F1_val={val_f1:.4f}  acc_val={val_acc:.4f}")

                # best checkpoint (F1)

                if y_true.size and val_f1 > best_f1 + 1e-4:
                    best_f1, best_epoch = val_f1, epoch
                    best_backbone = {k: v.detach().cpu().clone() for k,v in self._model.state_dict().items()}
                    best_head     = {k: v.detach().cpu().clone() for k,v in head.state_dict().items()}
                elif epoch - best_epoch >= patience:
                    self.log.info(f"[FT] early stop @ epoch {epoch+1} (patience={patience})")
                    break

            # Recharger meilleur état
            if best_backbone is not None and best_head is not None:
                self._model.load_state_dict(best_backbone)
                head.load_state_dict(best_head)

            # === COURBES ===
            os.makedirs("results", exist_ok=True)
            arch_or_name = (self.hf_model_name or self.arch).replace("/", "-")
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")

            plt.figure(figsize=(7.5,4))
            plt.plot(history["train_loss"], label="train")
            plt.plot(history["val_loss"],   label="val")
            plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Fine-tuning — loss"); plt.legend(); plt.tight_layout()
            plt.savefig(f"results/ft_{arch_or_name}_{ts}_loss.png", dpi=150); plt.close()

            plt.figure(figsize=(7.5,4))
            plt.plot(history["val_f1"], label="F1-macro (val)")
            plt.plot(history["val_acc"], label="accuracy (val)", linestyle="--")
            plt.xlabel("epoch"); plt.ylabel("score"); plt.title("Fine-tuning — validation F1/accuracy"); plt.legend(); plt.tight_layout()
            plt.savefig(f"results/ft_{arch_or_name}_{ts}_f1.png", dpi=150); plt.close()

            print(f"[INFO] Courbes FT : results/ft_{arch_or_name}_{ts}_loss.png | results/ft_{arch_or_name}_{ts}_f1.png")

            # 5) Stockage tête
            self._model.eval()
            self._trained_head = head.to(device).eval()
            from sklearn.preprocessing import LabelEncoder
            self.label_classes_ = le.classes_
            if self.save_head_path:
                to_save = {
                    "state_dict": self._trained_head.state_dict(),
                    "feat_dim": int(self._feat_dim),
                    "classes": self.label_classes_.tolist(),
                    "normalize_feat": bool(self.save_head_normalize),
                }
                torch.save(to_save, self.save_head_path)

        return self

    @profile_func
    def attach_head(self, head: nn.Module, classes: List[str] | np.ndarray, normalize_feat: Optional[bool] = None):
        """Attache une tête Linear entraînée + liste de classes."""
        self._lazy_load()
        self._trained_head = head.to(self._resolve_device()).eval()
        self.label_classes_ = np.asarray(classes)
        if normalize_feat is not None:
            self.save_head_normalize = bool(normalize_feat)

    @profile_func
    def load_head(self, path: str):
        """Charge une tête Linear + classes depuis torch.save(...)"""
        self._lazy_load()
        chk = torch.load(path, map_location=self._resolve_device())
        feat_dim = int(chk.get("feat_dim", int(self._feat_dim or 0)))
        classes = chk["classes"]
        head = nn.Linear(feat_dim, len(classes))
        head.load_state_dict(chk["state_dict"])
        self._trained_head = head.to(self._resolve_device()).eval()
        self.label_classes_ = np.asarray(classes)
        self.save_head_normalize = bool(chk.get("normalize_feat", True))

    @torch.no_grad()
    def _embed_batch(self, batch: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Embeddings [B, feat_dim] avec option de L2-normalisation (comme en FT)."""
        self._lazy_load()
        feats = self._model(batch)  # [B, feat_dim]
        if normalize:
            feats = feats / (feats.norm(dim=1, keepdim=True) + 1e-12)
        return feats

    @torch.no_grad()
    def predict_logits_from_paths(self, paths: List[str]) -> np.ndarray:
        """
        Calcule les logits par classe pour une liste de chemins d'images.
        Requiert self._trained_head et self.label_classes_.
        """
        assert self._trained_head is not None, "Aucune tête entraînée attachée/chargée. Utilise attach_head(...) ou load_head(...)."
        device = self._resolve_device()
        bs = int(self.batch_size)
        logits_all = []

        i = 0
        while i < len(paths):
            j = min(i + bs, len(paths))
            imgs = []
            for p in paths[i:j]:
                if os.path.exists(p):
                    try:
                        with Image.open(p).convert("RGB") as im:
                            imgs.append(self._preprocess(im))
                    except Exception:
                        imgs.append(None)
                else:
                    imgs.append(None)

            if any(t is not None for t in imgs):
                batch = torch.stack([t for t in imgs if t is not None], dim=0).to(device)
                feats = self._embed_batch(batch, normalize=self.save_head_normalize)
                logits = self._trained_head(feats).detach().cpu().numpy()
                it = iter(logits)
                for t in imgs:
                    if t is None:
                        logits_all.append(np.full((1, len(self.label_classes_)), np.nan))
                    else:
                        logits_all.append(next(it)[None, :])
            else:
                logits_all.extend([np.full((1, len(self.label_classes_)), np.nan) for _ in imgs])
            i = j

        return np.vstack(logits_all)

    def idx_to_label(self, class_idx: int) -> str:
        if self.label_classes_ is None:
            return str(class_idx)
        return str(self.label_classes_[class_idx])

    @torch.no_grad()
    def predict_proba_from_paths(self, paths: List[str]) -> np.ndarray:
        logits = self.predict_logits_from_paths(paths)
        mask = ~np.isnan(logits).any(axis=1)
        proba = np.full_like(logits, np.nan, dtype=np.float64)
        if mask.any():
            e = np.exp(logits[mask] - np.max(logits[mask], axis=1, keepdims=True))
            proba[mask] = e / (e.sum(axis=1, keepdims=True) + 1e-12)
        return proba

    @torch.no_grad()
    def topk_from_paths(self, paths: List[str], k: int = 5):
        logits = self.predict_logits_from_paths(paths)
        proba  = self.predict_proba_from_paths(paths)
        out = []
        for i in range(logits.shape[0]):
            if np.isnan(logits[i]).all():
                out.append([])
                continue
            idxs = np.argsort(-logits[i])[:k]
            items = []
            for j in idxs:
                items.append((int(j), self.idx_to_label(int(j)), float(logits[i, j]), float(proba[i, j])))
            out.append(items)
        return out

    @profile_func
    def transform(self, X):
        """
        X : DataFrame avec colonnes 'imageid' et 'productid'
        Retour : csr_matrix (n_samples, feat_dim)
        """
        self._lazy_load()
        device = self._resolve_device()

        n  = len(X)
        bs = int(self.batch_size)
        d  = int(self._feat_dim)
        out = np.zeros((n, d), dtype=self.dtype)

        # Chemins des fichiers
        paths = [self._path_from_row(X.iloc[i]) for i in range(n)]

        # Stats
        self.n_total = n
        self.n_loaded = 0
        self.n_missing = 0
        self.n_failed = 0

        with torch.no_grad():
            i = 0
            while i < n:
                j = min(i + bs, n)
                paths_slice = paths[i:j]

                imgs, idxs = [], []

                if self.num_workers > 0:
                    with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
                        futs = {ex.submit(self._load_one, p): k for k, p in enumerate(paths_slice, start=i)}
                        for fut in as_completed(futs):
                            k = futs[fut]
                            t = fut.result()
                            if t is None:
                                if not os.path.exists(paths[k]):
                                    self.n_missing += 1
                                else:
                                    self.n_failed += 1
                            else:
                                imgs.append(t)
                                idxs.append(k)
                else:
                    for k, p in enumerate(paths_slice, start=i):
                        if os.path.exists(p):
                            try:
                                with Image.open(p).convert("RGB") as im:
                                    imgs.append(self._preprocess(im))
                                idxs.append(k)
                            except Exception:
                                self.n_failed += 1
                        else:
                            self.n_missing += 1

                if imgs:
                    batch = torch.stack(imgs, dim=0).to(device)
                    feats = self._model(batch).detach().cpu().numpy().astype(self.dtype, copy=False)
                    # L2-normalisation
                    norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12
                    feats = feats / norms
                    for t, k in enumerate(idxs):
                        out[k, :] = feats[t]
                    self.n_loaded += len(idxs)
                if i % 200 == 0:
                    self.log.info("CNN progress: sample %d/%d", i, n)

                i = j

        return sparse.csr_matrix(out)

    # ------------------------ Diagnostics & save --------------------------------
    @profile_func
    def get_diagnostics(self) -> Dict[str, object]:
        input_size = None
        try:
            tf = getattr(self, "_preprocess", None)
            input_size = getattr(tf, "crop_size", None) or getattr(tf, "size", None)
            if isinstance(input_size, (tuple, list)):
                input_size = input_size[0]
        except Exception:
            pass
        return {
            "arch": self.arch if not self.hf_model_name else self.hf_model_name,
            "device": str(self._resolve_device()),
            "feat_dim": int(self._feat_dim or 0),
            "batch_size": int(self.batch_size),
            "use_imagenet_norm": getattr(self, "use_imagenet_norm", True),
            "n_total": int(self.n_total),
            "n_loaded": int(self.n_loaded),
            "n_missing": int(self.n_missing),
            "n_failed": int(self.n_failed),
            "loaded_ratio": float(self.n_loaded / max(1, self.n_total)),
            "num_workers" : int(self.num_workers),
            "input_size": int(input_size) if input_size else None,
            "trainable_last_n": int(getattr(self, "trainable_last_n", 0)),
            "finetune_epochs": int(getattr(self, "finetune_epochs", 0)),
            "hf_model_name": getattr(self, "hf_model_name", None),
            "foreach": bool(getattr(self, "foreach", False)),
            "mixup_alpha": float(self.mixup_alpha),
            "cutmix_alpha": float(self.cutmix_alpha),
            "hf_use_fast": bool(getattr(self, "hf_use_fast", True)),
        }

    @profile_func
    def save_model(self, path):
        state = {
            'state_dict': self._model.state_dict(),
            'arch': self.arch,
            'feat_dim': self._feat_dim,
            'use_imagenet_norm': self.use_imagenet_norm,
        }
        torch.save(state, path)

    # ------------------------ Grad-CAM (ResNet) --------------------------------
    def export_gradcam(self, image_paths: List[str], out_dir: str, alpha_overlay: float = 0.65):
        """Heatmaps Grad-CAM pour ResNet (layer4)."""
        self._lazy_load()
        if not hasattr(self._model, "layer4"):
            self.log.warning("Grad-CAM: layer4 introuvable (probablement ViT).")
            return
        os.makedirs(out_dir, exist_ok=True)
        device = self._resolve_device()
        model = self._model
        model.eval()

        feats, grads = [], []

        def fwd_hook(m, i, o): feats.append(o.detach())
        def bwd_hook(m, gi, go): grads.append(go[0].detach())

        h1 = model.layer4.register_forward_hook(fwd_hook)
        h2 = model.layer4.register_full_backward_hook(bwd_hook)

        for p in image_paths:
            try:
                with Image.open(p).convert("RGB") as im:
                    x = self._preprocess(im).unsqueeze(0).to(device)
                model.zero_grad()
                out = model(x)               # [1, C, H, W] après layer4→global pool→fc(Identity ici)
                score = out.mean()
                score.backward()

                A = feats.pop().squeeze(0)   # [C,H,W]
                G = grads.pop().squeeze(0)   # [C,H,W]
                weights = G.mean(dim=(1,2))
                cam = F.relu((weights[:, None, None] * A).sum(0))
                cam = (cam - cam.min()) / (cam.max() + 1e-9)
                cam = F.interpolate(cam[None, None, ...], size=im.size[::-1], mode="bilinear", align_corners=False).squeeze()

                heat = (cam.cpu().numpy() * 255).astype(np.uint8)
                heat = plt.cm.jet(heat)[:, :, :3]
                overlay = (1 - alpha_overlay)*np.asarray(im)/255. + alpha_overlay*heat
                overlay = np.clip(overlay, 0, 1)
                out_path = os.path.join(out_dir, os.path.basename(p).rsplit(".",1)[0] + "_gradcam.png")
                plt.imsave(out_path, overlay)
            except Exception as e:
                self.log.warning("Grad-CAM fail for %s: %s", p, e)

        h1.remove(); h2.remove()

    # ------------------- Attention Rollout (ViT) --------------------------------
    def _get_vit_model_processor(self):
        """Renvoie (model_HF, processor) pour rollout; construit si besoin."""
        self._lazy_load()
        if hasattr(self._model, "base") and hasattr(self._model, "processor"):
            return self._model.base, self._model.processor
        # si pas construit via HFBackbone (cas rare), on recharge
        proc, base = self._hf_from_pretrained(self.hf_model_name, self.hf_revision)
        base = base.to(self._resolve_device()).eval()
        return base, proc

    def export_vit_attention_rollout(self, image_paths: List[str], out_dir: str,
                                     head_fusion: str = "mean", discard_ratio: float = 0.0, alpha_residual: float = 0.5):
        """Attention Rollout pour ViT (CLS→patches)."""
        if not self.hf_model_name:
            self.log.warning("Attention Rollout: nécessite une branche ViT.")
            return
        os.makedirs(out_dir, exist_ok=True)
        model, processor = self._get_vit_model_processor()
        device = self._resolve_device()

        def fuse_heads(att):  # [H,T,T] -> [T,T]
            return att.max(dim=0).values if head_fusion == "max" else att.mean(dim=0)

        for p in image_paths:
            try:
                with Image.open(p).convert("RGB") as im:
                    inputs = processor(images=im, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    out = model(**inputs, output_attentions=True)
                atts = out.attentions  # list len=L, each [1,H,T,T]
                joint = torch.eye(atts[0].shape[-1], device=device)
                for A in atts:
                    A = fuse_heads(A[0])                     # [T,T]
                    if discard_ratio > 0:
                        flat = A.view(-1)
                        k = int(flat.numel() * discard_ratio)
                        if k > 0:
                            thresh = torch.topk(-flat, k).values.max().neg()
                            A = torch.where(A < thresh, torch.zeros_like(A), A)
                    A = A + alpha_residual * torch.eye(A.size(0), device=device)
                    A = A / A.sum(dim=-1, keepdim=True)
                    joint = A @ joint
                cls_to_patches = joint[0, 1:]             # [T-1]
                grid = int(np.sqrt(cls_to_patches.numel()))
                mask = cls_to_patches.reshape(1, 1, grid, grid)
                mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-9)
                mask = F.interpolate(mask, size=im.size[::-1], mode="bilinear", align_corners=False)[0,0]
                heat = (mask.detach().cpu().numpy() * 255).astype(np.uint8)
                heat = plt.cm.jet(heat)[:, :, :3]
                overlay = 0.35*np.asarray(im)/255. + 0.65*heat
                overlay = np.clip(overlay, 0, 1)
                out_path = os.path.join(out_dir, os.path.basename(p).rsplit(".",1)[0] + "_vitrollout.png")
                plt.imsave(out_path, overlay)
            except Exception as e:
                self.log.warning("ViT rollout fail for %s: %s", p, e)