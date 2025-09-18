# gradcam_resnet_head.py
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json, argparse


# Génère un fichier JSON index_to_label à partir d'une tête entraînée
# python gradcam_resnet_head.py --head artifacts/head_ft.pt --out features/labels_map.json
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True, help="artifacts/head_ft.pt")
    ap.add_argument("--out", required=True, help="features/labels_map.json")
    args = ap.parse_args()

    chk = torch.load(args.head, map_location="cpu")
    classes = chk.get("classes")
    if classes is None:
        raise SystemExit("Pas de 'classes' dans le head — est-ce que l'entrainement a bien fonctionné /enregistré la tête ?")

    payload = {
        "classes": classes,                                   # liste ordonnée
        "index_to_label": {str(i): lbl for i, lbl in enumerate(classes)}
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"OK → {args.out} ({len(classes)} classes)")


class ResNetGradCAMWithHead:
    def __init__(self, featurizer):
        """
        featurizer: instance CNNFeaturizer avec:
          - featurizer._model (ResNet, fc=Identity)
          - featurizer._preprocess
          - featurizer._trained_head (Linear) chargé/attaché
          - featurizer.label_classes_
        """
        self.fe = featurizer
        self.model = featurizer._model
        self.preprocess = featurizer._preprocess
        self.device = featurizer._resolve_device()
        assert self.fe._trained_head is not None, "Aucune tête attachée: appelle fe.attach_head(...) ou fe.load_head(...)."

        self.head = self.fe._trained_head
        self.normalize_feat = bool(getattr(self.fe, "save_head_normalize", True))

        self._acts = None
        self._grads = None

        def fwd_hook(module, inp, out):
            self._acts = out.detach()
            def bwd_hook(grad):
                self._grads = grad.detach()
            out.register_hook(bwd_hook)

        if hasattr(self.model, "layer4"):
            self._handle = self.model.layer4.register_forward_hook(fwd_hook)
        else:
            raise RuntimeError("layer4 introuvable (attendu pour ResNet).")

        self.model.eval()
        self.head.eval()

    def __del__(self):
        if hasattr(self, "_handle") and self._handle is not None:
            self._handle.remove()

    @torch.no_grad()
    def _prepare(self, path):
        img = Image.open(path).convert("RGB")
        x = self.preprocess(img)
        return img, x.unsqueeze(0).to(self.device)

    def generate(self, image_path, class_idx=None, overlay_alpha=0.45, show=True):
        orig_img, xb = self._prepare(image_path)

        # Embedding
        emb = self.model(xb)  # [1, feat_dim]
        if self.normalize_feat:
            emb = F.normalize(emb, dim=1)

        logits = self.head(emb)  # [1, C]
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        # Backward pour le score de la classe
        self.model.zero_grad(set_to_none=True)
        self.head.zero_grad(set_to_none=True)
        score = logits[0, class_idx]
        score.backward(retain_graph=True)

        # Grad-CAM depuis layer4
        acts = self._acts[0]   # [C, H, W]
        grads = self._grads[0] # [C, H, W]
        weights = grads.mean(dim=(1, 2))
        cam = torch.relu((weights[:, None, None] * acts).sum(dim=0))
        cam = (cam - cam.min()) / (cam.max() + 1e-12)
        cam_np = cam.detach().cpu().numpy()

        cam_img = Image.fromarray(np.uint8(cam_np * 255)).resize(orig_img.size, resample=Image.BILINEAR)

        # Heatmap + overlay
        cm = plt.get_cmap('jet')
        cam_color = (cm(np.array(cam_img)/255.0) * 255).astype(np.uint8)  # RGBA
        heatmap = Image.fromarray(cam_color[:, :, :3])
        overlay = Image.blend(orig_img.convert("RGB"), heatmap.convert("RGB"), alpha=float(overlay_alpha))

        # Mapping index -> label si dispo
        try:
            label = self.fe.idx_to_label(class_idx)
        except Exception:
            label = str(class_idx)

        if show:
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 3, 1); plt.title("Image"); plt.axis("off"); plt.imshow(orig_img)
            plt.subplot(1, 3, 2); plt.title("Heatmap"); plt.axis("off"); plt.imshow(heatmap)
            plt.subplot(1, 3, 3); plt.title(f"Overlay (class={label})"); plt.axis("off"); plt.imshow(overlay)
            plt.tight_layout(); plt.show()

        return heatmap, overlay, class_idx, label, logits.detach().cpu().numpy()