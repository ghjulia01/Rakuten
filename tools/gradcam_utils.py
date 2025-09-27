# tools/gradcam_utils.py
def run_vit_attention_rollout_batch(*, backbone, image_dir, samples, labels, outdir, device="cpu",
                                    head_fusion="mean", discard_ratio=0.0):
    raise NotImplementedError("Attention rollout ViT non implémenté pour le moment.")