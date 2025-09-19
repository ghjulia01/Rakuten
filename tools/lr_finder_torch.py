import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn.utils import clip_grad_norm_

@torch.no_grad()
def _set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr

def lr_find(model, dataloader, optimizer, criterion, device="cuda",
            start_lr=1e-6, end_lr=1.0, num_iters=200, beta=0.98,
            stop_factor=4.0, max_grad_norm=None, log_scale=True,
            plot_path="lr_finder_torch.png"):
    """LR range test (Leslie Smith). Retourne dict: lrs, losses, suggested_lr, plot_path."""
    model.train()
    model.to(device)

    mult = (end_lr / start_lr) ** (1.0 / max(1, num_iters))
    _set_lr(optimizer, start_lr)

    avg_loss = 0.0
    best_loss = float("inf")
    losses, lrs = [], []
    it = 0
    data_iter = iter(dataloader)

    while it < num_iters:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        if isinstance(batch, (list, tuple)):
            inputs, targets = batch[0], batch[1]
        else:
            inputs, targets = batch

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        if max_grad_norm is not None:
            clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        it += 1
        loss_val = float(loss.detach().cpu().item())
        avg_loss = beta * avg_loss + (1 - beta) * loss_val
        smooth_loss = avg_loss / (1 - beta ** it)

        lr = optimizer.param_groups[0]['lr']
        lrs.append(lr)
        losses.append(smooth_loss)

        if smooth_loss < best_loss:
            best_loss = smooth_loss
        if smooth_loss > stop_factor * best_loss or math.isnan(smooth_loss) or math.isinf(smooth_loss):
            break

        lr *= mult
        _set_lr(optimizer, lr)

    # plot
    plt.figure(figsize=(7,4))
    if log_scale:
        plt.xscale("log")
    plt.plot(lrs, losses)
    plt.xlabel("Learning rate")
    plt.ylabel("Smoothed loss")
    plt.title("LR Finder (PyTorch)")
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    # suggestion
    losses_np = np.array(losses)
    lrs_np = np.array(lrs)
    if len(losses_np) == 0:
        suggested = None
    else:
        min_idx = int(np.argmin(losses_np))
        min_loss = float(losses_np[min_idx])
        thresh = min_loss * 1.10
        left_idx = 0
        for i in range(min_idx, -1, -1):
            if losses_np[i] > thresh:
                left_idx = i + 1
                break
        suggested = float(lrs_np[max(left_idx, 0)])

    return {"lrs": lrs, "losses": losses, "suggested_lr": suggested, "plot_path": plot_path}