from tools.lr_finder_torch import lr_find

res = lr_find(
    model,
    dataloader=train_loader,
    optimizer=optimizer,   # sera écrasé par start_lr
    criterion=criterion,
    device="cuda",         # ou "cpu" / "dml"
    start_lr=1e-6,
    end_lr=1.0,
    num_iters=200,
    beta=0.98,
    stop_factor=4.0,
    plot_path="lr_finder_torch.png"
)
print("LR suggérée (Torch):", res["suggested_lr"])
from tools.lr_finder_torch import lr_find
