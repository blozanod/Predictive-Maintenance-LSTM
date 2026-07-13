"""Head training loop: seed control, early stopping on validation, per-step loss
logging to CSV (Task 1 train.py; RESEARCH_PLAN sec.6 learning curves).

Stage B economics (Task 2): the head trains entirely on-device. Features/labels are
moved to the GPU ONCE (by the sweep) and minibatches are sliced with a seeded
on-device permutation -- no DataLoader, no workers, no per-batch host->device
copies. Determinism is preserved (``use_deterministic_algorithms`` stays on for
heads); the embedding pass is the only place cuDNN benchmark mode is used, and it is
cached, not trained.

Seeds are threaded through numpy, torch, and CUDA (Task 2.3). Nothing here touches
the test set (Task 2.4).
"""

from __future__ import annotations

import csv
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn

from .config import Config
from . import heads as heads_mod
from .evaluate import rmse


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python/numpy/torch/CUDA for replicability (Task 2.3)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # cuBLAS determinism
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        torch.backends.cudnn.benchmark = False  # heads stay deterministic (Task 2)


def _to_device_tensor(x, device: str, dtype=torch.float32) -> torch.Tensor:
    """Array-like or tensor -> tensor on ``device`` (no copy if already correct)."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(x, np.float32), dtype=dtype, device=device)


def train_head(
    train_emb,
    train_labels,
    val_emb,
    val_labels,
    loss_type: str,
    config: Config,
    seed: Optional[int] = None,
    device: str = "cpu",
    log_csv_path: Optional[str | Path] = None,
) -> tuple[nn.Module, dict]:
    """Train an MLP head; early-stop on val RMSE; keep the best-val weights.

    Inputs may be numpy arrays or (ideally) tensors already on ``device`` -- the
    sweep passes on-GPU tensors so no host copies happen per cell. Returns
    (best_model, history) with per-step train loss + per-epoch val loss/RMSE; a tidy
    long CSV (step, epoch, metric, value) is written if ``log_csv_path`` is given.
    """
    if seed is None:
        seed = config.seed
    set_seed(seed, config.deterministic)

    Xtr = _to_device_tensor(train_emb, device)
    ytr = _to_device_tensor(train_labels, device)
    Xva = _to_device_tensor(val_emb, device)
    yva = _to_device_tensor(val_labels, device)
    val_labels_np = yva.detach().cpu().numpy().astype(np.float64)

    input_dim = Xtr.shape[1]
    model = heads_mod.build_head(input_dim, loss_type, config).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.head_lr,
                           weight_decay=config.head_weight_decay)

    n = Xtr.shape[0]
    bs = config.head_batch_size
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)  # seeded on-device shuffling for reproducible batches (Task 2.3)

    history = {"step": [], "epoch": [], "train_loss": [],
               "val_epoch": [], "val_loss": [], "val_rmse": []}
    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    patience_left = config.head_early_stopping_patience
    step = 0
    for epoch in range(config.head_max_epochs):
        model.train()
        perm = torch.randperm(n, generator=gen, device=device)
        for start in range(0, n, bs):
            idx = perm[start : start + bs]
            xb, yb = Xtr[idx], ytr[idx]
            opt.zero_grad()
            out = model(xb)
            loss = heads_mod.compute_loss(out, yb, loss_type, config)
            loss.backward()
            opt.step()
            history["step"].append(step)
            history["epoch"].append(epoch)
            history["train_loss"].append(float(loss.detach().cpu()))
            step += 1

        # ---- validation (early stopping on val RMSE in RUL units) ----
        model.eval()
        with torch.no_grad():
            val_out = model(Xva)
            val_loss = float(heads_mod.compute_loss(val_out, yva, loss_type, config))
            val_pred = heads_mod.decode(val_out, loss_type, config)
        val_rmse = rmse(val_labels_np, val_pred)
        history["val_epoch"].append(epoch)
        history["val_loss"].append(val_loss)
        history["val_rmse"].append(val_rmse)

        if val_rmse < best_val - 1e-6:
            best_val = val_rmse
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_left = config.head_early_stopping_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    history["best_val_rmse"] = best_val
    if log_csv_path is not None:
        _write_history_csv(history, log_csv_path)
    return model, history


def _write_history_csv(history: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "epoch", "metric", "value"])
        for s, e, tl in zip(history["step"], history["epoch"], history["train_loss"]):
            w.writerow([s, e, "train_loss", tl])
        for e, vl, vr in zip(history["val_epoch"], history["val_loss"], history["val_rmse"]):
            w.writerow(["", e, "val_loss", vl])
            w.writerow(["", e, "val_rmse", vr])


@torch.no_grad()
def predict_head(model: nn.Module, emb, loss_type: str,
                 config: Config, device: str = "cpu") -> np.ndarray:
    """Predict RUL for embedding rows using a trained head."""
    model.eval()
    X = _to_device_tensor(emb, device)
    return heads_mod.decode(model(X), loss_type, config)
