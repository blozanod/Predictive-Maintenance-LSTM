"""Head training loop: seed control, early stopping on validation, per-step loss
logging to CSV (Task 1 train.py; RESEARCH_PLAN sec.6 learning curves).

Seeds are threaded through numpy, torch, CUDA, and the DataLoader generator
(Task 2.3). ``deterministic`` turns on torch deterministic algorithms where
feasible. Nothing here touches the test set (Task 2.4).
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
from torch.utils.data import DataLoader, TensorDataset

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
        torch.backends.cudnn.benchmark = False


def _seeded_loader(
    X: np.ndarray, y: np.ndarray, batch_size: int, seed: int, shuffle: bool
) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(np.asarray(X, np.float32)),
                       torch.from_numpy(np.asarray(y, np.float32)))
    gen = torch.Generator()
    gen.manual_seed(seed)  # seeded shuffling for reproducible batches (Task 2.3)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=gen,
                      drop_last=False)


def train_head(
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    val_emb: np.ndarray,
    val_labels: np.ndarray,
    loss_type: str,
    config: Config,
    seed: Optional[int] = None,
    device: str = "cpu",
    log_csv_path: Optional[str | Path] = None,
) -> tuple[nn.Module, dict]:
    """Train an MLP head; early-stop on val RMSE; keep the best-val weights.

    Returns (best_model, history). ``history`` holds per-step train loss and
    per-epoch val loss/RMSE; if ``log_csv_path`` is given it is written as a tidy
    long CSV (step, epoch, metric, value) for the learning-curve plot.
    """
    if seed is None:
        seed = config.seed
    set_seed(seed, config.deterministic)

    input_dim = train_emb.shape[1]
    model = heads_mod.build_head(input_dim, loss_type, config).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.head_lr,
                           weight_decay=config.head_weight_decay)

    loader = _seeded_loader(train_emb, train_labels, config.head_batch_size, seed, True)
    val_X = torch.from_numpy(np.asarray(val_emb, np.float32)).to(device)
    val_y_t = torch.from_numpy(np.asarray(val_labels, np.float32)).to(device)

    history = {"step": [], "epoch": [], "train_loss": [],
               "val_epoch": [], "val_loss": [], "val_rmse": []}
    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    patience_left = config.head_early_stopping_patience
    step = 0
    for epoch in range(config.head_max_epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
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
            val_out = model(val_X)
            val_loss = float(heads_mod.compute_loss(val_out, val_y_t, loss_type, config))
            val_pred = heads_mod.decode(val_out, loss_type, config)
        val_rmse = rmse(np.asarray(val_labels, np.float64), val_pred)
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
def predict_head(model: nn.Module, emb: np.ndarray, loss_type: str,
                 config: Config, device: str = "cpu") -> np.ndarray:
    """Predict RUL for embedding rows using a trained head."""
    model.eval()
    X = torch.from_numpy(np.asarray(emb, np.float32)).to(device)
    return heads_mod.decode(model(X), loss_type, config)
