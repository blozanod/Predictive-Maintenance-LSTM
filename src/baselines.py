"""From-scratch and classical baselines (RESEARCH_PLAN sec.4).

All baselines share the interface ``fit(train_windows, train_labels, val_windows,
val_labels)`` / ``predict(windows) -> RUL`` and consume the SAME raw windows the
TSFM path caches, so a sweep never re-embeds (Task 3, Stage B). Any fitted scaler
is fit on the current fraction's TRAIN windows only (Task 2.4).

Reuse over reimplementation (Task 2.1):
  * GBM  -> ``lightgbm.LGBMRegressor`` on per-window summary statistics.
  * MiniRocket -> ``sktime`` ``MiniRocketMultivariate`` + ridge (RidgeCV).
  * catch22_gbm -> ``pycatch22`` (the 22 canonical time-series features per channel)
    -> ``lightgbm.LGBMRegressor`` -- the hand-crafted-indicator foil (RESEARCH_PLAN §6,
    RQ-D: "do TSFMs make hand-crafted indicators obsolete?").
The 1D-CNN and LSTM are genuine from-scratch NN baselines (no maintained library
reference exists for the C-MAPSS-specific architectures); they are the comparison
targets, implemented small and standard.

Heavy deps (lightgbm, sktime, pycatch22) are imported lazily so CPU smoke tests that
only use the mean/NN baselines need not install them.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch  # core dependency (also used by train.py); lightgbm/sktime stay lazy
import torch.nn as nn

from .config import Config
from . import data as data_mod


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------
class Baseline:
    name = "baseline"

    def fit(self, train_windows, train_labels, val_windows=None, val_labels=None):
        raise NotImplementedError

    def predict(self, windows) -> np.ndarray:
        raise NotImplementedError

    def _clip(self, rul: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(rul, np.float64), 0.0, self.config.max_rul)


# ---------------------------------------------------------------------------
# Floor baselines (cheap; catch bugs -- RESEARCH_PLAN sec.4)
# ---------------------------------------------------------------------------
class PredictMeanBaseline(Baseline):
    """Predict the mean training RUL for every test window."""
    name = "predict_mean"

    def __init__(self, config: Config, seed: int = 0):
        self.config = config
        self._mean = 0.0

    def fit(self, train_windows, train_labels, val_windows=None, val_labels=None):
        self._mean = float(np.mean(train_labels))
        return self

    def predict(self, windows) -> np.ndarray:
        return self._clip(np.full(len(windows), self._mean))


# ---------------------------------------------------------------------------
# GBM on per-window statistics (industrial default; the arXiv:2606.11990 baseline)
# ---------------------------------------------------------------------------
def window_statistics(windows: np.ndarray) -> np.ndarray:
    """Per-channel summary features: mean/std/min/max/q25/median/q75/slope/last.
    (Standard feature engineering feeding LightGBM -- not a library reimpl.)"""
    N, W, C = windows.shape
    x = windows.astype(np.float64)
    t = np.arange(W, dtype=np.float64)
    t_c = t - t.mean()
    denom = np.sum(t_c ** 2) or 1.0
    slope = np.tensordot(x - x.mean(axis=1, keepdims=True), t_c, axes=([1], [0])) / denom
    feats = [
        x.mean(axis=1), x.std(axis=1), x.min(axis=1), x.max(axis=1),
        np.percentile(x, 25, axis=1), np.percentile(x, 50, axis=1),
        np.percentile(x, 75, axis=1), slope, x[:, -1, :],
    ]
    return np.concatenate(feats, axis=1).astype(np.float32)  # (N, 9*C)


class GBMBaseline(Baseline):
    name = "gbm"

    def __init__(self, config: Config, seed: int = 0):
        self.config = config
        self.seed = seed
        self._model = None

    def fit(self, train_windows, train_labels, val_windows=None, val_labels=None):
        from lightgbm import LGBMRegressor  # reuse reference impl (Task 2.1)
        Xtr = window_statistics(train_windows)
        self._model = LGBMRegressor(random_state=self.seed, n_estimators=500,
                                    learning_rate=0.05, verbose=-1,
                                    n_jobs=-1)  # all cores (Task 2 perf)
        fit_kw = {}
        if val_windows is not None and len(val_windows):
            fit_kw["eval_set"] = [(window_statistics(val_windows), val_labels)]
        self._model.fit(Xtr, np.asarray(train_labels, np.float64), **fit_kw)
        return self

    def predict(self, windows) -> np.ndarray:
        return self._clip(self._model.predict(window_statistics(windows)))


# ---------------------------------------------------------------------------
# MiniRocket + ridge (generic frozen features -- the foil for TSFM embeddings)
# ---------------------------------------------------------------------------
class MiniRocketBaseline(Baseline):
    name = "minirocket"

    def __init__(self, config: Config, seed: int = 0):
        self.config = config
        self.seed = seed
        self._transform = None
        self._ridge = None

    @staticmethod
    def _to_sktime(windows: np.ndarray) -> np.ndarray:
        # sktime panel format: (n_instances, n_channels, series_length).
        return np.transpose(windows, (0, 2, 1)).astype(np.float32)

    def fit(self, train_windows, train_labels, val_windows=None, val_labels=None):
        # sktime's MiniRocket requires series length >= 9; the default
        # window_size=30 satisfies this comfortably (Dempster et al. 2021).
        from sktime.transformations.panel.rocket import MiniRocketMultivariate  # reuse
        from sklearn.linear_model import RidgeCV

        # n_jobs=-1: MiniRocket's kernel transform parallelizes trivially over cores
        # (joblib inside sktime). MiniRocket stays CPU (Task 2 baselines note).
        self._transform = MiniRocketMultivariate(random_state=self.seed, n_jobs=-1)
        Xtr = self._transform.fit_transform(self._to_sktime(train_windows))
        # Ridge with CV over alphas -- standard ROCKET head (Dempster et al. 2021).
        self._ridge = RidgeCV(alphas=np.logspace(-3, 3, 10))
        self._ridge.fit(np.asarray(Xtr), np.asarray(train_labels, np.float64))
        return self

    def predict(self, windows) -> np.ndarray:
        Xte = self._transform.transform(self._to_sktime(windows))
        return self._clip(self._ridge.predict(np.asarray(Xte)))


# ---------------------------------------------------------------------------
# catch22 features + GBM (the hand-crafted-indicator foil -- RESEARCH_PLAN §6, RQ-D)
# ---------------------------------------------------------------------------
def catch22_features(windows: np.ndarray) -> np.ndarray:
    """The 22 canonical catch22 features per CHANNEL per window, concatenated across
    channels (Lubba et al. 2019, ``pycatch22``). Shape ``(N, 22*C)`` -- the fixed,
    hand-crafted time-series indicator set the TSFM embedding is judged against.

    ``pycatch22`` is imported lazily inside the loop (a core dep exercised by the
    catch22_gbm test; the mean/NN CPU smoke tests never call this). Degenerate (e.g.
    constant) channels can yield NaN for some features -- LightGBM consumes NaN
    natively, so they are left as-is rather than silently imputed."""
    import pycatch22  # reuse reference impl (Task 2.1); catch22 = Lubba et al. 2019
    N, W, C = windows.shape
    x = np.asarray(windows, np.float64)
    feats = np.empty((N, C * 22), np.float32)
    for i in range(N):
        row: list = []
        for c in range(C):
            row.extend(pycatch22.catch22_all(x[i, :, c].tolist())["values"])
        feats[i] = row
    return feats


class Catch22GBMBaseline(Baseline):
    """catch22 indicators -> LightGBM, the SAME interface as ``GBMBaseline`` but with
    the 22 canonical hand-crafted features per channel in place of the window
    statistics -- the cheap "are hand-crafted indicators enough?" foil (RQ-D)."""
    name = "catch22_gbm"

    def __init__(self, config: Config, seed: int = 0):
        self.config = config
        self.seed = seed
        self._model = None

    def fit(self, train_windows, train_labels, val_windows=None, val_labels=None):
        from lightgbm import LGBMRegressor  # reuse reference impl (Task 2.1)
        Xtr = catch22_features(train_windows)
        self._model = LGBMRegressor(random_state=self.seed, n_estimators=500,
                                    learning_rate=0.05, verbose=-1,
                                    n_jobs=-1)  # all cores (Task 2 perf)
        fit_kw = {}
        if val_windows is not None and len(val_windows):
            fit_kw["eval_set"] = [(catch22_features(val_windows), val_labels)]
        self._model.fit(Xtr, np.asarray(train_labels, np.float64), **fit_kw)
        return self

    def predict(self, windows) -> np.ndarray:
        return self._clip(self._model.predict(catch22_features(windows)))


# ---------------------------------------------------------------------------
# From-scratch NN baselines (torch): 1D-CNN (Li et al. 2018-style) and LSTM
# ---------------------------------------------------------------------------
def _train_torch_regressor(model, config, train_windows, train_labels,
                           val_windows, val_labels, seed, max_epochs, patience):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from .train import set_seed
    from .evaluate import rmse

    set_seed(seed, config.deterministic)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Scale channels on TRAIN windows only (no leakage -- Task 2.4).
    mean, std = data_mod.fit_channel_scaler(train_windows)
    Xtr = data_mod.apply_channel_scaler(train_windows, mean, std)
    ytr = np.asarray(train_labels, np.float32) / config.max_rul  # target in [0,1]

    ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    gen = torch.Generator(); gen.manual_seed(seed)
    loader = DataLoader(ds, batch_size=config.baseline_batch_size, shuffle=True,
                        generator=gen)
    opt = torch.optim.Adam(model.parameters(), lr=config.baseline_lr)
    lossf = torch.nn.MSELoss()

    have_val = val_windows is not None and len(val_windows)
    if have_val:
        Xva = torch.from_numpy(data_mod.apply_channel_scaler(val_windows, mean, std)).to(device)

    best = float("inf"); best_state = None; left = patience
    for _epoch in range(max_epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = lossf(model(xb).squeeze(-1), yb)
            loss.backward(); opt.step()
        if have_val:
            model.eval()
            with torch.no_grad():
                pred = model(Xva).squeeze(-1).cpu().numpy() * config.max_rul
            vr = rmse(np.asarray(val_labels, np.float64), np.clip(pred, 0, config.max_rul))
            if vr < best - 1e-6:
                best = vr; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}; left = patience
            else:
                left -= 1
                if left <= 0:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, mean, std, device


class _CNN1D(nn.Module):
    """Small 1D-CNN over (window, channels); Li et al. 2018-style temporal convs."""
    def __init__(self, n_channels, window_size):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(32, 1)

    def forward(self, x):  # x: (N, W, C) -> conv wants (N, C, W)
        z = self.body(x.transpose(1, 2)).squeeze(-1)
        return self.head(z)


class _LSTMNet(nn.Module):
    def __init__(self, n_channels, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(n_channels, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):  # x: (N, W, C)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])  # last time step


class _TorchBaseline(Baseline):
    def __init__(self, config: Config, seed: int = 0):
        self.config = config
        self.seed = seed
        self._model = None
        self._mean = self._std = None
        self._device = "cpu"

    def _build(self):
        raise NotImplementedError

    def fit(self, train_windows, train_labels, val_windows=None, val_labels=None):
        model = self._build(train_windows.shape[2], train_windows.shape[1])
        self._model, self._mean, self._std, self._device = _train_torch_regressor(
            model, self.config, train_windows, train_labels, val_windows, val_labels,
            self.seed, self.config.baseline_max_epochs,
            self.config.baseline_early_stopping_patience,
        )
        return self

    def predict(self, windows) -> np.ndarray:
        import torch
        Xw = data_mod.apply_channel_scaler(windows, self._mean, self._std)
        self._model.eval()
        with torch.no_grad():
            pred = self._model(torch.from_numpy(Xw).to(self._device)).squeeze(-1).cpu().numpy()
        return self._clip(pred * self.config.max_rul)


class CNNBaseline(_TorchBaseline):
    name = "cnn"
    def _build(self, n_channels, window_size):
        return _CNN1D(n_channels, window_size)


class LSTMBaseline(_TorchBaseline):
    name = "lstm"
    def _build(self, n_channels, window_size):
        return _LSTMNet(n_channels)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
BASELINES = {
    "predict_mean": PredictMeanBaseline,
    "gbm": GBMBaseline,
    "minirocket": MiniRocketBaseline,
    "catch22_gbm": Catch22GBMBaseline,
    "cnn": CNNBaseline,
    "lstm": LSTMBaseline,
}


def make_baseline(name: str, config: Config, seed: int = 0) -> Baseline:
    if name not in BASELINES:
        raise KeyError(f"unknown baseline {name!r}; choices: {sorted(BASELINES)}")
    return BASELINES[name](config, seed)
