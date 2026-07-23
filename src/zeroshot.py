"""Zero-shot health-index forecasting -- the 0-failures endpoint of RQ-B (RQ-Z; §39).

The strongest possible answer to "how many failures before deploying": **none**. No
head, no training. We build a scalar HEALTH INDEX from the sensors, calibrate a failure
threshold from run-to-failure history (labels of *other* units, never the test unit's),
forecast the index forward with a TSFM's native forecasting mode, and read the
predicted RUL off the horizon at which the forecast crosses the threshold.

Pipeline (each step a documented judgment call):
  * **Health index** -- DECISION (uncited): the first principal component of the
    z-standardized sensor channels (unsupervised; fit on train sensor rows only, no RUL
    labels used), oriented so it INCREASES toward failure (sign flipped to correlate
    positively with elapsed cycles). One monotone degradation scalar per cycle.
  * **Failure threshold** -- DECISION (uncited): the ``threshold_quantile`` (default
    median) of the health index at each train unit's LAST (failure) cycle. This uses
    only the fleet's run-to-failure endpoints, not the censored test units.
  * **Multiple seeds (bootstrap)** -- the pipeline is deterministic given its
    calibration set, so each seed resamples that set (train units, with replacement)
    before fitting the transform / threshold / floors. The reported seed-mean is then
    robust to a single lucky/unlucky draw of observed failures rather than one point
    estimate; the win-rule's paired-seed test becomes meaningful (>= 2 seeds).
  * **Forecast + crossing** -- the forecaster predicts the index ``horizon`` steps
    ahead from each test unit's observed trajectory; predicted RUL = the first step at
    which the forecast reaches the threshold (0 if already past it, ``horizon`` if it
    never crosses), clipped to ``[0, max_rul]``.

Scored with the standard both-protocol metrics against the ``predict_mean`` and
``cycle_reg`` floors (RESEARCH_PLAN §8) -- the fair "no-training" comparison.

The forecaster is injected via ``forecaster_factory`` (the CPU-test seam, mirroring
``embedder_factory``): any object with ``forecast(series_1d, horizon) -> (horizon,)``.
The default is a Chronos-2 forecaster whose backbone load/call is the sanctioned
``# pragma: no cover`` boundary; tests pass a mock returning a fixed trajectory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .config import Config
from . import data as data_mod
from .evaluate import (
    evaluate_predictions, append_result_row, completed_cells, save_run_metadata,
    RESULTS_SCHEMA_VERSION,
)

# One row per (model, seed): the arm has no training axis, but it IS run over multiple
# seeds -- each seed BOOTSTRAPS the observed-failure calibration set (which historical
# failures set the health-index transform + threshold + floors), so the reported
# seed-mean is robust to a single lucky/unlucky calibration draw rather than betting on
# one. n_units stays 0 (the 0-target-failures endpoint of RQ-B).
ZEROSHOT_KEYS = ["model", "dataset", "seed", "loss"]


# ---------------------------------------------------------------------------
# Health index (unsupervised: no RUL labels) + threshold calibration
# ---------------------------------------------------------------------------
def build_health_index(train_X: np.ndarray, train_time: np.ndarray) -> dict:
    """Fit the health-index transform on train sensor rows: per-channel standardization
    + the first principal component, oriented to increase with elapsed cycles. Returns
    ``{mean, std, direction}`` for ``health_index`` to apply. Uses no RUL labels."""
    X = np.asarray(train_X, np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    Z = (X - mean) / std
    # first principal component via SVD of the centered standardized matrix
    _u, _s, vt = np.linalg.svd(Z - Z.mean(axis=0), full_matrices=False)
    direction = vt[0]
    idx = Z @ direction
    # orient toward failure: index should increase with elapsed cycles
    t = np.asarray(train_time, np.float64)
    if np.std(idx) > 0 and np.std(t) > 0 and np.corrcoef(idx, t)[0, 1] < 0:
        direction = -direction
    return {"mean": mean, "std": std, "direction": direction}


def health_index(X: np.ndarray, transform: dict) -> np.ndarray:
    """Apply a fitted ``build_health_index`` transform to sensor rows ``(n, C)`` ->
    ``(n,)`` health-index trajectory."""
    Z = (np.asarray(X, np.float64) - transform["mean"]) / transform["std"]
    return Z @ transform["direction"]


def threshold_crossing_rul(series: np.ndarray, forecaster, threshold: float,
                           horizon: int) -> float:
    """Predicted RUL from a health-index trajectory: 0 if the last observed value is
    already at/above ``threshold``; else forecast ``horizon`` steps and return the first
    (1-based) step that reaches ``threshold``, or ``horizon`` if it never does."""
    series = np.asarray(series, np.float64)
    if series[-1] >= threshold:
        return 0.0
    forecast = np.asarray(forecaster.forecast(series, horizon), np.float64)
    crossings = np.where(forecast >= threshold)[0]
    return float(crossings[0] + 1) if crossings.size else float(horizon)


# ---------------------------------------------------------------------------
# Default (real) forecaster -- backbone load/call is the pragma boundary
# ---------------------------------------------------------------------------
class ChronosForecaster:
    """Chronos-2 native forecasting wrapper (the default zero-shot forecaster). Only the
    backbone import + ``predict`` call are GPU-only; instantiation is cheap."""

    def __init__(self, config: Config, device: Optional[str] = None):
        self.config = config
        self.model_name = config.model_name
        self._device = device
        self._pipeline = None

    def forecast(self, series, horizon: int):  # pragma: no cover -- GPU-only backbone
        import torch
        from chronos import Chronos2Pipeline
        if self._pipeline is None:
            device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._pipeline = Chronos2Pipeline.from_pretrained(self.model_name,
                                                              device_map=device)
        ctx = [np.asarray(series, np.float32)[None, :]]
        with torch.inference_mode():
            quantiles, mean = self._pipeline.predict(ctx, prediction_length=int(horizon))
        return np.asarray(mean[0], np.float64).reshape(-1)[:horizon]


# ---------------------------------------------------------------------------
# The zero-shot run
# ---------------------------------------------------------------------------
def _zeroshot_row(config: Config, model: str, loss: str, seed: int, y_true, y_pred) -> dict:
    return {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "model": model, "n_units": 0, "seed": int(seed), "loss": loss,
        "dataset": config.dataset, "max_rul": config.max_rul,
        "window_size": config.window_size,
        "tsfm_context_length": config.effective_tsfm_context(),
        "head_features": config.head_features, "pooling": config.pooling,
        "baseline_window": "",
        **evaluate_predictions(y_true, y_pred, config.max_rul),
    }


def run_zeroshot(
    config: Config,
    device: str = "cpu",
    forecaster_factory: Optional[Callable[[Config], object]] = None,
    horizon: Optional[int] = None,
    threshold_quantile: float = 0.5,
    seeds: Optional[list[int]] = None,
    out_csv: Optional[str | Path] = None,
) -> Path:
    """Zero-shot RUL via health-index threshold crossing, scored against the
    ``predict_mean`` and ``cycle_reg`` floors. Writes ``zeroshot.csv`` (one row per
    ``(model, seed)``).

    Multiple seeds (default ``config.sweep_seeds``): the method is deterministic given
    its calibration set, so each seed BOOTSTRAPS that set -- it resamples, with
    replacement, which train run-to-failure units set the (unsupervised) health-index
    transform, the failure threshold, and the two floors. Reporting the seed-mean over
    these draws is robust to a single lucky/unlucky calibration sample (the health index
    still uses NO RUL labels). The forecaster is deterministic, but each seed's transform
    + threshold differ, so predictions genuinely vary across seeds.

    ``forecaster_factory(config)`` injects a CPU mock; ``horizon`` defaults to
    ``max_rul``. Restartable: completed ``(model, seed)`` cells are skipped."""
    out_csv = Path(out_csv) if out_csv else config.results_path("zeroshot.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    run_dir = config.results_path("zeroshot_runs")
    save_run_metadata(config, run_dir / "run_metadata.json")
    horizon = int(horizon) if horizon is not None else int(config.max_rul)
    cols = config.sensor_columns
    model_tag = config.model_name.split("/")[-1] + "_zeroshot"
    seeds = list(seeds) if seeds is not None else list(config.sweep_seeds)
    done = completed_cells(out_csv, ZEROSHOT_KEYS)

    df_train, df_test = data_mod.load_prepared(config)
    train_units = np.unique(df_train["unit_number"].to_numpy())
    # Group each train unit's rows ONCE; the per-seed bootstrap indexes into this.
    train_groups = {int(u): g for u, g in df_train.groupby("unit_number", sort=True)}
    # The test units are fixed across seeds: their endpoint truth + last cycle.
    test_items = [(int(u), g) for u, g in df_test.groupby("unit_number", sort=True)]
    te_truth = np.asarray([float(g["actual_rul"].to_numpy()[-1]) for _u, g in test_items],
                          np.float64)
    te_time = np.asarray([float(g["time_cycles"].to_numpy()[-1]) for _u, g in test_items],
                         np.float64)

    forecaster = None  # built once, on first seed that needs a zero-shot prediction
    for seed in seeds:
        # Bootstrap the calibration set: which observed failures we happen to have.
        rng = np.random.default_rng(seed)
        drawn = rng.choice(train_units, size=train_units.size, replace=True)
        cal_frames = [train_groups[int(u)] for u in drawn]
        cal = pd.concat(cal_frames, ignore_index=True)

        # Unsupervised health index + failure threshold, fit on THIS draw (no RUL labels).
        transform = build_health_index(cal[cols].to_numpy(np.float64),
                                       cal["time_cycles"].to_numpy(np.float64))
        last_idx = [health_index(g[cols].to_numpy(np.float64), transform)[-1]
                    for g in cal_frames]
        threshold = float(np.quantile(np.asarray(last_idx, np.float64), threshold_quantile))
        # Floors fit on the SAME draw (matched seeds for the paired win-rule test).
        mean_rul = float(cal["clipped_rul"].mean())
        slope, intercept = np.polyfit(cal["time_cycles"].to_numpy(np.float64),
                                      cal["clipped_rul"].to_numpy(np.float64), 1)

        # ---- zero-shot predictions per test unit (this seed's transform + threshold) ----
        if (model_tag, config.dataset, str(seed), "zeroshot") not in done:
            if forecaster is None:
                forecaster = (forecaster_factory(config) if forecaster_factory is not None
                              else ChronosForecaster(config, device))
            preds = [threshold_crossing_rul(
                        health_index(g[cols].to_numpy(np.float64), transform),
                        forecaster, threshold, horizon)
                     for _u, g in test_items]
            preds = np.clip(np.asarray(preds, np.float64), 0.0, float(config.max_rul))
            append_result_row(out_csv, _zeroshot_row(config, model_tag, "zeroshot", seed,
                                                     te_truth, preds))
            done.add((model_tag, config.dataset, str(seed), "zeroshot"))

        # ---- floors: predict-mean + cycle-count linear regression (this draw) ----
        floors = {
            "predict_mean": np.full(len(te_truth), mean_rul),
            "cycle_reg": np.clip(slope * te_time + intercept, 0.0, float(config.max_rul)),
        }
        for name, pred in floors.items():
            if (name, config.dataset, str(seed), "native") in done:
                continue
            append_result_row(out_csv, _zeroshot_row(config, name, "native", seed, te_truth,
                                                     np.clip(pred, 0.0, float(config.max_rul))))
            done.add((name, config.dataset, str(seed), "native"))
    return out_csv
