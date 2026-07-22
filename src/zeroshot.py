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

from .config import Config
from . import data as data_mod
from .evaluate import (
    evaluate_predictions, append_result_row, completed_cells, save_run_metadata,
    RESULTS_SCHEMA_VERSION,
)

# A zero-shot run has no unit-count / seed training axis; one row per model.
ZEROSHOT_KEYS = ["model", "dataset", "loss"]


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
def _zeroshot_row(config: Config, model: str, loss: str, y_true, y_pred) -> dict:
    return {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "model": model, "n_units": 0, "seed": 0, "loss": loss,
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
    out_csv: Optional[str | Path] = None,
) -> Path:
    """Zero-shot RUL via health-index threshold crossing, scored against the
    ``predict_mean`` and ``cycle_reg`` floors. Writes ``zeroshot.csv`` (one row per
    model). ``forecaster_factory(config)`` injects a CPU mock; ``horizon`` defaults to
    ``max_rul``. Restartable: models already present are skipped."""
    out_csv = Path(out_csv) if out_csv else config.results_path("zeroshot.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    run_dir = config.results_path("zeroshot_runs")
    save_run_metadata(config, run_dir / "run_metadata.json")
    horizon = int(horizon) if horizon is not None else int(config.max_rul)
    cols = config.sensor_columns
    model_tag = config.model_name.split("/")[-1] + "_zeroshot"
    done = completed_cells(out_csv, ZEROSHOT_KEYS)

    df_train, df_test = data_mod.load_prepared(config)

    # ---- health index transform + failure threshold (train run-to-failure) ----
    transform = build_health_index(df_train[cols].to_numpy(np.float64),
                                   df_train["time_cycles"].to_numpy(np.float64))
    last_idx = [health_index(g[cols].to_numpy(np.float64), transform)[-1]
                for _u, g in df_train.groupby("unit_number", sort=True)]
    threshold = float(np.quantile(np.asarray(last_idx, np.float64), threshold_quantile))

    # ---- zero-shot predictions per test unit ----
    if (model_tag, config.dataset, "zeroshot") not in done:
        forecaster = (forecaster_factory(config) if forecaster_factory is not None
                      else ChronosForecaster(config, device))
        units, preds, truths = [], [], []
        for uid, g in df_test.groupby("unit_number", sort=True):
            series = health_index(g[cols].to_numpy(np.float64), transform)
            preds.append(threshold_crossing_rul(series, forecaster, threshold, horizon))
            truths.append(float(g["actual_rul"].to_numpy()[-1]))
            units.append(int(uid))
        preds = np.clip(np.asarray(preds, np.float64), 0.0, float(config.max_rul))
        truths = np.asarray(truths, np.float64)
        append_result_row(out_csv, _zeroshot_row(config, model_tag, "zeroshot", truths, preds))
        done.add((model_tag, config.dataset, "zeroshot"))

    # ---- floors: predict-mean + cycle-count linear regression (no failures) ----
    tr_last = df_train.groupby("unit_number").tail(1)
    mean_rul = float(df_train["clipped_rul"].mean())
    slope, intercept = np.polyfit(df_train["time_cycles"].to_numpy(np.float64),
                                  df_train["clipped_rul"].to_numpy(np.float64), 1)
    te_last = df_test.groupby("unit_number", sort=True).tail(1)
    te_truth = te_last["actual_rul"].to_numpy(np.float64)
    te_time = te_last["time_cycles"].to_numpy(np.float64)
    _ = tr_last  # (kept for symmetry / provenance; floors are fit on all train rows)

    floors = {
        "predict_mean": np.full(len(te_truth), mean_rul),
        "cycle_reg": np.clip(slope * te_time + intercept, 0.0, float(config.max_rul)),
    }
    for name, pred in floors.items():
        if (name, config.dataset, "native") in done:
            continue
        append_result_row(out_csv, _zeroshot_row(config, name, "native", te_truth,
                                                 np.clip(pred, 0.0, float(config.max_rul))))
        done.add((name, config.dataset, "native"))
    return out_csv
