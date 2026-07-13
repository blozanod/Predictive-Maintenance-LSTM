"""Horizon-stratified evaluation: how does error depend on how FAR from failure
the prediction is made?

The main protocol scores ONE prediction per test unit (its final observed cycle),
so far-from-failure behavior -- the predictions that buy planning time -- is
invisible in results_v2.csv. This module evaluates EVERY test cycle:

  Stage A-H  ``build_horizon_cache``: one variable-length context per test cycle
             (the exact training-row construction, applied to the test
             trajectories), embedded once and cached SEPARATELY from the main
             Stage A cache (which stays valid).
  Stage B-H  ``run_horizon_eval``: train heads/baselines on the standard train
             cache at chosen unit counts, predict every test cycle, and stratify
             metrics by the UNCLIPPED true RUL (``horizon.csv``), with per-cycle
             predictions saved for trajectory plots (``horizon_predictions.csv``).

Reading the output (IMPORTANT, Task 2.5 honesty rule):
  * Bins BELOW ``max_rul`` measure genuine accuracy at that horizon.
  * The ``>= max_rul`` bin measures SATURATION QUALITY only -- training labels are
    clipped at ``max_rul`` (125), so no model here can express "fails in 180
    cycles"; the correct clipped answer while healthy is exactly the cap. True
    longer-horizon skill needs a ``max_rul`` raise, which re-keys the caches
    (labels are cached with the windows) -- recorded as follow-up, CHANGES.md §16.
  * Test rows exist only for cycles >= ``window_size`` (same as training rows);
    test units shorter than ``window_size`` are excluded (none in FD001).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .config import Config
from . import data as data_mod
from . import train as train_mod
from . import baselines as baselines_mod
from .features import HeadFeatureBuilder, raw_last_cycle
from .evaluate import (
    rmse, mae, nasa_score, append_result_row, completed_cells, save_run_metadata,
    RESULTS_SCHEMA_VERSION,
)

HORIZON_KEYS = ["model", "n_units", "seed", "loss"]
DEFAULT_BIN_EDGES = (0.0, 25.0, 50.0, 75.0, 100.0, 125.0, float("inf"))


def horizon_cache_path(config: Config) -> Path:
    """Sidecar cache next to the main Stage A cache; keyed identically so it
    invalidates with it, but separate so building it never touches the main cache."""
    return Path(config.cache_dir) / f"horizon_{config.embedding_cache_key()}.npz"


# ---------------------------------------------------------------------------
# Stage A-H: embed every test cycle (idempotent)
# ---------------------------------------------------------------------------
def build_horizon_cache(
    config: Config,
    embedder=None,
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """Embed one variable-length context per TEST cycle >= window_size. Labels are
    the UNCLIPPED per-cycle ``actual_rul`` (provided RUL + cycles still to run);
    fixed windows are co-cached for the baselines."""
    cache_path = horizon_cache_path(config)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not overwrite:
        return cache_path

    _, df_test, rul_truth = data_mod.load_cmapss(config)
    df_test = data_mod.add_test_rul(df_test, rul_truth, config)
    ws, tsfm_ctx, cols = config.window_size, config.effective_tsfm_context(), config.sensor_columns

    te_w, te_y, te_u = data_mod.make_windows(df_test, cols, ws, target_col="actual_rul")
    te_ctx, te_y2, te_u2 = data_mod.make_windows_varlen(
        df_test, cols, ws, tsfm_ctx, target_col="actual_rul")
    assert np.array_equal(te_u, te_u2) and np.allclose(te_y, te_y2), "horizon varlen misaligned"

    if embedder is None:
        from .embeddings import ChronosEmbedder
        embedder = ChronosEmbedder(config)
    te_emb, te_ls = embedder.embed_windows(te_ctx)
    if verbose and getattr(embedder, "last_throughput", None):
        print(f"[Stage A-H] test-all-cycles embed throughput: "
              f"{embedder.last_throughput:.1f} windows/s ({len(te_ctx)} windows)")

    store_dtype = np.dtype(config.embedding_storage_dtype)
    saver = np.savez_compressed if config.cache_compressed else np.savez
    saver(cache_path,
          test_windows=te_w.astype(np.float32),
          test_labels=te_y.astype(np.float32),   # UNCLIPPED actual RUL per cycle
          test_units=te_u,
          test_emb=te_emb.astype(store_dtype),
          test_locscale=te_ls.astype(np.float32))
    cache_path.with_suffix(".json").write_text(json.dumps(
        {"embedding_key_fields": config._embedding_key_fields(),
         "embedder": embedder.describe(), "n_test_cycle_windows": int(te_w.shape[0])},
        indent=2, sort_keys=True))
    return cache_path


def load_horizon_cache(config: Config) -> dict:
    cache_path = horizon_cache_path(config)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Horizon cache {cache_path} not found. Run build_horizon_cache first.")
    with np.load(cache_path) as npz:
        out = {k: npz[k] for k in npz.files}
    out["test_emb"] = out["test_emb"].astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Per-bin metrics
# ---------------------------------------------------------------------------
def horizon_bin_rows(y_true_unclipped, y_pred, max_rul: float,
                     bin_edges: Sequence[float] = DEFAULT_BIN_EDGES) -> list[dict]:
    """Stratify error by UNCLIPPED true RUL. Metrics are against the CLIPPED truth
    (the trainable target); ``bias`` = mean(pred - clipped_true), so bias < 0 means
    the model predicts failure EARLIER than truth at that horizon (conservative).
    ``nasa_mean`` is the per-cycle mean of the PHM08 score (the raw score is a sum,
    so it is not comparable across bins of different size). Includes an ``all`` row
    (bin_lo='all')."""
    y_true = np.asarray(y_true_unclipped, np.float64)
    y_pred = np.asarray(y_pred, np.float64)
    y_clip = np.clip(y_true, None, float(max_rul))
    rows = []

    def _one(mask, lo, hi):
        n = int(mask.sum())
        if n == 0:
            return None
        yt, yp = y_clip[mask], y_pred[mask]
        return {"bin_lo": lo, "bin_hi": hi, "n_bin": n,
                "rmse_clipped": rmse(yt, yp), "mae_clipped": mae(yt, yp),
                "bias": float(np.mean(yp - yt)),
                "nasa_mean": nasa_score(yt, yp) / n}

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        r = _one((y_true >= lo) & (y_true < hi), lo, "inf" if np.isinf(hi) else hi)
        if r is not None:
            rows.append(r)
    rows.append(_one(np.ones_like(y_true, dtype=bool), "all", "all"))
    return rows


# ---------------------------------------------------------------------------
# Stage B-H: train at chosen unit counts, evaluate every test cycle
# ---------------------------------------------------------------------------
def run_horizon_eval(
    config: Config,
    cache: Optional[dict] = None,           # main Stage A cache (train side)
    hcache: Optional[dict] = None,          # horizon cache (test all-cycles side)
    n_units_list: Optional[list[int]] = None,
    seeds: Optional[list[int]] = None,
    losses: Optional[list[str]] = None,
    baseline_names: Optional[list[str]] = None,
    bin_edges: Sequence[float] = DEFAULT_BIN_EDGES,
    device: str = "cpu",
    out_csv: Optional[str | Path] = None,
    preds_csv: Optional[str | Path] = None,
) -> Path:
    """Train (TSFM head per loss + baselines) at each unit count x seed on the
    STANDARD train cache, predict EVERY test cycle, and append per-RUL-bin metric
    rows to ``horizon.csv`` + per-cycle predictions to ``horizon_predictions.csv``.
    Restartable: completed (model, n_units, seed, loss) cells are skipped."""
    import torch
    from .embeddings import load_embedding_cache
    from .sweep import _to_device_cache

    if cache is None:
        cache = load_embedding_cache(config)
    if hcache is None:
        hcache = load_horizon_cache(config)
    out_csv = Path(out_csv) if out_csv else Path(config.results_dir) / "horizon.csv"
    preds_csv = Path(preds_csv) if preds_csv else Path(config.results_dir) / "horizon_predictions.csv"
    run_dir = Path(config.results_dir) / "horizon_runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_run_metadata(config, run_dir / "run_metadata.json")

    seeds = seeds if seeds is not None else config.sweep_seeds[:3]
    losses = losses if losses is not None else list(config.losses)
    baseline_names = baseline_names if baseline_names is not None else ["gbm", "lstm"]
    model_tag = config.model_name.split("/")[-1] + "_mlp"

    dc = _to_device_cache(cache, device)     # train side on device
    t = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=device)
    h_emb, h_ls = t(hcache["test_emb"]), t(hcache["test_locscale"])
    h_raw = t(raw_last_cycle(hcache["test_windows"]))
    h_y = np.asarray(hcache["test_labels"], np.float64)      # unclipped, per cycle
    h_u = np.asarray(hcache["test_units"])

    all_units = np.unique(dc["tr_u"])
    n_units_list = n_units_list if n_units_list is not None else [len(all_units)]
    done = completed_cells(out_csv, HORIZON_KEYS)

    def _emit(model_name: str, n_units: int, seed: int, loss: str, pred: np.ndarray):
        for bin_row in horizon_bin_rows(h_y, pred, config.max_rul, bin_edges):
            append_result_row(out_csv, {
                "schema_version": RESULTS_SCHEMA_VERSION,
                "model": model_name, "n_units": int(n_units), "seed": int(seed),
                "loss": loss, "dataset": config.dataset, "max_rul": config.max_rul,
                "window_size": config.window_size,
                "tsfm_context_length": config.effective_tsfm_context(),
                "head_features": config.head_features, "pooling": config.pooling,
                **bin_row,
            })
        for unit, yt, yp in zip(h_u, h_y, pred):
            append_result_row(preds_csv, {
                "model": model_name, "n_units": int(n_units), "seed": int(seed),
                "loss": loss, "unit": int(unit),
                "true_rul": float(yt), "pred": float(yp)})

    for n_units in n_units_list:
        if n_units > len(all_units):
            continue
        for seed in seeds:
            sampled = data_mod.subsample_units(all_units, n_units, seed)
            train_u, val_u = data_mod.unit_train_val_split(sampled, config.val_fraction, seed)
            tr_mask, va_mask = np.isin(dc["tr_u"], train_u), np.isin(dc["tr_u"], val_u)
            tr_i = torch.as_tensor(np.where(tr_mask)[0], device=device)
            va_i = torch.as_tensor(np.where(va_mask)[0], device=device)

            # ---- TSFM head per loss (features assembled leakage-safe per cell) ----
            builder = HeadFeatureBuilder(config).fit(dc["tr_ls"][tr_i], dc["tr_raw"][tr_i])
            Xtr = builder.transform(dc["tr_emb"][tr_i], dc["tr_ls"][tr_i], dc["tr_raw"][tr_i])
            Xva = builder.transform(dc["tr_emb"][va_i], dc["tr_ls"][va_i], dc["tr_raw"][va_i])
            Xh = builder.transform(h_emb, h_ls, h_raw)
            for loss in losses:
                if (model_tag, str(n_units), str(seed), loss) in done:
                    continue
                model, _ = train_mod.train_head(
                    Xtr, dc["tr_y"][tr_i], Xva, dc["tr_y"][va_i], loss, config,
                    seed=seed, device=device)
                pred = train_mod.predict_head(model, Xh, loss, config, device=device)
                _emit(model_tag, n_units, seed, loss, pred)
                done.add((model_tag, str(n_units), str(seed), loss))

            # ---- baselines on the fixed windows (window_size only) ----
            tr_w, tr_y = cache["train_windows"], cache["train_labels"]
            for bname in baseline_names:
                if (bname, str(n_units), str(seed), "native") in done:
                    continue
                bl = baselines_mod.make_baseline(bname, config, seed=seed)
                bl.fit(tr_w[tr_mask], tr_y[tr_mask], tr_w[va_mask], tr_y[va_mask])
                pred = bl.predict(hcache["test_windows"])
                _emit(bname, n_units, seed, "native", pred)
                done.add((bname, str(n_units), str(seed), "native"))
    return out_csv
