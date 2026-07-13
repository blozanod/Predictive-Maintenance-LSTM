"""Sweep + ablation runners (Task 1; RESEARCH_PLAN sec.6).

Two entry points:
  * ``run_ablation``  -- the pre-sweep grid (Task 1.5): full data, MSE, 3 seeds,
    ``tsfm_context_length`` x ``head_features`` (+ the raw-fusion arm and the pooling
    variants at the best cell). Picks the winning (context, features, pooling) cell.
  * ``run_sweep``     -- the full data-fraction x loss x seed sweep at ONE chosen
    config, plus the baselines. Writes ``results_v2.csv`` (both-protocol metrics,
    Task 1.4). Checkpoints every cell and skips completed cells on restart.

The TSFM head consumes ONLY the Stage A cache (pooled embeddings + loc/scale + fixed
raw windows). The whole cache is moved to the GPU once per run and heads train with
on-device tensor slicing (Task 2). Baselines run as native regressors on the cached
raw windows (loss column ``native``); a per-baseline window override (Task 1.5)
re-windows the raw series only for the baselines that need it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .config import Config, CACHE_SCHEMA_VERSION
from . import data as data_mod
from . import train as train_mod
from . import baselines as baselines_mod
from .features import HeadFeatureBuilder, raw_last_cycle
from .evaluate import (
    evaluate_predictions, append_result_row, completed_cells, save_run_metadata,
    archive_results_v1, RESULTS_SCHEMA_VERSION,
)

# Completed-cell keys. The main sweep is one config per file, so the config axes are
# constant and need not key cells; the ablation varies them, so it keys on them.
CELL_KEYS = ["model", "n_units", "seed", "loss"]
ABLATION_KEYS = ["model", "tsfm_context_length", "head_features", "pooling", "seed", "loss"]


# ---------------------------------------------------------------------------
# Shared TSFM-head machinery (cache on device once; per-cell feature assembly)
# ---------------------------------------------------------------------------
def _to_device_cache(cache: dict, device: str) -> dict:
    """Move the cached TSFM signals to ``device`` tensors ONCE (Task 2 Stage B).
    Baselines keep the numpy raw windows."""
    import torch
    t = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=device)
    return {
        "tr_emb": t(cache["train_emb"]),
        "tr_ls": t(cache["train_locscale"]),
        # slice last cycle in numpy first: only (N, C) reaches the GPU, not (N, W, C).
        "tr_raw": t(raw_last_cycle(cache["train_windows"])),
        "tr_y": t(cache["train_labels"]),
        "tr_u": np.asarray(cache["train_units"]),
        "te_emb": t(cache["test_emb"]),
        "te_ls": t(cache["test_locscale"]),
        "te_raw": t(raw_last_cycle(cache["test_windows"])),
        "te_y": np.asarray(cache["test_labels"], np.float64),  # unclipped, for evaluate
        "te_u": np.asarray(cache["test_units"]),
    }


def _fit_predict_tsfm(config: Config, dc: dict, tr_mask: np.ndarray, va_mask: np.ndarray,
                      loss: str, seed: int, device: str,
                      curve_path: Optional[Path] = None) -> np.ndarray:
    """Assemble head features (leakage-safe, per-fraction), train the head, and
    predict RUL on the fixed test set. Returns test predictions (numpy)."""
    import torch
    tr_i = torch.as_tensor(np.where(tr_mask)[0], device=device)
    va_i = torch.as_tensor(np.where(va_mask)[0], device=device)

    builder = HeadFeatureBuilder(config).fit(dc["tr_ls"][tr_i], dc["tr_raw"][tr_i])
    Xtr = builder.transform(dc["tr_emb"][tr_i], dc["tr_ls"][tr_i], dc["tr_raw"][tr_i])
    Xva = builder.transform(dc["tr_emb"][va_i], dc["tr_ls"][va_i], dc["tr_raw"][va_i])
    Xte = builder.transform(dc["te_emb"], dc["te_ls"], dc["te_raw"])
    ytr, yva = dc["tr_y"][tr_i], dc["tr_y"][va_i]

    model, _hist = train_mod.train_head(
        Xtr, ytr, Xva, yva, loss, config, seed=seed, device=device, log_csv_path=curve_path,
    )
    return train_mod.predict_head(model, Xte, loss, config, device=device)


def _row(config: Config, model: str, n_units: int, seed: int, loss: str,
         y_true, y_pred, baseline_window: object = "") -> dict:
    """Assemble one results row: identity + config provenance + both-protocol
    metrics. Keys are in a FIXED order across every caller so the CSV columns stay
    aligned (``append_result_row`` writes the header from the first row). For TSFM
    rows ``baseline_window`` is blank; baselines set it to their window length."""
    metrics = evaluate_predictions(y_true, y_pred, config.max_rul)
    return {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "model": model, "n_units": int(n_units), "seed": int(seed), "loss": loss,
        "dataset": config.dataset, "max_rul": config.max_rul,
        "window_size": config.window_size,
        "tsfm_context_length": config.effective_tsfm_context(),
        "head_features": config.head_features, "pooling": config.pooling,
        "baseline_window": baseline_window,
        **metrics,
    }


# ---------------------------------------------------------------------------
# Baseline windows (default: cache; per-baseline override re-windows raw series)
# ---------------------------------------------------------------------------
def _baseline_window_sets(config: Config, cache: dict,
                          baseline_names: list[str]) -> dict[int, dict]:
    """Map each needed baseline window size -> {train/test windows, labels, units}.

    The base ``window_size`` reuses the cache. Override sizes (Task 1.5) are built
    from the raw C-MAPSS series (loaded once). Padding a longer test window may
    fabricate cycles for the 37/100 FD001 units shorter than it -- a known baseline
    limitation, noted in CHANGES.md; the TSFM path is padding-free."""
    sizes = {config.baseline_windows.get(b, config.window_size) for b in baseline_names}
    out: dict[int, dict] = {}
    base = config.window_size
    out[base] = {
        "tr_w": cache["train_windows"], "tr_y": cache["train_labels"], "tr_u": cache["train_units"],
        "te_w": cache["test_windows"], "te_y": cache["test_labels"], "te_u": cache["test_units"],
    }
    override_sizes = [s for s in sizes if s != base]
    if override_sizes:
        df_train, df_test, rul = data_mod.load_cmapss(config)
        df_train = data_mod.add_train_rul(df_train, config)
        df_test = data_mod.add_test_rul(df_test, rul, config)
        cols = config.sensor_columns
        for s in override_sizes:
            tr_w, tr_y, tr_u = data_mod.make_windows(df_train, cols, s, target_col="clipped_rul")
            te_w, te_y, te_u = data_mod.make_test_last_windows(
                df_test, cols, s, target_col="actual_rul", pad_short=config.pad_short_test_units)
            out[s] = {"tr_w": tr_w, "tr_y": tr_y, "tr_u": tr_u,
                      "te_w": te_w, "te_y": te_y, "te_u": te_u}
    return out


# ---------------------------------------------------------------------------
# Run-dir bookkeeping
# ---------------------------------------------------------------------------
def _save_sampled_units(run_dir: Path, n_units: int, seed: int,
                        sampled: np.ndarray, train_u: np.ndarray, val_u: np.ndarray) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"units_n{n_units}_seed{seed}.json").write_text(json.dumps(
        {"n_units": int(n_units), "seed": int(seed),
         "sampled_units": [int(u) for u in sampled],
         "train_units": [int(u) for u in train_u],
         "val_units": [int(u) for u in val_u]}, indent=2))


# ---------------------------------------------------------------------------
# Main sweep (one chosen config) -> results_v2.csv
# ---------------------------------------------------------------------------
def run_sweep(
    config: Config,
    cache: Optional[dict] = None,
    results_csv: Optional[str | Path] = None,
    run_dir: Optional[str | Path] = None,
    baseline_names: Optional[list[str]] = None,
    losses: Optional[list[str]] = None,
    device: str = "cpu",
) -> Path:
    """Full data-fraction x loss x seed sweep at ``config``, plus baselines. Appends
    to ``results_v2.csv`` (both-protocol metrics). Restartable."""
    from .embeddings import load_embedding_cache  # local import: no embedder needed

    if cache is None:
        cache = load_embedding_cache(config)
    run_dir = Path(run_dir) if run_dir else Path(config.results_dir) / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    archive_results_v1(config.results_dir)  # never overwrite the v1 numbers (Task 1.4)
    results_csv = Path(results_csv) if results_csv else Path(config.results_dir) / "results_v2.csv"
    curves_dir = run_dir / "learning_curves"
    save_run_metadata(config, run_dir / "run_metadata.json")

    losses = losses if losses is not None else config.losses
    if baseline_names is None:
        baseline_names = ["predict_mean", "gbm", "minirocket", "cnn", "lstm"]
    model_tag = config.model_name.split("/")[-1] + "_mlp"

    dc = _to_device_cache(cache, device)
    bwin = _baseline_window_sets(config, cache, baseline_names)
    tr_u = dc["tr_u"]
    all_units = np.unique(tr_u)
    done = completed_cells(results_csv, CELL_KEYS)

    for n_units in config.data_unit_counts:
        if n_units > len(all_units):
            continue
        for seed in config.sweep_seeds:
            sampled = data_mod.subsample_units(all_units, n_units, seed)
            train_u, val_u = data_mod.unit_train_val_split(sampled, config.val_fraction, seed)
            _save_sampled_units(run_dir, n_units, seed, sampled, train_u, val_u)
            tr_mask = np.isin(tr_u, train_u)
            va_mask = np.isin(tr_u, val_u)

            # ---- TSFM MLP head, one arm per loss (cached embeddings only) ----
            for loss in losses:
                key = (model_tag, str(n_units), str(seed), loss)
                if key in done:
                    continue
                curve = curves_dir / f"{model_tag}_n{n_units}_seed{seed}_{loss}.csv"
                pred = _fit_predict_tsfm(config, dc, tr_mask, va_mask, loss, seed, device, curve)
                append_result_row(results_csv,
                                  _row(config, model_tag, n_units, seed, loss, dc["te_y"], pred))
                done.add(key)

            # ---- baselines on cached (or override-windowed) raw windows ----
            for bname in baseline_names:
                key = (bname, str(n_units), str(seed), "native")
                if key in done:
                    continue
                ws = config.baseline_windows.get(bname, config.window_size)
                bd = bwin[ws]
                b_tr_mask = np.isin(bd["tr_u"], train_u)
                b_va_mask = np.isin(bd["tr_u"], val_u)
                bl = baselines_mod.make_baseline(bname, config, seed=seed)
                bl.fit(bd["tr_w"][b_tr_mask], bd["tr_y"][b_tr_mask],
                       bd["tr_w"][b_va_mask], bd["tr_y"][b_va_mask])
                pred = bl.predict(bd["te_w"])
                append_result_row(results_csv,
                                  _row(config, bname, n_units, seed, "native", bd["te_y"], pred,
                                       baseline_window=ws))
                done.add(key)

    return results_csv


# ---------------------------------------------------------------------------
# Ablation (Task 1.5): pick the winning (context, features, pooling) cell
# ---------------------------------------------------------------------------
def run_ablation(
    config: Config,
    device: str = "cpu",
    contexts: Optional[list[int]] = None,
    feature_sets: Optional[list[str]] = None,
    pooling_variants: Optional[list[str]] = None,
    seeds: Optional[list[int]] = None,
    n_units: Optional[int] = None,
    ablation_csv: Optional[str | Path] = None,
    embedder_factory: Optional[Callable[[Config], object]] = None,
) -> Path:
    """Full-data, MSE, ``seeds``-seed ablation. Builds the Stage A cache for each
    (context, pooling) as needed (idempotent), trains the TSFM head over the grid,
    and appends rows to ``ablation.csv``. Restartable via completed-cell detection.

    Grid (Task 1.5): ``contexts`` x ``feature_sets`` at the default pooling, then at
    the best (context, features) cell add the ``emb+locscale+raw`` arm and the
    ``pooling_variants``. ``embedder_factory(cfg)`` lets CPU tests inject a mock; if
    None, Chronos-2 (GPU) is used.
    """
    from .embeddings import build_embedding_cache, load_embedding_cache

    contexts = contexts or [30, 60, 120, 256]
    feature_sets = feature_sets or ["emb", "emb+locscale"]
    pooling_variants = pooling_variants or ["mean", "last_content"]
    seeds = seeds or [0, 1, 2]
    ablation_csv = Path(ablation_csv) if ablation_csv else Path(config.results_dir) / "ablation.csv"
    run_dir = Path(config.results_dir) / "ablation_runs"
    curves_dir = run_dir / "learning_curves"
    run_dir.mkdir(parents=True, exist_ok=True)
    model_tag = config.model_name.split("/")[-1] + "_mlp"
    done = completed_cells(ablation_csv, ABLATION_KEYS)

    def _ensure_cache(cfg: Config) -> dict:
        emb = embedder_factory(cfg) if embedder_factory is not None else None
        build_embedding_cache(cfg, embedder=emb)
        return load_embedding_cache(cfg)

    def _n_units(cache: dict) -> int:
        return n_units if n_units is not None else int(np.unique(cache["train_units"]).size)

    def _run_cell(cfg: Config, cache: dict):
        dc = _to_device_cache(cache, device)
        tr_u = dc["tr_u"]
        all_units = np.unique(tr_u)
        nu = _n_units(cache)
        for seed in seeds:
            key = (model_tag, str(cfg.effective_tsfm_context()), cfg.head_features,
                   cfg.pooling, str(seed), "mse")
            if key in done:
                continue
            sampled = data_mod.subsample_units(all_units, nu, seed)
            train_u, val_u = data_mod.unit_train_val_split(sampled, cfg.val_fraction, seed)
            tr_mask, va_mask = np.isin(tr_u, train_u), np.isin(tr_u, val_u)
            curve = curves_dir / f"c{cfg.effective_tsfm_context()}_{cfg.head_features}_{cfg.pooling}_s{seed}.csv"
            pred = _fit_predict_tsfm(cfg, dc, tr_mask, va_mask, "mse", seed, device, curve)
            append_result_row(ablation_csv, _row(cfg, model_tag, nu, seed, "mse", dc["te_y"], pred))
            done.add(key)

    # ---- Phase 1: contexts x feature_sets at the default pooling ----
    for context in contexts:
        cfg_c = config.replace(tsfm_context_length=context, pooling=config.pooling)
        cache = _ensure_cache(cfg_c)
        for hf in feature_sets:
            _run_cell(cfg_c.replace(head_features=hf), cache)

    # ---- Pick best (context, features) so far, then Phase 2 ----
    best = select_best_ablation_cell(ablation_csv)
    best_context = int(best["tsfm_context_length"])
    best_features = best["head_features"]

    # 2a: raw-fusion arm at the best context (default pooling; cache already built).
    cfg_raw = config.replace(tsfm_context_length=best_context, pooling=config.pooling,
                             head_features="emb+locscale+raw")
    _run_cell(cfg_raw, _ensure_cache(cfg_raw))

    # 2b: pooling variants at the best (context, features) cell.
    for pooling in pooling_variants:
        cfg_p = config.replace(tsfm_context_length=best_context, pooling=pooling,
                               head_features=best_features)
        _run_cell(cfg_p, _ensure_cache(cfg_p))

    return ablation_csv


def select_best_ablation_cell(ablation_csv: str | Path,
                              metric: str = "rmse_clipped") -> dict:
    """Return the ablation cell (dict of axes + mean ``metric``) with the lowest
    seed-mean ``metric`` -- the winning (context, features, pooling). Selection is on
    the literature-comparable clipped RMSE (Task 1.4/1.5)."""
    from .evaluate import load_results
    rows = load_results(ablation_csv)
    if not rows:
        raise ValueError(f"no ablation rows in {ablation_csv}")
    groups: dict[tuple, list[float]] = {}
    for r in rows:
        k = (int(r["tsfm_context_length"]), r["head_features"], r["pooling"])
        groups.setdefault(k, []).append(r[metric])
    best_k, best_vals = min(groups.items(), key=lambda kv: float(np.mean(kv[1])))
    return {
        "tsfm_context_length": best_k[0],
        "head_features": best_k[1],
        "pooling": best_k[2],
        f"mean_{metric}": float(np.mean(best_vals)),
        f"std_{metric}": float(np.std(best_vals)),
        "n_seeds": len(best_vals),
    }


# ---------------------------------------------------------------------------
# Baseline window comparison (Task 1.5): GBM/LSTM at window 30 vs 120, full data
# ---------------------------------------------------------------------------
def run_baseline_window_comparison(
    config: Config,
    windows: Optional[list[int]] = None,
    baseline_names: Optional[list[str]] = None,
    seeds: Optional[list[int]] = None,
    device: str = "cpu",
    out_csv: Optional[str | Path] = None,
) -> Path:
    """Rerun GBM/LSTM at each window in ``windows`` (default {30, 120}) at full data
    and write both-protocol metrics, so the better per-baseline window can be adopted
    via ``config.baseline_windows`` (equal-tuning-budget fairness, RESEARCH_PLAN
    sec.6). Windows are built from the raw series (no embedding needed)."""
    windows = windows or [config.window_size, 120]
    baseline_names = baseline_names or ["gbm", "lstm"]
    seeds = seeds or [0, 1, 2]
    out_csv = Path(out_csv) if out_csv else Path(config.results_dir) / "baseline_window_comparison.csv"

    df_train, df_test, rul = data_mod.load_cmapss(config)
    df_train = data_mod.add_train_rul(df_train, config)
    df_test = data_mod.add_test_rul(df_test, rul, config)
    cols = config.sensor_columns
    done = completed_cells(out_csv, ["model", "baseline_window", "seed"])

    for ws in windows:
        tr_w, tr_y, tr_u = data_mod.make_windows(df_train, cols, ws, target_col="clipped_rul")
        te_w, te_y, te_u = data_mod.make_test_last_windows(
            df_test, cols, ws, target_col="actual_rul", pad_short=config.pad_short_test_units)
        all_units = np.unique(tr_u)
        for seed in seeds:
            train_u, val_u = data_mod.unit_train_val_split(all_units, config.val_fraction, seed)
            tr_mask, va_mask = np.isin(tr_u, train_u), np.isin(tr_u, val_u)
            for bname in baseline_names:
                if (bname, str(ws), str(seed)) in done:
                    continue
                bl = baselines_mod.make_baseline(bname, config, seed=seed)
                bl.fit(tr_w[tr_mask], tr_y[tr_mask], tr_w[va_mask], tr_y[va_mask])
                pred = bl.predict(te_w)
                append_result_row(out_csv, _row(config, bname, len(all_units), seed, "native",
                                                te_y, pred, baseline_window=ws))
                done.add((bname, str(ws), str(seed)))
    return out_csv


# ---------------------------------------------------------------------------
# Fairness arms: the engine-age floor + GBM-with-age (plan §4; CHANGES.md §19)
# ---------------------------------------------------------------------------
def run_fairness_baselines(
    config: Config,
    results_csv: Optional[str | Path] = None,
    n_units_list: Optional[list[int]] = None,
    seeds: Optional[list[int]] = None,
) -> Path:
    """Two arms that hand the baselines the ELAPSED-CYCLES signal the TSFM's
    variable-length context implicitly carries (CHANGES.md §12 caveat 2):

    * ``cycle_reg`` -- linear regression clipped-RUL ~ elapsed cycles, the plan §4
      "linear regression on cycle count" floor. Quantifies how much skill is just
      reading the engine's age.
    * ``gbm_age``   -- the standard GBM whose windows carry ``time_cycles`` as an
      extra leading channel, so ``window_statistics`` includes elapsed cycles
      (last value), its slope, etc. If gbm_age closes the gap to the TSFM, the
      long-context advantage was age, not representation.

    Appends rows to the MAIN results CSV (default ``results_v2.csv``) over the
    standard (n_units x seed) grid so the data-scaling plot picks them up.
    Known caveat (as for all fixed-window baselines, §14): front-padding short
    test units repeats the first cycle's ``time_cycles``; the LAST value -- the
    true age at prediction time -- is always real. Restartable; CPU-only."""
    results_csv = Path(results_csv) if results_csv else Path(config.results_dir) / "results_v2.csv"
    seeds = seeds if seeds is not None else list(config.sweep_seeds)

    df_train, df_test, rul = data_mod.load_cmapss(config)
    df_train = data_mod.add_train_rul(df_train, config)
    df_test = data_mod.add_test_rul(df_test, rul, config)
    ws = config.window_size
    age_cols = ["time_cycles"] + list(config.sensor_columns)
    tr_w, tr_y, tr_u = data_mod.make_windows(df_train, age_cols, ws, target_col="clipped_rul")
    te_w, te_y, te_u = data_mod.make_test_last_windows(
        df_test, age_cols, ws, target_col="actual_rul", pad_short=config.pad_short_test_units)
    tr_age, te_age = tr_w[:, -1, 0], te_w[:, -1, 0]   # elapsed cycles at prediction time

    all_units = np.unique(tr_u)
    n_units_list = n_units_list if n_units_list is not None else list(config.data_unit_counts)
    done = completed_cells(results_csv, CELL_KEYS)

    for n_units in n_units_list:
        if n_units > len(all_units):
            continue
        for seed in seeds:
            sampled = data_mod.subsample_units(all_units, n_units, seed)
            train_u, val_u = data_mod.unit_train_val_split(sampled, config.val_fraction, seed)
            tr_mask, va_mask = np.isin(tr_u, train_u), np.isin(tr_u, val_u)

            if ("cycle_reg", str(n_units), str(seed), "native") not in done:
                slope, intercept = np.polyfit(tr_age[tr_mask], tr_y[tr_mask], 1)
                pred = np.clip(slope * te_age + intercept, 0.0, float(config.max_rul))
                append_result_row(results_csv, _row(config, "cycle_reg", len(sampled), seed,
                                                    "native", te_y, pred, baseline_window=ws))
                done.add(("cycle_reg", str(n_units), str(seed), "native"))

            if ("gbm_age", str(n_units), str(seed), "native") not in done:
                bl = baselines_mod.make_baseline("gbm", config, seed=seed)
                bl.fit(tr_w[tr_mask], tr_y[tr_mask], tr_w[va_mask], tr_y[va_mask])
                pred = bl.predict(te_w)
                append_result_row(results_csv, _row(config, "gbm_age", len(sampled), seed,
                                                    "native", te_y, pred, baseline_window=ws))
                done.add(("gbm_age", str(n_units), str(seed), "native"))
    return results_csv
