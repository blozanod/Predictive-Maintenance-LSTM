"""Cold-start transfer: train the RUL head on a SOURCE fleet, deploy on a TARGET
fleet with 0..k recorded failures.

Deployment question: a new plant has NO failure history (the usual answer is an
anomaly detector). The TSFM backbone is generic; only the small head is task- and
fleet-specific. Can a head trained on an analogous fleet produce useful RUL from
day one, and how fast do a few target failures close the remaining gap?

Three arms per shot count k (``transfer.csv`` column ``mode``):
  * ``zero_shot``       -- head trained on ALL source units, evaluated unchanged on
                           the target test set (k is 0 by definition).
  * ``target_only``     -- head trained on the k sampled target units only (the
                           from-zero curve a plant could build after k failures).
  * ``source+target``   -- head trained on all source units PLUS the k target
                           units (does prior fleet data still help once local
                           failures exist?).

Honesty rules:
  * ALL preprocessing statistics travel with the training data: the head-feature
    standardizer (loc/scale, raw-last) is fit on the arm's TRAIN rows only --
    source rows for zero_shot, so the target is scored under source statistics,
    exactly as a day-one deployment would be.
  * The TSFM path has no cross-unit scaler (Chronos-2 instance-norm, CHANGES.md
    §2), so transfer needs no scaler surgery; GBM's window-statistics features are
    likewise scaler-free. From-scratch NN baselines are NOT included by default --
    they would need a fitted scaler policy decision; add deliberately if needed.
  * Datasets must share ``sensor_columns`` and ``window_size`` (asserted). The
    default FD001<->FD003 pair is valid a-priori: both single-operating-condition,
    same non-constant sensor set. FD002/FD004 are supported through condition-wise
    normalization (CHANGES.md §21), fit PER DATASET on its own train split --
    legitimate for day-one deployment because condition statistics need no failure
    labels, only operating sensor data the target plant already has. A warning is
    printed only if a multi-condition dataset is run with the normalization
    explicitly disabled.
  * Shot counts must be >= 2 (a k=1 arm has no unit left for the validation split).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .config import Config, MULTI_CONDITION_DATASETS
from . import data as data_mod
from . import train as train_mod
from . import baselines as baselines_mod
from .features import HeadFeatureBuilder
from .evaluate import (
    evaluate_predictions, append_result_row, completed_cells, save_run_metadata,
    RESULTS_SCHEMA_VERSION,
)

TRANSFER_KEYS = ["model", "mode", "source_dataset", "target_dataset",
                 "n_target_units", "seed", "loss"]  # pair in the key: multi-pair CSVs (§21)


def _transfer_row(config: Config, model: str, mode: str, source: str, target: str,
                  n_target_units: int, seed: int, loss: str, y_true, y_pred) -> dict:
    return {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "model": model, "mode": mode,
        "source_dataset": source, "target_dataset": target,
        "n_target_units": int(n_target_units), "seed": int(seed), "loss": loss,
        "max_rul": config.max_rul, "window_size": config.window_size,
        "tsfm_context_length": config.effective_tsfm_context(),
        "head_features": config.head_features, "pooling": config.pooling,
        **evaluate_predictions(y_true, y_pred, config.max_rul),
    }


def run_transfer_eval(
    config: Config,
    source_dataset: str = "FD001",
    target_dataset: str = "FD003",
    shots: Optional[list[int]] = None,
    seeds: Optional[list[int]] = None,
    losses: Optional[list[str]] = None,
    baseline_names: Optional[list[str]] = None,
    embedder_factory: Optional[Callable[[Config], object]] = None,
    device: str = "cpu",
    out_csv: Optional[str | Path] = None,
) -> Path:
    """Zero-shot + few-shot transfer sweep; appends to ``transfer.csv``.

    Builds/loads BOTH datasets' Stage A caches at ``config``'s embedding settings
    (idempotent; needs the GPU embedder for a first-time target cache, injectable
    via ``embedder_factory`` for CPU tests). Evaluation is the standard last-cycle
    test protocol of the TARGET dataset, both label protocols. Restartable."""
    import torch
    from .embeddings import build_embedding_cache, load_embedding_cache
    from .sweep import _to_device_cache

    shots = shots if shots is not None else [2, 5, 10, 25]
    if any(k < 2 for k in shots):
        raise ValueError("shots must be >= 2 (k=1 leaves no unit for the val split)")
    seeds = seeds if seeds is not None else config.sweep_seeds[:3]
    losses = losses if losses is not None else ["mse"]
    baseline_names = baseline_names if baseline_names is not None else ["gbm"]
    out_csv = Path(out_csv) if out_csv else Path(config.results_dir) / "transfer.csv"
    run_dir = Path(config.results_dir) / "transfer_runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    src_cfg = config.replace(dataset=source_dataset)
    tgt_cfg = config.replace(dataset=target_dataset)
    for cfg in (src_cfg, tgt_cfg):
        if cfg.dataset in MULTI_CONDITION_DATASETS and not cfg.effective_condition_norm():
            print(f"WARNING: {cfg.dataset} has multiple operating conditions but "
                  f"condition-wise normalization is explicitly disabled -- treat "
                  f"these transfer numbers as exploratory only.")
    save_run_metadata(tgt_cfg, run_dir / "run_metadata.json")
    for cfg in (src_cfg, tgt_cfg):
        emb = embedder_factory(cfg) if embedder_factory is not None else None
        build_embedding_cache(cfg, embedder=emb)  # idempotent
    src = _to_device_cache(load_embedding_cache(src_cfg), device)
    tgt = _to_device_cache(load_embedding_cache(tgt_cfg), device)
    assert src["tr_emb"].shape[1] == tgt["tr_emb"].shape[1], "embedding dims differ"
    assert src["tr_raw"].shape[1] == tgt["tr_raw"].shape[1], "sensor channels differ"

    model_tag = config.model_name.split("/")[-1] + "_mlp"
    src_units = np.unique(src["tr_u"])
    tgt_units = np.unique(tgt["tr_u"])
    done = completed_cells(out_csv, TRANSFER_KEYS)

    def _train_predict_head(loss: str, seed: int, parts: list[tuple[dict, object, object]]):
        """``parts``: list of (device_cache, train_idx, val_idx). Fits the feature
        standardizer on the CONCATENATED train rows of the arm, trains, and
        predicts the TARGET test set."""
        cat = torch.cat
        tr_ls = cat([c["tr_ls"][i] for c, i, _ in parts])
        tr_raw = cat([c["tr_raw"][i] for c, i, _ in parts])
        builder = HeadFeatureBuilder(config).fit(tr_ls, tr_raw)
        Xtr = cat([builder.transform(c["tr_emb"][i], c["tr_ls"][i], c["tr_raw"][i])
                   for c, i, _ in parts])
        ytr = cat([c["tr_y"][i] for c, i, _ in parts])
        Xva = cat([builder.transform(c["tr_emb"][v], c["tr_ls"][v], c["tr_raw"][v])
                   for c, _, v in parts])
        yva = cat([c["tr_y"][v] for c, _, v in parts])
        Xte = builder.transform(tgt["te_emb"], tgt["te_ls"], tgt["te_raw"])
        model, _ = train_mod.train_head(Xtr, ytr, Xva, yva, loss, config,
                                        seed=seed, device=device)
        return train_mod.predict_head(model, Xte, loss, config, device=device)

    def _idx(dc: dict, units: np.ndarray):
        return torch.as_tensor(np.where(np.isin(dc["tr_u"], units))[0], device=device)

    def _baseline_predict(bname: str, seed: int,
                          parts: list[tuple[dict, np.ndarray, np.ndarray]]) -> np.ndarray:
        """``parts``: list of (numpy window dict, train_mask, val_mask) over the
        respective train windows; predicts the target test windows."""
        tr_w = np.concatenate([c["w"][m] for c, m, _ in parts])
        tr_y = np.concatenate([c["y"][m] for c, m, _ in parts])
        va_w = np.concatenate([c["w"][v] for c, _, v in parts])
        va_y = np.concatenate([c["y"][v] for c, _, v in parts])
        bl = baselines_mod.make_baseline(bname, config, seed=seed)
        bl.fit(tr_w, tr_y, va_w, va_y)
        return bl.predict(tgt_win["te_w"])

    # numpy window views for the baselines
    src_np = load_embedding_cache(src_cfg)
    tgt_np = load_embedding_cache(tgt_cfg)
    src_win = {"w": src_np["train_windows"], "y": src_np["train_labels"], "u": src_np["train_units"]}
    tgt_win = {"w": tgt_np["train_windows"], "y": tgt_np["train_labels"], "u": tgt_np["train_units"],
               "te_w": tgt_np["test_windows"]}
    te_y = np.asarray(tgt_np["test_labels"], np.float64)  # unclipped target truth

    def _emit(model: str, mode: str, k: int, seed: int, loss: str, pred: np.ndarray):
        append_result_row(out_csv, _transfer_row(
            config, model, mode, source_dataset, target_dataset, k, seed, loss, te_y, pred))
        done.add((model, mode, source_dataset, target_dataset, str(k), str(seed), loss))

    for seed in seeds:
        s_tr_u, s_va_u = data_mod.unit_train_val_split(src_units, config.val_fraction, seed)
        s_head = (src, _idx(src, s_tr_u), _idx(src, s_va_u))
        s_win = (src_win, np.isin(src_win["u"], s_tr_u), np.isin(src_win["u"], s_va_u))

        # ---- zero-shot: source-only training, target test ----
        for loss in losses:
            if (model_tag, "zero_shot", source_dataset, target_dataset, "0", str(seed), loss) not in done:
                _emit(model_tag, "zero_shot", 0, seed, loss,
                      _train_predict_head(loss, seed, [s_head]))
        for bname in baseline_names:
            if (bname, "zero_shot", source_dataset, target_dataset, "0", str(seed), "native") not in done:
                _emit(bname, "zero_shot", 0, seed, "native",
                      _baseline_predict(bname, seed, [s_win]))

        # ---- few-shot: k target units, with and without the source fleet ----
        for k in shots:
            if k > len(tgt_units):
                continue
            sampled = data_mod.subsample_units(tgt_units, k, seed)
            t_tr_u, t_va_u = data_mod.unit_train_val_split(sampled, config.val_fraction, seed)
            t_head = (tgt, _idx(tgt, t_tr_u), _idx(tgt, t_va_u))
            t_win = (tgt_win, np.isin(tgt_win["u"], t_tr_u), np.isin(tgt_win["u"], t_va_u))
            for loss in losses:
                if (model_tag, "target_only", source_dataset, target_dataset, str(k), str(seed), loss) not in done:
                    _emit(model_tag, "target_only", k, seed, loss,
                          _train_predict_head(loss, seed, [t_head]))
                if (model_tag, "source+target", source_dataset, target_dataset, str(k), str(seed), loss) not in done:
                    _emit(model_tag, "source+target", k, seed, loss,
                          _train_predict_head(loss, seed, [s_head, t_head]))
            for bname in baseline_names:
                if (bname, "target_only", source_dataset, target_dataset, str(k), str(seed), "native") not in done:
                    _emit(bname, "target_only", k, seed, "native",
                          _baseline_predict(bname, seed, [t_win]))
                if (bname, "source+target", source_dataset, target_dataset, str(k), str(seed), "native") not in done:
                    _emit(bname, "source+target", k, seed, "native",
                          _baseline_predict(bname, seed, [s_win, t_win]))
    return out_csv
