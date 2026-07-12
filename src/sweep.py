"""Data-fraction x loss x seed sweep runner (Task 1 sweep.py; RESEARCH_PLAN sec.6).

Consumes ONLY the Stage A cache (raw windows + labels + unit_ids + pooled
embeddings). It never constructs an embedder and never re-embeds -- that is the
whole point of the caching economics (Task 3). If you find an embedder reference
in this file, it is a bug.

Per cell it: subsamples engine units (by unit, seeded; sampled IDs saved to the run
dir -- Task 2.3), splits train/val by unit (no unit crosses a split -- Task 2.4),
trains an MLP head per loss on the cached embeddings, fits the baselines on the same
cached raw windows, evaluates on the fixed test set, and appends a metrics row.
Checkpointing after every cell + completed-cell detection make it restartable
(Task 3, Stage B).

Loss arms (mse/corn/quantile) apply to the TSFM MLP head. Baselines run as their
native regressors (loss column = "native"); see CHANGES.md for this protocol note.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from .config import Config
from . import data as data_mod
from . import train as train_mod
from . import baselines as baselines_mod
from .evaluate import (
    evaluate_predictions, append_result_row, completed_cells, save_run_metadata,
)

CELL_KEYS = ["model", "n_units", "seed", "loss"]


def _mask_for_units(unit_ids: np.ndarray, target_units: np.ndarray) -> np.ndarray:
    return np.isin(unit_ids, np.asarray(target_units))


def _save_sampled_units(run_dir: Path, n_units: int, seed: int,
                        sampled: np.ndarray, train_u: np.ndarray, val_u: np.ndarray) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"units_n{n_units}_seed{seed}.json").write_text(json.dumps(
        {"n_units": int(n_units), "seed": int(seed),
         "sampled_units": [int(u) for u in sampled],
         "train_units": [int(u) for u in train_u],
         "val_units": [int(u) for u in val_u]}, indent=2))


def run_sweep(
    config: Config,
    cache: Optional[dict] = None,
    results_csv: Optional[str | Path] = None,
    run_dir: Optional[str | Path] = None,
    baseline_names: Optional[list[str]] = None,
    losses: Optional[list[str]] = None,
    device: str = "cpu",
) -> Path:
    """Run the full grid, appending to ``results_csv``. Returns its path.

    ``cache`` may be pre-loaded (dict from ``embeddings.load_embedding_cache``);
    if None it is loaded from ``config.cache_path()``.
    """
    from .embeddings import load_embedding_cache  # local import: no embedder needed

    if cache is None:
        cache = load_embedding_cache(config)
    run_dir = Path(run_dir) if run_dir else Path(config.results_dir) / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    results_csv = Path(results_csv) if results_csv else Path(config.results_dir) / "results.csv"
    curves_dir = run_dir / "learning_curves"
    save_run_metadata(config, run_dir / "run_metadata.json")

    losses = losses if losses is not None else config.losses
    if baseline_names is None:
        baseline_names = ["predict_mean", "gbm", "minirocket", "cnn", "lstm"]
    model_tag = config.model_name.split("/")[-1] + "_mlp"

    tr_emb, tr_win = cache["train_emb"], cache["train_windows"]
    tr_y, tr_u = cache["train_labels"], cache["train_units"]
    te_emb, te_win, te_y = cache["test_emb"], cache["test_windows"], cache["test_labels"]

    all_units = np.unique(tr_u)
    done = completed_cells(results_csv, CELL_KEYS)

    for n_units in config.data_unit_counts:
        if n_units > len(all_units):
            continue
        for seed in config.sweep_seeds:
            sampled = data_mod.subsample_units(all_units, n_units, seed)
            train_u, val_u = data_mod.unit_train_val_split(sampled, config.val_fraction, seed)
            _save_sampled_units(run_dir, n_units, seed, sampled, train_u, val_u)

            tr_mask = _mask_for_units(tr_u, train_u)
            va_mask = _mask_for_units(tr_u, val_u)

            # ---- TSFM MLP head, one arm per loss (cached embeddings only) ----
            for loss in losses:
                key = (model_tag, str(n_units), str(seed), loss)
                if key in done:
                    continue
                curve = curves_dir / f"{model_tag}_n{n_units}_seed{seed}_{loss}.csv"
                model, _hist = train_mod.train_head(
                    tr_emb[tr_mask], tr_y[tr_mask], tr_emb[va_mask], tr_y[va_mask],
                    loss, config, seed=seed, device=device, log_csv_path=curve,
                )
                pred = train_mod.predict_head(model, te_emb, loss, config, device=device)
                _append(results_csv, model_tag, n_units, seed, loss, te_y, pred, config)
                done.add(key)

            # ---- baselines on the SAME cached raw windows (no re-embedding) ----
            for bname in baseline_names:
                key = (bname, str(n_units), str(seed), "native")
                if key in done:
                    continue
                bl = baselines_mod.make_baseline(bname, config, seed=seed)
                bl.fit(tr_win[tr_mask], tr_y[tr_mask], tr_win[va_mask], tr_y[va_mask])
                pred = bl.predict(te_win)
                _append(results_csv, bname, n_units, seed, "native", te_y, pred, config)
                done.add(key)

    return results_csv


def _append(results_csv, model, n_units, seed, loss, y_true, y_pred, config: Config):
    metrics = evaluate_predictions(y_true, y_pred)
    append_result_row(results_csv, {
        "model": model, "n_units": int(n_units), "seed": int(seed), "loss": loss,
        "dataset": config.dataset, "max_rul": config.max_rul,
        "window_size": config.window_size, "pooling": config.pooling,
        **metrics,
    })
