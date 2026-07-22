"""The factor-probe harness -- the engine for every Tier-2 playbook chapter (§38).

A probe = "sweep ONE playbook factor over a set of levels on an anchor dataset with a
reduced roster (top-2 TSFMs + top-2 cheap foils + best NN), scoring each level with the
win-rule". ``run_factor_probe`` applies the level's intervention as a ``Config``
override, builds the (idempotent) Stage A cache at the intervened shape, runs the TSFM
head(s) + the reduced baselines at the ablation-winner unit count, and appends
``probe_<factor>.csv`` rows keyed by ``(dataset, model, factor, level, n_units, seed,
loss)`` -- a success-map input (``src/scoring.py``). ``probe_roster`` resolves the
reduced roster from a Tier-1 results glob so probes automatically use the strongest
comparators.

Interventions (RESEARCH_PLAN §1) are additive/subtractive collection choices, expressed
as config overrides so a probe never mutates a kept reading:
  * ``channels``  (RQ-C, subtractive): each level is a named ``sensor_columns`` subset.
  * ``noise``     (RQ-H, perturbative, SIM ONLY): each level is a ``noise_injection``
    spec; ``data.load_prepared`` fires the real-dataset guard.
  * any other factor whose levels are already ``Config``-override dicts (the Phase-B
    aggregation/feature-mode knobs slot in here with no harness change).

CPU-testable: the ``embedder_factory`` seam injects a mock, exactly as in
``run_ablation`` / ``run_campaign``; restart skips completed cells.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .config import Config
from . import data as data_mod
from . import baselines as baselines_mod
from .evaluate import append_result_row, completed_cells

# One probe cell (a success-map row's identity). factor/level join the standard axes.
PROBE_KEYS = ["dataset", "model", "factor", "level", "n_units", "seed", "loss"]

# Reduced-roster categories (RESEARCH_PLAN §6): cheap non-DL foils and from-scratch NN.
FOIL_MODELS = ("gbm", "minirocket", "catch22_gbm")
NN_MODELS = ("cnn", "lstm")


def _level_overrides(factor: str, level_value) -> dict:
    """Config overrides that realize ``level_value`` for ``factor``. ``channels`` takes
    a channel-name list, ``noise`` a ``noise_injection`` spec dict; any other factor
    passes a ready-made override dict straight through (the Phase-B knobs)."""
    if factor == "channels":
        return {"sensor_columns": list(level_value)}
    if factor == "noise":
        return {"noise_injection": dict(level_value)}
    if isinstance(level_value, dict):
        return dict(level_value)
    raise ValueError(
        f"factor {factor!r}: each level must be a dict of config overrides (got "
        f"{type(level_value).__name__}); 'channels' takes a channel-name list and "
        f"'noise' takes a noise-spec dict.")


def run_factor_probe(
    config: Config,
    factor: str,
    levels: dict,
    models: Optional[list[str]] = None,
    baselines: Optional[list[str]] = None,
    device: str = "cpu",
    seeds: Optional[list[int]] = None,
    n_units: Optional[int] = None,
    losses: Optional[list[str]] = None,
    embedder_factory: Optional[Callable[[Config], object]] = None,
    out_csv: Optional[str | Path] = None,
) -> Path:
    """Sweep ``factor`` over ``levels`` (a ``{level_name: level_value}`` dict) with the
    reduced ``models`` + ``baselines`` roster, appending win-rule-ready rows to
    ``probe_<factor>.csv``. ``n_units`` defaults to the full fleet (the ablation-winner
    volume); ``embedder_factory`` injects a CPU mock. Restartable."""
    from .embeddings import build_embedding_cache, load_embedding_cache
    from .sweep import _to_device_cache, _fit_predict_tsfm, _row

    models = models if models is not None else [config.model_name]
    baselines = baselines if baselines is not None else ["gbm", "predict_mean"]
    seeds = seeds if seeds is not None else list(config.sweep_seeds)
    losses = losses if losses is not None else ["mse"]
    out_csv = Path(out_csv) if out_csv else config.results_path(f"probe_{factor}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    done = completed_cells(out_csv, PROBE_KEYS)

    def _probe_row(cfg, model, level_name, nu, seed, loss, y_true, y_pred, bwin=""):
        row = _row(cfg, model, nu, seed, loss, y_true, y_pred, baseline_window=bwin)
        row["factor"] = factor
        row["level"] = level_name
        return row

    for level_name, level_value in levels.items():
        over = _level_overrides(factor, level_value)
        for model_name in models:
            cfg = config.replace(model_name=model_name, **over)
            emb = embedder_factory(cfg) if embedder_factory is not None else None
            build_embedding_cache(cfg, embedder=emb)          # idempotent per (level, model)
            cache = load_embedding_cache(cfg)
            dc = _to_device_cache(cache, device)
            all_units = np.unique(dc["tr_u"])
            nu = n_units if n_units is not None else int(all_units.size)
            model_tag = model_name.split("/")[-1] + "_mlp"
            te_y = np.asarray(cache["test_labels"], np.float64)
            for seed in seeds:
                sampled = data_mod.subsample_units(all_units, nu, seed)
                train_u, val_u = data_mod.unit_train_val_split(sampled, cfg.val_fraction, seed)
                tr_mask = np.isin(dc["tr_u"], train_u)
                va_mask = np.isin(dc["tr_u"], val_u)
                # ---- TSFM head, one arm per loss ----
                for loss in losses:
                    key = (cfg.dataset, model_tag, factor, level_name, str(nu), str(seed), loss)
                    if key in done:
                        continue
                    pred = _fit_predict_tsfm(cfg, dc, tr_mask, va_mask, loss, seed, device)
                    append_result_row(out_csv, _probe_row(
                        cfg, model_tag, level_name, nu, seed, loss, dc["te_y"], pred))
                    done.add(key)
                # ---- reduced baselines on the same cached windows ----
                b_tr = np.isin(cache["train_units"], train_u)
                b_va = np.isin(cache["train_units"], val_u)
                for bname in baselines:
                    bkey = (cfg.dataset, bname, factor, level_name, str(nu), str(seed), "native")
                    if bkey in done:
                        continue
                    bl = baselines_mod.make_baseline(bname, cfg, seed=seed)
                    bl.fit(cache["train_windows"][b_tr], cache["train_labels"][b_tr],
                           cache["train_windows"][b_va], cache["train_labels"][b_va])
                    pred = bl.predict(cache["test_windows"])
                    append_result_row(out_csv, _probe_row(
                        cfg, bname, level_name, nu, seed, "native", te_y, pred,
                        bwin=cfg.window_size))
                    done.add(bkey)
    return out_csv


def probe_roster(
    results_glob: str | Path,
    metric: str = "nasa_clipped",
) -> tuple[list[str], list[str], Optional[str]]:
    """Resolve the Tier-2 reduced roster from a Tier-1 results glob: the two best TSFM
    heads, the two best cheap foils, and the single best from-scratch NN, ranked by the
    seed-and-cell-mean of ``metric`` (lower is better). Deterministic; empties/absent
    categories return fewer entries (or ``None`` for the NN)."""
    from .scoring import _load_glob, is_tsfm_model

    rows = _load_glob(results_glob)
    per_model: dict[str, list[float]] = {}
    for r in rows:
        if r.get(metric) in (None, ""):
            continue
        per_model.setdefault(str(r["model"]), []).append(float(r[metric]))
    means = {m: float(np.mean(v)) for m, v in per_model.items()}
    tsfms = sorted((m for m in means if is_tsfm_model(m)), key=lambda m: means[m])
    foils = sorted((m for m in means if m in FOIL_MODELS), key=lambda m: means[m])
    nns = sorted((m for m in means if m in NN_MODELS), key=lambda m: means[m])
    return tsfms[:2], foils[:2], (nns[0] if nns else None)
