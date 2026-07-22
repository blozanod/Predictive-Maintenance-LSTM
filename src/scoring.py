"""Scoring & the win-rule -- the formal realization of RESEARCH_PLAN §8 (CHANGES.md §36).

Reads the per-combo result CSVs the campaign writes (one per ``<dataset>_<model>``),
assembles the **success map**, and applies the per-cell win / tie / loss / hollow rule
that turns a pile of metrics into the study's headline object.

Definitions (RESEARCH_PLAN §8):
  * **Primary metric** = ``nasa_clipped`` -- the asymmetric PHM08 score that punishes
    late predictions hardest. Lower is better for every metric here (all are errors),
    so "beats" means a strictly lower seed-mean and ``margin`` is signed so positive =
    the TSFM is better. RMSE is reported alongside for context, never as the decider.
  * **Strongest baseline per cell** = the toughest bar: the best (lowest) seed-mean
    over the COMPETITOR baseline rows in that ``(dataset, n_units[, factor, level])``
    cell -- not a fixed reference model. The trivial floors (``predict_mean``,
    ``cycle_reg``) are NOT competitors (RESEARCH_PLAN §6 lists them as *floors*, apart
    from the from-scratch / cheap-feature baselines); they drive the hollow guard
    instead, which is what makes "even the winner fails" reachable.
  * **Win** iff the TSFM's seed-mean beats the strongest baseline's by more than
    ``config.win_margin`` AND a paired-seed t-test (shared seeds => valid pairing)
    supports it at ``config.win_alpha``. **Loss** is the significant reverse.
    Everything else -- within-margin or not significant -- is a **tie**.
  * **Hollow** overrides a "win" when the absolute-floor guard fires: even the winning
    TSFM's error is no better than the trivial ``predict_mean`` floor, so the win is
    over baselines that all fail and does not count as a success condition.

Nothing here re-embeds or re-runs anything -- it scores EXISTING CSVs, so none of the
scoring config fields are cache keys.
"""

from __future__ import annotations

import glob as _glob
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .config import Config
from .evaluate import load_results, paired_ttest

# The primary win metric (RESEARCH_PLAN §8) and the RMSE reported alongside it.
PRIMARY_METRIC = "nasa_clipped"
SECONDARY_METRIC = "rmse_clipped"

# The from-scratch / cheap-feature / floor comparators (RESEARCH_PLAN §6). Any model
# NOT ending in ``_mlp`` is a baseline; the TSFM heads are ``<tag>_mlp``. This set is
# the documented roster so a typo'd model name is never silently treated as a TSFM.
BASELINE_MODELS = frozenset({
    "predict_mean", "gbm", "minirocket", "cnn", "lstm",
    "cycle_reg", "gbm_age", "catch22_gbm",
})
# The trivial FLOORS (RESEARCH_PLAN §6: "+ floors: predict-mean, cycle-count linear
# regression"). They are the usability guard, NOT competitors -- a TSFM that beats
# every real baseline but not the trivial floor is a hollow win.
# DECISION (uncited): treating predict_mean/cycle_reg as floors (not part of the
# "strongest baseline" competitor bar) is what makes the hollow guard reachable;
# RESEARCH_PLAN §6 lists them as floors, apart from the competitor baselines.
FLOOR_MODELS = frozenset({"predict_mean", "cycle_reg"})
# The absolute-floor reference model (the "everything fails" guard).
FLOOR_MODEL = "predict_mean"


def is_tsfm_model(model: str) -> bool:
    """A TSFM head row (``<model-tag>_mlp``) as opposed to a baseline comparator."""
    return str(model).endswith("_mlp")


def is_competitor_baseline(model: str) -> bool:
    """A real competitor baseline: a non-TSFM, non-floor model (gbm, minirocket, cnn,
    lstm, gbm_age, catch22_gbm, ...). The bar the win-rule must clear."""
    return not is_tsfm_model(model) and str(model) not in FLOOR_MODELS


def _seed_mean(seedvals: dict[int, float]) -> float:
    return float(np.mean(list(seedvals.values())))


def _grouped_seed_values(rows: Iterable[dict], metric: str,
                         cell_fields: tuple[str, ...]) -> dict[tuple, dict[str, dict[int, float]]]:
    """``{cell: {model: {seed: metric_value}}}`` over the given cell axes. Rows missing
    the metric or a cell field are skipped (mixed-schema CSVs never crash scoring)."""
    out: dict[tuple, dict[str, dict[int, float]]] = {}
    for r in rows:
        if r.get(metric) in (None, "") or any(r.get(f) in (None, "") for f in cell_fields):
            continue
        cell = tuple(r[f] for f in cell_fields)
        out.setdefault(cell, {}).setdefault(str(r["model"]), {})[int(r["seed"])] = float(r[metric])
    return out


def strongest_baseline_per_cell(
    rows: Iterable[dict],
    metric: str = PRIMARY_METRIC,
    cell_fields: tuple[str, ...] = ("dataset", "n_units"),
) -> dict[tuple, tuple[str, float]]:
    """``{cell: (best_baseline_name, best_seed_mean)}`` -- the toughest baseline bar in
    each cell (the lowest baseline seed-mean; lower is better)."""
    grouped = _grouped_seed_values(rows, metric, cell_fields)
    out: dict[tuple, tuple[str, float]] = {}
    for cell, by_model in grouped.items():
        best_model, best_mean = None, np.inf
        for model, seedvals in by_model.items():
            if not is_competitor_baseline(model):
                continue
            mean = _seed_mean(seedvals)
            if mean < best_mean:
                best_model, best_mean = model, mean
        if best_model is not None:
            out[cell] = (best_model, best_mean)
    return out


def _classify(margin: float, p: float, config: Config) -> str:
    """win/tie/loss from a signed margin (>0 => TSFM better) and a paired-seed p-value.
    win/loss require the difference to clear ``win_margin`` AND be significant at
    ``win_alpha``; otherwise the cell is a tie (within noise or under-powered)."""
    significant = (p == p) and p < config.win_alpha       # p == p is False for nan
    if significant and margin > config.win_margin:
        return "win"
    if significant and margin < -config.win_margin:
        return "loss"
    return "tie"


def win_verdict(
    rows: Iterable[dict],
    config: Config,
    metric: str = PRIMARY_METRIC,
    cell_fields: tuple[str, ...] = ("dataset", "n_units"),
) -> dict[tuple, dict]:
    """Per (cell, TSFM-model) verdict against the strongest baseline in that cell.

    Returns ``{cell + (model,): verdict_dict}`` where ``verdict_dict`` carries the
    verdict (win/tie/loss/hollow), the signed ``margin``, the paired-seed ``p``, both
    seed-means, the chosen baseline, the floor, and the paired-seed count. ``rows`` is
    the combined TSFM + baseline rows (a cell with no baseline is skipped -- there is
    no bar to clear)."""
    rows = list(rows)
    grouped = _grouped_seed_values(rows, metric, cell_fields)
    strongest = strongest_baseline_per_cell(rows, metric, cell_fields)
    out: dict[tuple, dict] = {}
    for cell, by_model in grouped.items():
        if cell not in strongest:
            continue
        best_baseline, baseline_mean = strongest[cell]
        base_seedvals = by_model[best_baseline]
        floor = _seed_mean(by_model[FLOOR_MODEL]) if FLOOR_MODEL in by_model else None
        for model, seedvals in by_model.items():
            if not is_tsfm_model(model):
                continue
            tsfm_mean = _seed_mean(seedvals)
            margin = baseline_mean - tsfm_mean          # >0 => TSFM beats the bar
            shared = sorted(set(seedvals) & set(base_seedvals))
            _t, p = paired_ttest([seedvals[s] for s in shared],
                                 [base_seedvals[s] for s in shared])
            verdict = _classify(margin, p, config)
            # Absolute-floor guard: a "win" where the TSFM is no better than the
            # trivial predict-mean floor is hollow (everything fails there).
            if verdict == "win" and floor is not None and tsfm_mean >= floor - config.win_margin:
                verdict = "hollow"
            out[cell + (model,)] = {
                "verdict": verdict, "margin": float(margin), "p": float(p),
                "tsfm_mean": tsfm_mean, "best_baseline": best_baseline,
                "baseline_mean": float(baseline_mean),
                "floor": None if floor is None else float(floor),
                "n_seeds": len(shared),
            }
    return out


def _load_glob(results_glob: str | Path) -> list[dict]:
    """All rows from every CSV matching ``results_glob`` (a glob pattern OR a single
    path OR a directory), concatenated -- the campaign writes one CSV per combo."""
    p = Path(results_glob)
    if p.is_dir():
        paths = sorted(p.glob("*.csv"))
    elif any(ch in str(results_glob) for ch in "*?[") or not p.exists():
        paths = [Path(x) for x in sorted(_glob.glob(str(results_glob)))]
    else:
        paths = [p]
    rows: list[dict] = []
    for path in paths:
        rows.extend(load_results(path))
    return rows


def success_map(
    results_glob: str | Path,
    config: Optional[Config] = None,
    metric: str = PRIMARY_METRIC,
    cell_fields: tuple[str, ...] = ("dataset", "n_units"),
    secondary_metric: str = SECONDARY_METRIC,
) -> list[dict]:
    """The headline deliverable object: one row per (cell, TSFM model) with the
    verdict + margin + p + the seed-means, ordered deterministically. ``results_glob``
    may be a glob, a directory of per-combo CSVs, or one CSV. ``config`` supplies the
    win margin/alpha (defaults to ``Config()`` if omitted). ``plots.plot_success_map``
    renders the returned table."""
    config = config if config is not None else Config()
    rows = _load_glob(results_glob)
    verdicts = win_verdict(rows, config, metric, cell_fields)
    # RMSE-alongside: the secondary metric's TSFM seed-mean per (cell, model).
    rmse_grouped = _grouped_seed_values(rows, secondary_metric, cell_fields)
    table: list[dict] = []
    for key in sorted(verdicts, key=lambda k: tuple(str(x) for x in k)):
        cell, model = key[:-1], key[-1]
        v = verdicts[key]
        rmse_seedvals = rmse_grouped.get(cell, {}).get(model)
        row = {field: cell[i] for i, field in enumerate(cell_fields)}
        row["model"] = model
        row.update({
            "verdict": v["verdict"], "metric": metric,
            "margin": v["margin"], "p": v["p"], "n_seeds": v["n_seeds"],
            "tsfm_mean": v["tsfm_mean"], "best_baseline": v["best_baseline"],
            "baseline_mean": v["baseline_mean"], "floor": v["floor"],
            f"tsfm_{secondary_metric}": (_seed_mean(rmse_seedvals)
                                         if rmse_seedvals else float("nan")),
        })
        table.append(row)
    return table
