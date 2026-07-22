"""Metrics, the C-MAPSS test protocol, and run provenance.

Metrics (RESEARCH_PLAN sec.6): RMSE (comparability), MAE, and the asymmetric NASA
scoring function (punishes late predictions -- the maintenance-relevant metric).

Provenance (Task 2.3): every run writes its full resolved config + git code state +
package versions alongside metrics, so any results CSV is reproducible.

This module reads NOTHING but predictions and truth arrays -- it is the only place
the test labels are consumed (Task 2.4). It imports no training code (train.py
imports ``rmse`` from here), so keep it dependency-light.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from .config import Config

# Results-file schema version (distinct from the cache schema). v2 = both-protocol
# metric columns (clipped + unclipped) + ablation axes (Task 1.4).
RESULTS_SCHEMA_VERSION = 2

# Numeric metric columns written per row (both protocols, Task 1.4).
METRIC_FIELDS = (
    "rmse_clipped", "mae_clipped", "nasa_clipped",
    "rmse_unclipped", "mae_unclipped", "nasa_unclipped",
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, np.float64), np.asarray(y_pred, np.float64)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, np.float64), np.asarray(y_pred, np.float64)
    return float(np.mean(np.abs(y_pred - y_true)))


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """PHM08 asymmetric score (Saxena et al. 2008): late predictions (underestimated
    RUL, d>0) are penalised more heavily than early ones. d = pred - true;
    s = sum(exp(-d/13) - 1) for d<0, sum(exp(d/10) - 1) for d>=0. Lower is better."""
    d = np.asarray(y_pred, np.float64) - np.asarray(y_true, np.float64)
    late = d >= 0
    s = np.where(late, np.exp(d / 10.0) - 1.0, np.exp(-d / 13.0) - 1.0)
    return float(np.sum(s))


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                         max_rul: float) -> dict:
    """Compute BOTH test-label protocols (Task 1.4) from the UNCLIPPED truth.

    * ``*_clipped``   -- ground-truth RUL clipped at ``max_rul`` (predictions are
                         already in [0, max_rul]); the literature-comparable numbers.
    * ``*_unclipped`` -- against the raw RUL_FDxxx.txt target; inflated by the 11/100
                         FD001 units with true RUL > 125 the head cannot reach.
    """
    y_true = np.asarray(y_true, np.float64)
    y_pred = np.asarray(y_pred, np.float64)
    y_clip = np.clip(y_true, None, float(max_rul))
    return {
        "rmse_clipped": rmse(y_clip, y_pred),
        "mae_clipped": mae(y_clip, y_pred),
        "nasa_clipped": nasa_score(y_clip, y_pred),
        "rmse_unclipped": rmse(y_true, y_pred),
        "mae_unclipped": mae(y_true, y_pred),
        "nasa_unclipped": nasa_score(y_true, y_pred),
        "n": int(len(y_true)),
    }


# ---------------------------------------------------------------------------
# Earliness -- "too early is also bad" (RESEARCH_PLAN §8; CHANGES.md §37)
#
# Sign convention is the horizon `bias` convention (CHANGES.md §16) and nasa_score's:
# d = pred - true. d >= 0 is the penalized "dangerously LATE" side (the model claims
# MORE remaining life than there is -> a failure is missed); d < 0 is "wastefully
# EARLY" (predicts less life than remains -> premature maintenance wastes useful life).
# NASA score already punishes lateness as the scalar; this layer makes the two-sided
# distribution explicit and sweeps the cost trade-off instead of fixing one ratio.
# ---------------------------------------------------------------------------
def earliness_histogram(y_true, y_pred, bin_edges) -> dict:
    """Distribution of the signed error ``d = pred - true`` across ``bin_edges``
    (internal breakpoints; ``-inf`` / ``+inf`` are prepended/appended). Returns the
    per-bin fractions plus the headline split: ``frac_late`` (d >= 0, dangerous) vs
    ``frac_early`` (d < 0, wasteful). Include 0.0 in ``bin_edges`` so no bin straddles
    the late/early boundary. Raises on an empty input (no distribution to report)."""
    y_true = np.asarray(y_true, np.float64)
    y_pred = np.asarray(y_pred, np.float64)
    n = int(len(y_true))
    if n == 0:
        raise ValueError("earliness_histogram needs at least one prediction")
    d = y_pred - y_true
    edges = [-np.inf, *sorted(float(e) for e in bin_edges), np.inf]
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (d >= lo) & (d < hi)
        n_bin = int(mask.sum())
        bins.append({"lo": lo, "hi": hi, "n_bin": n_bin, "frac": n_bin / n,
                     "side": "late" if lo >= 0 else "early"})
    return {"n": n, "mean_signed_error": float(d.mean()),
            "frac_late": float(np.mean(d >= 0)), "frac_early": float(np.mean(d < 0)),
            "bins": bins}


def cost_curve(y_true, y_pred, cost_ratios) -> dict:
    """Maintenance cost swept over ``cost_ratios`` = late_cost / early_cost (early_cost
    fixed at 1): ``cost(ratio) = Σ max(0, true-pred) + ratio · Σ max(0, pred-true)`` --
    the wasteful under-prediction total plus ``ratio`` times the dangerous
    over-prediction total. No single arbitrary ratio; the whole curve is the result
    (RESEARCH_PLAN §8). Returns ``{ratio: cost}``."""
    y_true = np.asarray(y_true, np.float64)
    y_pred = np.asarray(y_pred, np.float64)
    d = y_pred - y_true
    late = float(np.clip(d, 0.0, None).sum())    # over-prediction of RUL (dangerous)
    early = float(np.clip(-d, 0.0, None).sum())  # under-prediction of RUL (wasteful)
    return {float(r): early + float(r) * late for r in cost_ratios}


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def package_versions() -> dict:
    import importlib
    # The v2 "when do TSFMs work" build adds four more backbones and the censoring /
    # cheap-feature libraries (IMPLEMENTATION_PLAN §2); record their versions in the
    # provenance JSON so a run states exactly which backbone/library builds produced it.
    # Modules absent on a given machine simply report "not-installed" -- the import names
    # are momentfm (MOMENT), uni2ts (Moirai), timesfm (TimesFM), tsfm_public (granite-tsfm
    # / TTM), pycatch22 (catch22 foil), and sksurv/lifelines (censored metrics).
    versions = {}
    for m in ["numpy", "pandas", "scipy", "sklearn", "torch",
              "coral_pytorch", "chronos", "lightgbm", "sktime",
              "momentfm", "uni2ts", "timesfm", "tsfm_public", "pycatch22",
              "sksurv", "lifelines"]:
        try:
            versions[m] = getattr(importlib.import_module(m), "__version__", "unknown")
        except Exception:
            versions[m] = "not-installed"
    return versions


def git_state(repo_dir: str | Path = ".") -> dict:
    def _run(args):
        try:
            return subprocess.check_output(["git", *args], cwd=str(repo_dir),
                                           stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None
    commit = _run(["rev-parse", "HEAD"])
    describe = _run(["describe", "--always", "--dirty", "--tags"])
    status = _run(["status", "--porcelain"])
    return {"commit": commit, "describe": describe,
            "dirty": bool(status) if status is not None else None}


def run_metadata(config: Config, repo_dir: str | Path = ".") -> dict:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": config.to_dict(),
        "git": git_state(repo_dir),
        "packages": package_versions(),
    }


def save_run_metadata(config: Config, path: str | Path, repo_dir: str | Path = ".") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run_metadata(config, repo_dir), indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# Results CSV (data-scaling curve rows) with completed-cell detection
# ---------------------------------------------------------------------------
def append_result_row(csv_path: str | Path, row: dict) -> None:
    """Append one metrics row, writing a header if the file is new. Used to
    checkpoint the sweep after every grid cell (Task 3, Stage B)."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def ensure_csv_schema(csv_path: str | Path, fieldnames: list[str]) -> None:
    """Fail LOUDLY if an existing results CSV has a different header than the rows
    about to be appended. ``append_result_row`` writes values in the new row's key
    order without re-writing the header, so appending a changed schema would
    silently misalign columns -- the caller must archive the old file instead."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return
    with open(csv_path, newline="") as f:
        header = next(csv.reader(f), None)
    if header is not None and header != list(fieldnames):
        raise ValueError(
            f"{csv_path} has header {header} but rows to append have fields "
            f"{list(fieldnames)}. Move/archive the old file (e.g. into "
            f"results/archive/) and rerun -- appending would silently misalign columns."
        )


def paired_ttest(a_values, b_values) -> tuple[float, float]:
    """Paired t statistic + two-sided p-value of ``a - b`` over the shared seeds
    (scipy ``ttest_rel``). Returns ``(nan, nan)`` when fewer than 2 pairs or the
    difference has no variance (ties / constant offset) -- so callers never see
    scipy's +/-inf. The single reusable paired-seed test core (generalized from the
    horizon CORN-vs-MSE comparison to arbitrary model pairs; CHANGES.md §36)."""
    from scipy import stats as _stats
    a = np.asarray(a_values, np.float64)
    b = np.asarray(b_values, np.float64)
    d = a - b
    if len(d) >= 2 and not np.allclose(d, d[0]):
        t, p = _stats.ttest_rel(a, b)
        return float(t), float(p)
    return float("nan"), float("nan")


def paired_seed_ttest(
    horizon_csv: str | Path,
    model: str = "chronos-2_mlp",
    loss_a: str = "corn",
    loss_b: str = "mse",
    metric: str = "mae_clipped",
) -> list[dict]:
    """Paired-by-seed t-test of ``loss_a`` vs ``loss_b`` per (max_rul, n_units,
    RUL-bin) cell of ``horizon.csv``. Pairing on seed is valid because both loss
    arms of a seed share the sampled units and the train/val split (sweep
    protocol), so seed-to-seed variation cancels in the difference.

    Returns one dict per cell: seed-mean of each arm, ``mean_delta`` = mean(a - b)
    (negative => ``loss_a`` better when the metric is an error), t statistic and
    two-sided p-value (scipy ``ttest_rel``; nan when < 2 paired seeds). With 5
    seeds this is a low-powered test -- treat p-values as descriptive and read
    them alongside the per-bin means, not instead of them."""
    cells: dict[tuple, dict[str, dict[int, float]]] = {}
    with open(horizon_csv, newline="") as f:
        for r in csv.DictReader(f):
            if r["model"] != model or r["loss"] not in (loss_a, loss_b):
                continue
            key = (r["max_rul"], int(r["n_units"]), r["bin_lo"], r["bin_hi"])
            cell = cells.setdefault(key, {loss_a: {}, loss_b: {}})
            cell[r["loss"]][int(r["seed"])] = float(r[metric])
    out = []
    for (max_rul, n_units, bin_lo, bin_hi), arms in sorted(cells.items()):
        seeds = sorted(set(arms[loss_a]) & set(arms[loss_b]))
        a = np.array([arms[loss_a][s] for s in seeds], np.float64)
        b = np.array([arms[loss_b][s] for s in seeds], np.float64)
        t, p = paired_ttest(a, b)  # shared paired-seed core (nan-safe)
        out.append({
            "max_rul": max_rul, "n_units": n_units, "bin_lo": bin_lo, "bin_hi": bin_hi,
            "metric": metric, "n_seeds": len(seeds),
            f"mean_{loss_a}": float(a.mean()) if len(seeds) else float("nan"),
            f"mean_{loss_b}": float(b.mean()) if len(seeds) else float("nan"),
            "mean_delta": float((a - b).mean()) if len(seeds) else float("nan"),
            "t": float(t), "p": float(p),
        })
    return out


def load_results(csv_path: str | Path) -> list[dict]:
    """Load a results CSV as a list of row dicts (numeric fields coerced)."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []
    int_fields = ("n_units", "seed", "tsfm_context_length", "schema_version", "n")
    float_fields = METRIC_FIELDS + ("rmse", "mae", "nasa_score")  # + legacy v1 names
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            for k in int_fields:
                if r.get(k) not in (None, ""):
                    r[k] = int(float(r[k]))
            for k in float_fields:
                if r.get(k) not in (None, ""):
                    r[k] = float(r[k])
            rows.append(r)
    return rows


def aggregate_data_scaling(
    csv_path: str | Path, metric: str = "rmse_clipped",
    dataset: Optional[str] = None,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Aggregate the data-scaling curve: for each (model, loss) series return
    (unit_counts, mean, std) of ``metric`` over seeds -- the headline figure with
    error bands (RESEARCH_PLAN sec.6). Keeps NO logic in the notebook. ``metric``
    defaults to the literature-comparable clipped RMSE (Task 1.4). ``dataset``
    filters to one dataset's rows -- results CSVs may hold several (§21), and
    pooling them would silently average across datasets."""
    rows = load_results(csv_path)
    if dataset is not None:
        rows = [r for r in rows if r.get("dataset", "") == dataset]
    series: dict[str, dict[int, list[float]]] = {}
    for r in rows:
        label = r["model"] if r.get("loss") in (None, "", "native") else f"{r['model']}[{r['loss']}]"
        series.setdefault(label, {}).setdefault(r["n_units"], []).append(r[metric])
    out = {}
    for label, by_n in series.items():
        ns = np.array(sorted(by_n))
        mean = np.array([np.mean(by_n[n]) for n in ns])
        std = np.array([np.std(by_n[n]) for n in ns])
        out[label] = (ns, mean, std)
    return out


def archive_results_v1(results_dir: str | Path) -> Optional[Path]:
    """Preserve a pre-existing ``results.csv`` as ``results_v1.csv`` before the
    v2 schema starts writing ``results_v2.csv`` (Task 1.4 -- never overwrite v1).
    Idempotent; returns the archive path if it created/kept one, else None.
    """
    results_dir = Path(results_dir)
    legacy = results_dir / "results.csv"
    archive = results_dir / "results_v1.csv"
    if legacy.exists() and not archive.exists():
        shutil.copy2(legacy, archive)
        return archive
    return archive if archive.exists() else None


def load_learning_curve(curve_csv: str | Path) -> dict[str, tuple[list, list]]:
    """Load a per-cell learning-curve CSV into {metric: (x, y)} for plotting."""
    out: dict[str, tuple[list, list]] = {"train_loss": ([], []),
                                         "val_loss": ([], []), "val_rmse": ([], [])}
    with open(curve_csv, newline="") as f:
        for r in csv.DictReader(f):
            metric = r["metric"]
            x = float(r["step"]) if r["step"] not in ("", None) else float(r["epoch"])
            out.setdefault(metric, ([], []))
            out[metric][0].append(x)
            out[metric][1].append(float(r["value"]))
    return out


def completed_cells(csv_path: str | Path, key_fields: list[str]) -> set:
    """Return the set of already-completed cell keys (tuples of ``key_fields``
    values) so a restarted sweep skips finished cells (Task 3, Stage B)."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return set()
    done = set()
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            done.add(tuple(r.get(k) for k in key_fields))
    return done
