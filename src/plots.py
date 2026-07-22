"""Stage C figures: readable, saved-to-disk plots (no logic in the notebook).

Fixes to the original inline notebook plotting:
- Every figure is SAVED (``<results_dir>/figures/<name>.png`` at 300 dpi + ``.pdf``)
  instead of only shown in the Colab output.
- The ``predict_mean`` floor (~41 RMSE / ~2e4 NASA) is drawn as a flat gray
  reference line and EXCLUDED from the y-limits, so it no longer squashes the
  10-25 RMSE band where the real models live.
- NASA-score panels use a log y-axis (seed values span 3+ orders of magnitude).
- Fixed, colorblind-safe color+marker per model family (Okabe-Ito); loss arms of
  the same model share the family color and differ by linestyle.
"""
from __future__ import annotations

import csv as _csv
import re
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from .evaluate import aggregate_data_scaling, load_learning_curve, load_results

# Verdict palette for the success map (Okabe-Ito, colorblind-safe -- matches the module
# convention). win=green, loss=vermillion, tie=neutral gray, hollow=orange (a "win"
# where everything fails, flagged apart). Ordered low->high for the discrete colormap.
_VERDICT_ORDER = ("loss", "hollow", "tie", "win")
_VERDICT_CODE = {v: i for i, v in enumerate(_VERDICT_ORDER)}
_VERDICT_COLORS = {"loss": "#D55E00", "hollow": "#E69F00",
                   "tie": "#BBBBBB", "win": "#009E73"}
_VERDICT_INITIAL = {"loss": "L", "hollow": "H", "tie": "T", "win": "W"}

# Okabe-Ito, fixed assignment per model family (never cycled). gbm_age shares
# GBM's hue (same family, augmented features) but differs in marker + linestyle.
_FAMILY_STYLE = {
    "chronos-2_mlp": dict(color="#0072B2", marker="o"),
    "gbm": dict(color="#E69F00", marker="s"),
    "gbm_age": dict(color="#E69F00", marker="P", ls="--"),
    "lstm": dict(color="#009E73", marker="^"),
    "cnn": dict(color="#D55E00", marker="v"),
    "minirocket": dict(color="#CC79A7", marker="D"),
}
_FALLBACK_COLORS = ["#56B4E9", "#F0E442", "#000000"]
_LOSS_LINESTYLE = {"mse": "-", "corn": "--", "quantile": ":", "native": "-", "": "-"}
# Floors are drawn as flat reference lines and excluded from y-limits.
_FLOOR_STYLE = {"predict_mean": ("predict-mean floor", ":"),
                "cycle_reg": ("cycle-age floor (linear)", "-.")}


def _series_style(label: str) -> dict:
    """Style for an ``aggregate_data_scaling`` label (``model`` or ``model[loss]``).
    A family's own ``ls`` wins over the loss-derived linestyle."""
    m = re.fullmatch(r"(.+?)\[(.+)\]", label)
    family, loss = (m.group(1), m.group(2)) if m else (label, "native")
    style = _FAMILY_STYLE.get(family)
    if style is None:  # unknown family: deterministic fallback, no cycling
        idx = sum(family.encode()) % len(_FALLBACK_COLORS)
        style = dict(color=_FALLBACK_COLORS[idx], marker="x")
    return dict({"ls": _LOSS_LINESTYLE.get(loss, "-")}, **style)


def _save(fig, out_dir: Path, name: str, prefix: str = "") -> list[Path]:
    """Save ``fig`` as ``<out_dir>/<prefix><name>.{png,pdf}``. ``prefix`` carries the
    experiment name (``config.result_prefix()``) so figures from different
    experiments never overwrite each other; "" keeps the historical flat names."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext, kw in (("png", {"dpi": 300}), ("pdf", {})):
        p = out_dir / f"{prefix}{name}.{ext}"
        fig.savefig(p, bbox_inches="tight", **kw)
        paths.append(p)
    return paths


def _unit_count_xaxis(ax, unit_counts) -> None:
    ax.set_xscale("log")
    ax.set_xticks(sorted(unit_counts))
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.minorticks_off()
    ax.set_xlabel("training engine units")


def plot_data_scaling(
    results_csv: str | Path,
    out_dir: str | Path,
    metrics: Optional[list[tuple[str, str]]] = None,
    sanity_gate: Optional[float] = 14.0,
    show: bool = True,
    prefix: str = "",
) -> list[Path]:
    """The headline figure(s): metric vs. training units, mean +/- std over seeds.

    One figure per (dataset, metric) -- results CSVs may hold several datasets
    (dataset is part of the sweep cell key, CHANGES.md §21), and pooling their
    rows into one curve per model would silently average across datasets. The
    dataset tag joins the filename only when the CSV holds more than one.
    Returns the saved file paths.
    """
    out_dir = Path(out_dir)
    metrics = metrics or [
        ("rmse_clipped", "test RMSE (clipped protocol)"),
        ("rmse_unclipped", "test RMSE (unclipped protocol)"),
        ("nasa_clipped", "NASA score (clipped protocol)"),
    ]
    all_rows = load_results(results_csv)
    datasets = sorted({r.get("dataset", "") for r in all_rows})
    saved: list[Path] = []
    for ds in datasets:
        ds_tag = f"{ds}_" if len(datasets) > 1 and ds else ""
        for metric, ylabel in metrics:
            agg = aggregate_data_scaling(results_csv, metric=metric, dataset=ds)
            log_y = metric.startswith("nasa")
            fig, ax = plt.subplots(figsize=(7.5, 5))
            lo, hi, all_ns = np.inf, -np.inf, set()
            for label in sorted(agg):
                ns, mean, std = agg[label]
                all_ns.update(int(n) for n in ns)
                family = label.split("[")[0]
                if family in _FLOOR_STYLE:
                    floor_label, floor_ls = _FLOOR_STYLE[family]
                    ax.axhline(float(np.mean(mean)), color="#888888", ls=floor_ls,
                               lw=1.2, label=floor_label)
                    continue
                st = _series_style(label)
                ax.plot(ns, mean, lw=2, ms=5, label=label, **st)
                band_lo = np.maximum(mean - std, 1e-9) if log_y else mean - std
                ax.fill_between(ns, band_lo, mean + std, color=st["color"], alpha=0.15, lw=0)
                lo, hi = min(lo, band_lo.min()), max(hi, (mean + std).max())
            if metric == "rmse_clipped" and sanity_gate:
                ax.axhline(sanity_gate, color="#444444", ls="--", lw=1,
                           label=f"sanity gate ({sanity_gate:g})")
                hi = max(hi, sanity_gate)
            _unit_count_xaxis(ax, all_ns)
            if log_y:
                ax.set_yscale("log")
            else:
                pad = 0.05 * (hi - lo)
                ax.set_ylim(max(0.0, lo - pad), hi + pad)
            ax.set_ylabel(ylabel)
            ax.set_title(f"Data-scaling curve ({ds or 'unknown dataset'}) — {ylabel}")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8, framealpha=0.9)
            saved += _save(fig, out_dir, f"data_scaling_{ds_tag}{metric}", prefix)
            plt.show() if show else plt.close(fig)
    return saved


def plot_ablation(
    ablation_csv: str | Path,
    out_dir: str | Path,
    metric: str = "rmse_clipped",
    show: bool = True,
    prefix: str = "",
) -> list[Path]:
    """Context-length ablation: ``metric`` (mean +/- std over seeds) vs. TSFM
    context length, one line per head_features, at the default pooling; the
    pooling variants are drawn as annotated points at their own context."""
    out_dir = Path(out_dir)
    rows = load_results(ablation_csv)
    cells: dict[tuple[str, str], dict[int, list[float]]] = {}
    for r in rows:
        key = (r["head_features"], r["pooling"])
        cells.setdefault(key, {}).setdefault(int(r["tsfm_context_length"]), []).append(r[metric])

    fig, ax = plt.subplots(figsize=(7.5, 5))
    default_pooling = "forecast_token"
    palette = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
    features = sorted({f for f, _ in cells})
    contexts: set[int] = set()
    for i, feat in enumerate(features):
        by_ctx = cells.get((feat, default_pooling))
        if not by_ctx:
            continue
        ns = np.array(sorted(by_ctx))
        contexts.update(int(n) for n in ns)
        mean = np.array([np.mean(by_ctx[n]) for n in ns])
        std = np.array([np.std(by_ctx[n]) for n in ns])
        c = palette[i % len(palette)]
        ax.plot(ns, mean, marker="o", lw=2, color=c, label=f"{feat} ({default_pooling})")
        ax.fill_between(ns, mean - std, mean + std, color=c, alpha=0.15, lw=0)
    for (feat, pooling), by_ctx in sorted(cells.items()):
        if pooling == default_pooling:
            continue
        for ctx, vals in by_ctx.items():
            ax.errorbar([ctx], [np.mean(vals)], yerr=[np.std(vals)], fmt="D",
                        ms=5, capsize=3, color="#444444")
            ax.annotate(f"{feat}, {pooling}", (ctx, np.mean(vals)),
                        textcoords="offset points", xytext=(6, -4), fontsize=7)
    ax.set_xscale("log")
    ax.set_xticks(sorted(contexts))
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.minorticks_off()
    ax.set_xlabel("TSFM context length (cycles; contexts are truncated to available history)")
    ax.set_ylabel(metric)
    ax.set_title("Ablation: context length x head features (full data)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, framealpha=0.9)
    saved = _save(fig, out_dir, f"ablation_{metric}", prefix)
    plt.show() if show else plt.close(fig)
    return saved


def _bin_label(lo, hi) -> str:
    if str(lo) == "all":
        return "all"
    if str(hi) == "inf":
        return f"≥{int(float(lo))}"
    return f"{int(float(lo))}–{int(float(hi))}"


def plot_horizon(
    horizon_csv: str | Path,
    out_dir: str | Path,
    show: bool = True,
    prefix: str = "",
) -> list[Path]:
    """Horizon-stratified error: MAE and bias vs. true-RUL bin, one figure per
    (label cap, training-unit count) arm -- the cap arms (CHANGES.md §18) get
    separate figures because their bins differ. The right-most bin (true RUL >=
    max_rul) is shaded: with clipped training labels it measures saturation
    quality, not long-horizon skill (src/horizon.py docstring)."""
    out_dir = Path(out_dir)
    rows = load_results(horizon_csv)
    rows = [r for r in rows if str(r.get("bin_lo")) != "all"]
    saved: list[Path] = []
    arms = sorted({(r.get("dataset", ""), int(float(r["max_rul"])), r["n_units"])
                   for r in rows})
    for ds, max_rul, n_units in arms:
        sub = [r for r in rows
               if r["n_units"] == n_units and int(float(r["max_rul"])) == max_rul
               and r.get("dataset", "") == ds]
        bins = sorted({(float(r["bin_lo"]),
                        float("inf") if str(r["bin_hi"]) == "inf" else float(r["bin_hi"]))
                       for r in sub})
        centers = np.arange(len(bins))
        labels = sorted({r["model"] if r["loss"] in ("", "native") else f"{r['model']}[{r['loss']}]"
                         for r in sub})
        fig, (ax_mae, ax_bias) = plt.subplots(1, 2, figsize=(13, 4.8))
        for label in labels:
            m = re.fullmatch(r"(.+?)\[(.+)\]", label)
            model, loss = (m.group(1), m.group(2)) if m else (label, "native")
            st = _series_style(label)
            mae_m, mae_s, bias_m, bias_s = [], [], [], []
            for lo, hi in bins:
                vals = [(float(r["mae_clipped"]), float(r["bias"])) for r in sub
                        if r["model"] == model and r["loss"] == loss
                        and float(r["bin_lo"]) == lo]
                mae_v, bias_v = zip(*vals)
                mae_m.append(np.mean(mae_v)); mae_s.append(np.std(mae_v))
                bias_m.append(np.mean(bias_v)); bias_s.append(np.std(bias_v))
            for ax, mean, std in ((ax_mae, mae_m, mae_s), (ax_bias, bias_m, bias_s)):
                mean, std = np.asarray(mean), np.asarray(std)
                ax.plot(centers, mean, lw=2, ms=5, label=label, **st)
                ax.fill_between(centers, mean - std, mean + std,
                                color=st["color"], alpha=0.15, lw=0)
        for ax, ylabel in ((ax_mae, "MAE (clipped, cycles)"),
                           (ax_bias, "bias = mean(pred − true) (cycles)")):
            if np.isinf(bins[-1][1]):  # saturation regime marker
                ax.axvspan(len(bins) - 1.5, len(bins) - 0.5, color="#888888", alpha=0.12)
                ax.annotate("saturation regime\n(labels clipped)",
                            (len(bins) - 1, ax.get_ylim()[1]), ha="center", va="top",
                            fontsize=7, color="#555555")
            ax.set_xticks(centers)
            ax.set_xticklabels([_bin_label(lo, "inf" if np.isinf(hi) else hi)
                                for lo, hi in bins])
            ax.set_xlabel("true RUL at prediction time (cycles)")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
        ax_bias.axhline(0, color="#444444", lw=1)
        ax_mae.legend(fontsize=8, framealpha=0.9)
        fig.suptitle(f"Error vs. prediction horizon "
                     f"({ds}, trained on {n_units} units, label cap {max_rul})")
        saved += _save(fig, out_dir, f"horizon_{ds}_mr{max_rul}_n{n_units}", prefix)
        plt.show() if show else plt.close(fig)
    return saved


def plot_horizon_trajectories(
    preds_csv: str | Path,
    out_dir: str | Path,
    models: Optional[list[str]] = None,
    n_units: Optional[int] = None,
    seed: int = 0,
    max_units_shown: int = 4,
    max_rul: Optional[float] = None,
    dataset: Optional[str] = None,
    show: bool = True,
    prefix: str = "",
) -> list[Path]:
    """Predicted vs. true RUL along a few test-unit trajectories (the qualitative
    view of far-end behavior: does the prediction track the truth or flatline?).
    ``max_rul`` selects the cap arm when the predictions file carries several
    (CHANGES.md §18) and draws the cap line -- predictions cannot exceed it, so
    against the UNCLIPPED truth line everything above the cap is unreachable."""
    import csv as _csv
    out_dir = Path(out_dir)
    rows = []
    with open(preds_csv, newline="") as f:
        for r in _csv.DictReader(f):
            rows.append({"model": r["model"], "loss": r["loss"],
                         "dataset": r.get("dataset"),
                         "max_rul": float(r["max_rul"]) if r.get("max_rul") else None,
                         "n_units": int(r["n_units"]), "seed": int(r["seed"]),
                         "unit": int(r["unit"]), "true": float(r["true_rul"]),
                         "pred": float(r["pred"])})
    datasets = sorted({r["dataset"] for r in rows if r["dataset"] is not None})
    if dataset is not None and datasets:
        rows = [r for r in rows if r["dataset"] == dataset]
        if not rows:
            raise ValueError(f"no rows for dataset={dataset}; file has {datasets}")
    elif len(datasets) > 1:
        raise ValueError(f"predictions file mixes datasets {datasets}; pass "
                         f"dataset= to select one (unit IDs collide across datasets)")
    caps = sorted({r["max_rul"] for r in rows if r["max_rul"] is not None})
    if max_rul is not None and caps:
        rows = [r for r in rows if r["max_rul"] == float(max_rul)]
        if not rows:
            raise ValueError(f"no rows with max_rul={max_rul}; file has caps {caps}")
    elif len(caps) > 1:
        raise ValueError(f"predictions file mixes label caps {caps}; pass max_rul= "
                         f"to select one arm")
    # Pick an AVAILABLE (n_units, seed) instead of assuming seed 0 / max exist.
    # horizon_predictions.csv only carries cells the run actually (re)emitted, so a
    # restart that skipped "done" cells (e.g. horizon.csv kept but predictions
    # archived) may lack seed 0 -- fall back with a note rather than crashing.
    avail_units = sorted({r["n_units"] for r in rows})
    if n_units is None:
        n_units = max(avail_units)
    elif n_units not in avail_units:
        raise ValueError(f"no predictions for n_units={n_units}; file has "
                         f"{avail_units}. Rerun the horizon eval for that unit count.")
    seeds_here = sorted({r["seed"] for r in rows if r["n_units"] == n_units})
    if not seeds_here:
        raise ValueError(f"no prediction rows for n_units={n_units}.")
    if seed not in seeds_here:
        alt = seeds_here[0]
        print(f"[plot_horizon_trajectories] seed {seed} absent for n_units="
              f"{n_units} (present: {seeds_here}); using seed {alt}. This usually "
              f"means the horizon run skipped seed {seed} as already-done while its "
              f"predictions were archived -- archive horizon.csv and "
              f"horizon_predictions.csv TOGETHER, or rerun to regenerate all seeds.")
        seed = alt
    rows = [r for r in rows if r["n_units"] == n_units and r["seed"] == seed]
    arms = sorted({(r["model"], r["loss"]) for r in rows})
    if models:
        arms = [a for a in arms if a[0] in models]
    # longest test units are the most informative far-end examples
    lengths: dict[int, int] = {}
    for r in rows:
        lengths[r["unit"]] = lengths.get(r["unit"], 0) + 1
    units = sorted(sorted(lengths, key=lengths.get, reverse=True)[:max_units_shown])

    fig, axes = plt.subplots(1, len(units), figsize=(4.2 * len(units), 4),
                             sharey=True, squeeze=False)
    for ax, unit in zip(axes[0], units):
        drew_truth = False
        for model, loss in arms:
            pts = sorted([(r["true"], r["pred"]) for r in rows
                          if r["unit"] == unit and r["model"] == model and r["loss"] == loss],
                         reverse=True)
            if not pts:
                continue
            true = np.array([p[0] for p in pts])
            pred = np.array([p[1] for p in pts])
            x = -true  # time axis: cycles-to-failure counting down, left -> right
            if not drew_truth:
                ax.plot(x, true, color="#444444", lw=1.2, ls="--", label="true RUL")
                drew_truth = True
            label = model if loss in ("", "native") else f"{model}[{loss}]"
            ax.plot(x, pred, lw=1.6, alpha=0.9, label=label,
                    color=_series_style(label)["color"])
        if max_rul is not None:
            ax.axhline(max_rul, color="#888888", ls=":", lw=1,
                       label=f"label cap ({max_rul:g})")
        ax.set_title(f"test unit {unit}")
        ax.set_xlabel("−(true RUL)  → failure at 0")
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel("RUL (cycles)")
    axes[0][0].legend(fontsize=8, framealpha=0.9)
    fig.suptitle(f"Prediction trajectories (trained on {n_units} units, seed {seed})")
    ds_tag = f"_{dataset}" if dataset else (f"_{datasets[0]}" if datasets else "")
    cap_tag = f"_mr{int(max_rul)}" if max_rul is not None else ""
    saved = _save(fig, out_dir,
                  f"horizon_trajectories{ds_tag}{cap_tag}_n{n_units}_seed{seed}", prefix)
    plt.show() if show else plt.close(fig)
    return saved


def plot_transfer(
    transfer_csv: str | Path,
    out_dir: str | Path,
    metric: str = "rmse_clipped",
    show: bool = True,
    prefix: str = "",
) -> list[Path]:
    """Cold-start curve: ``metric`` on the TARGET test set vs. number of target
    failures used. zero_shot arms are horizontal reference lines (they use no
    target units); target_only vs source+target separate by linestyle."""
    out_dir = Path(out_dir)
    rows = load_results(transfer_csv)
    src = rows[0]["source_dataset"] if rows else "?"
    tgt = rows[0]["target_dataset"] if rows else "?"
    series: dict[tuple[str, str, str], dict[int, list[float]]] = {}
    for r in rows:
        key = (r["model"], r["loss"], r["mode"])
        series.setdefault(key, {}).setdefault(int(r["n_target_units"]), []).append(float(r[metric]))

    mode_ls = {"target_only": "-", "source+target": "--"}
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ks: set[int] = set()
    for (model, loss, mode), by_k in sorted(series.items()):
        label_base = model if loss in ("", "native") else f"{model}[{loss}]"
        color = _series_style(label_base)["color"]
        if mode == "zero_shot":
            v = [x for vals in by_k.values() for x in vals]
            ax.axhline(np.mean(v), color=color, ls=":", lw=1.6,
                       label=f"{label_base} zero-shot (source-only)")
            continue
        kk = np.array(sorted(by_k))
        ks.update(int(k) for k in kk)
        mean = np.array([np.mean(by_k[k]) for k in kk])
        std = np.array([np.std(by_k[k]) for k in kk])
        ax.plot(kk, mean, marker="o", ms=5, lw=2, color=color, ls=mode_ls.get(mode, "-"),
                label=f"{label_base} {mode}")
        ax.fill_between(kk, mean - std, mean + std, color=color, alpha=0.12, lw=0)
    if ks:
        _unit_count_xaxis(ax, ks)
    ax.set_xlabel(f"target failures used (units of {tgt})")
    ax.set_ylabel(metric)
    ax.set_title(f"Cold-start transfer: {src} → {tgt}")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, framealpha=0.9)
    saved = _save(fig, out_dir, f"transfer_{src}_to_{tgt}_{metric}", prefix)
    plt.show() if show else plt.close(fig)
    return saved


_CURVE_STEM = re.compile(r"n(?P<n>\d+)_seed(?P<seed>\d+)_(?P<loss>[a-z]+)$")


def plot_learning_curves(
    curves_dir: str | Path,
    out_dir: str | Path,
    metric: str = "val_rmse",
    losses: Optional[list[str]] = None,
    show: bool = True,
    prefix: str = "",
) -> list[Path]:
    """Validation-RMSE learning curves, one panel per loss arm. Curves are colored
    by training-unit count (sequential: darker = more units); seeds share a color.
    Replaces the unreadable 60-entry flat legend with a 6-entry per-n legend."""
    out_dir = Path(out_dir)
    files = sorted(Path(curves_dir).glob("*.csv"))
    groups: dict[tuple[str, int], list[Path]] = {}
    for f in files:
        m = _CURVE_STEM.search(f.stem)
        if m:
            groups.setdefault((m.group("loss"), int(m.group("n"))), []).append(f)
    if not groups:
        raise FileNotFoundError(f"no parseable learning-curve CSVs in {curves_dir}")
    losses = losses or sorted({loss for loss, _ in groups})
    n_values = sorted({n for _, n in groups})
    cmap = plt.get_cmap("Blues")
    shade = {n: cmap(0.35 + 0.6 * i / max(1, len(n_values) - 1))
             for i, n in enumerate(n_values)}

    fig, axes = plt.subplots(1, len(losses), figsize=(6.5 * len(losses), 4.5),
                             sharey=True, squeeze=False)
    for ax, loss in zip(axes[0], losses):
        for n in n_values:
            for j, f in enumerate(groups.get((loss, n), [])):
                xs, ys = load_learning_curve(f)[metric]
                ax.plot(xs, ys, color=shade[n], alpha=0.8, lw=1.4,
                        label=f"{n} units" if j == 0 else None)
        ax.set_title(f"loss = {loss}")
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, title="training units", framealpha=0.9)
    axes[0][0].set_ylabel(metric)
    fig.suptitle(f"Learning curves ({metric})")
    saved = _save(fig, out_dir, f"learning_curves_{metric}", prefix)
    plt.show() if show else plt.close(fig)
    return saved


# ---------------------------------------------------------------------------
# v2 "when do TSFMs work" figures (CHANGES.md §35-§37)
# ---------------------------------------------------------------------------
def _cond_sort_key(c):
    """Sort conditions that may be ints (unit counts) or strings (factor levels):
    numerics first (ascending), then strings (lexicographic)."""
    try:
        return (0, float(c), "")
    except (TypeError, ValueError):
        return (1, 0.0, str(c))


def plot_success_map(
    success_table: list,
    out_dir: str | Path,
    condition_field: Optional[str] = None,
    show: bool = True,
    prefix: str = "",
) -> list[Path]:
    """The headline figure: a win/tie/loss/hollow heatmap of models x conditions,
    faceted per dataset (``scoring.success_map`` rows in, one PNG/PDF per dataset out).
    ``condition_field`` is the column on the x-axis (auto: ``level`` for a probe table,
    else ``n_units``). Each cell is colored by its verdict and annotated W/T/L/H."""
    from matplotlib.colors import ListedColormap

    out_dir = Path(out_dir)
    rows = list(success_table)
    if not rows:
        raise ValueError("plot_success_map got an empty success table")
    if condition_field is None:
        condition_field = "level" if "level" in rows[0] else "n_units"
    cmap = ListedColormap([_VERDICT_COLORS[v] for v in _VERDICT_ORDER]
                          ).with_extremes(bad="#FFFFFF")
    datasets = sorted({r.get("dataset", "") for r in rows}, key=lambda s: str(s))
    saved: list[Path] = []
    for ds in datasets:
        sub = [r for r in rows if r.get("dataset", "") == ds]
        models = sorted({r["model"] for r in sub})
        conds = sorted({r[condition_field] for r in sub}, key=_cond_sort_key)
        mi = {m: i for i, m in enumerate(models)}
        ci = {c: j for j, c in enumerate(conds)}
        grid = np.full((len(models), len(conds)), np.nan)
        for r in sub:
            grid[mi[r["model"]], ci[r[condition_field]]] = _VERDICT_CODE[r["verdict"]]
        fig, ax = plt.subplots(figsize=(1.4 * len(conds) + 3, 0.6 * len(models) + 2))
        ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=-0.5, vmax=3.5, aspect="auto")
        for r in sub:
            ax.text(ci[r[condition_field]], mi[r["model"]], _VERDICT_INITIAL[r["verdict"]],
                    ha="center", va="center", color="white", fontweight="bold", fontsize=9)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([str(c) for c in conds])
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)
        ax.set_xlabel(condition_field)
        ax.set_title(f"Success map ({ds or 'unknown dataset'})")
        handles = [plt.matplotlib.patches.Patch(color=_VERDICT_COLORS[v], label=v)
                   for v in ("win", "tie", "loss", "hollow")]
        ax.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
        ds_tag = f"{ds}_" if len(datasets) > 1 and ds else ""
        saved += _save(fig, out_dir, f"success_map_{ds_tag}{condition_field}", prefix)
        plt.show() if show else plt.close(fig)
    return saved


def _read_csv_rows(path: str | Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(_csv.DictReader(f))


def plot_earliness(
    earliness_csv: str | Path,
    out_dir: str | Path,
    show: bool = True,
    prefix: str = "",
) -> list[Path]:
    """Two-sided earliness: per model, the mean fraction of predictions that are
    dangerously LATE (d >= 0) vs wastefully EARLY (d < 0), averaged over cells/seeds
    (``earliness.csv`` from ``horizon.run_earliness``)."""
    out_dir = Path(out_dir)
    # frac_late/frac_early are constant within a cell -> dedupe by cell before averaging
    seen: dict[tuple, tuple[float, float]] = {}
    for r in _read_csv_rows(earliness_csv):
        key = (r["model"], r.get("n_units"), r.get("seed"), r.get("loss"))
        seen[key] = (float(r["frac_late"]), float(r["frac_early"]))
    per_model: dict[str, list[tuple[float, float]]] = {}
    for (model, *_), fracs in seen.items():
        per_model.setdefault(model, []).append(fracs)
    models = sorted(per_model)
    late = [float(np.mean([f[0] for f in per_model[m]])) for m in models]
    early = [float(np.mean([f[1] for f in per_model[m]])) for m in models]
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(1.3 * len(models) + 3, 4.5))
    ax.bar(x - 0.2, late, 0.4, color="#D55E00", label="dangerously late (d ≥ 0)")
    ax.bar(x + 0.2, early, 0.4, color="#0072B2", label="wastefully early (d < 0)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("fraction of predictions")
    ax.set_title("Earliness: dangerously late vs. wastefully early")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8, framealpha=0.9)
    saved = _save(fig, out_dir, "earliness", prefix)
    plt.show() if show else plt.close(fig)
    return saved


def plot_cost_curve(
    cost_curve_csv: str | Path,
    out_dir: str | Path,
    show: bool = True,
    prefix: str = "",
) -> list[Path]:
    """Maintenance cost vs. the late:early cost ratio, one line per model (mean over
    seeds/cells), log-log -- the "no single arbitrary ratio" view (``cost_curve.csv``)."""
    out_dir = Path(out_dir)
    series: dict[str, dict[float, list[float]]] = {}
    for r in _read_csv_rows(cost_curve_csv):
        series.setdefault(r["model"], {}).setdefault(
            float(r["cost_ratio"]), []).append(float(r["cost"]))
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for model in sorted(series):
        by_ratio = series[model]
        ratios = np.array(sorted(by_ratio))
        mean = np.array([np.mean(by_ratio[r]) for r in ratios])
        ax.plot(ratios, mean, marker="o", lw=2, ms=5, label=model,
                color=_series_style(model)["color"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("late-cost : early-cost ratio")
    ax.set_ylabel("total cost")
    ax.set_title("Cost curve (late-vs-early trade-off swept)")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8, framealpha=0.9)
    saved = _save(fig, out_dir, "cost_curve", prefix)
    plt.show() if show else plt.close(fig)
    return saved


def plot_cross_tsfm(
    fairness_csv: str | Path,
    out_dir: str | Path,
    metric: str = "rmse_clipped",
    show: bool = True,
    prefix: str = "",
) -> list[Path]:
    """The RQ-M five-model comparison: ``metric`` (mean +/- std over seeds) per model in
    its NATIVE aggregation vs. the COMMON mean-pooled representation, so the ranking can
    be checked for aggregation artifacts (``representation_fairness.csv``)."""
    out_dir = Path(out_dir)
    rows = load_results(fairness_csv)
    series: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        series.setdefault((r["model"], r.get("mode", "native")), []).append(float(r[metric]))
    models = sorted({m for m, _ in series})
    modes = ["native", "common"]
    mode_color = {"native": "#0072B2", "common": "#E69F00"}
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(1.5 * len(models) + 3, 4.8))
    for k, mode in enumerate(modes):
        means = [float(np.mean(series[(m, mode)])) if (m, mode) in series else np.nan
                 for m in models]
        stds = [float(np.std(series[(m, mode)])) if (m, mode) in series else 0.0
                for m in models]
        ax.bar(x + (k - 0.5) * 0.4, means, 0.4, yerr=stds, capsize=3,
               color=mode_color[mode], label=mode)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel(metric)
    ax.set_title("Cross-TSFM: native vs. common representation (RQ-M)")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8, framealpha=0.9, title="aggregation")
    saved = _save(fig, out_dir, f"cross_tsfm_{metric}", prefix)
    plt.show() if show else plt.close(fig)
    return saved
