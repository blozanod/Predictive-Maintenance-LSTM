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

import re
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from .evaluate import aggregate_data_scaling, load_learning_curve, load_results

# Okabe-Ito, fixed assignment per model family (never cycled).
_FAMILY_STYLE = {
    "chronos-2_mlp": dict(color="#0072B2", marker="o"),
    "gbm": dict(color="#E69F00", marker="s"),
    "lstm": dict(color="#009E73", marker="^"),
    "cnn": dict(color="#D55E00", marker="v"),
    "minirocket": dict(color="#CC79A7", marker="D"),
}
_FALLBACK_COLORS = ["#56B4E9", "#F0E442", "#000000"]
_LOSS_LINESTYLE = {"mse": "-", "corn": "--", "quantile": ":", "native": "-", "": "-"}
_FLOOR_MODEL = "predict_mean"


def _series_style(label: str) -> dict:
    """Style for an ``aggregate_data_scaling`` label (``model`` or ``model[loss]``)."""
    m = re.fullmatch(r"(.+?)\[(.+)\]", label)
    family, loss = (m.group(1), m.group(2)) if m else (label, "native")
    style = _FAMILY_STYLE.get(family)
    if style is None:  # unknown family: deterministic fallback, no cycling
        idx = sum(family.encode()) % len(_FALLBACK_COLORS)
        style = dict(color=_FALLBACK_COLORS[idx], marker="x")
    return dict(style, ls=_LOSS_LINESTYLE.get(loss, "-"))


def _save(fig, out_dir: Path, name: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext, kw in (("png", {"dpi": 300}), ("pdf", {})):
        p = out_dir / f"{name}.{ext}"
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
) -> list[Path]:
    """The headline figure(s): metric vs. training units, mean +/- std over seeds.

    One figure per metric in ``metrics`` (``(column, ylabel)`` pairs). Returns the
    saved file paths.
    """
    out_dir = Path(out_dir)
    metrics = metrics or [
        ("rmse_clipped", "test RMSE (clipped protocol)"),
        ("rmse_unclipped", "test RMSE (unclipped protocol)"),
        ("nasa_clipped", "NASA score (clipped protocol)"),
    ]
    saved: list[Path] = []
    for metric, ylabel in metrics:
        agg = aggregate_data_scaling(results_csv, metric=metric)
        log_y = metric.startswith("nasa")
        fig, ax = plt.subplots(figsize=(7.5, 5))
        lo, hi, all_ns = np.inf, -np.inf, set()
        for label in sorted(agg):
            ns, mean, std = agg[label]
            all_ns.update(int(n) for n in ns)
            if label.startswith(_FLOOR_MODEL):
                ax.axhline(float(np.mean(mean)), color="#888888", ls=":", lw=1.2,
                           label="predict-mean floor")
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
        ax.set_title(f"Data-scaling curve (FD001) — {ylabel}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, framealpha=0.9)
        saved += _save(fig, out_dir, f"data_scaling_{metric}")
        plt.show() if show else plt.close(fig)
    return saved


def plot_ablation(
    ablation_csv: str | Path,
    out_dir: str | Path,
    metric: str = "rmse_clipped",
    show: bool = True,
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
    saved = _save(fig, out_dir, f"ablation_{metric}")
    plt.show() if show else plt.close(fig)
    return saved


_CURVE_STEM = re.compile(r"n(?P<n>\d+)_seed(?P<seed>\d+)_(?P<loss>[a-z]+)$")


def plot_learning_curves(
    curves_dir: str | Path,
    out_dir: str | Path,
    metric: str = "val_rmse",
    losses: Optional[list[str]] = None,
    show: bool = True,
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
    saved = _save(fig, out_dir, f"learning_curves_{metric}")
    plt.show() if show else plt.close(fig)
    return saved
