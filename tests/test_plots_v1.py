"""The v1 Stage-C figures (`src/plots.py`): ablation, horizon, trajectories, transfer
and learning curves.

Same contract as `tests/test_plots_v2.py`: Agg backend, tiny synthetic CSVs, `show=False`,
`tmp_path` outputs, assert the files exist -- plus the fail-loud branches each renderer
guards (mixed datasets / mixed label caps / missing unit counts / no parseable curves),
which are the ones a notebook actually trips over.
"""

from __future__ import annotations

import csv as _csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

from src.evaluate import append_result_row
from src.plots import (_bin_label, _series_style, plot_ablation, plot_horizon,
                       plot_horizon_trajectories, plot_learning_curves,
                       plot_success_map, plot_transfer)


# ---------------------------------------------------------------------------
# style helpers
# ---------------------------------------------------------------------------
def test_series_style_known_family_and_deterministic_fallback():
    """A registered family keeps its fixed color/marker; an unknown one gets a
    deterministic fallback (never a cycling color), and the loss tag sets the linestyle."""
    gbm = _series_style("gbm")
    assert gbm["color"] == "#E69F00" and gbm["marker"] == "s"
    assert _series_style("chronos-2_mlp[corn]")["ls"] == "--"     # loss-derived linestyle
    assert _series_style("gbm_age")["ls"] == "--"                 # family's own ls wins
    a, b = _series_style("brand-new-model"), _series_style("brand-new-model")
    assert a == b and a["marker"] == "x"                          # deterministic fallback


def test_bin_label_variants():
    assert _bin_label("all", "all") == "all"
    assert _bin_label(100.0, "inf") == "≥100"
    assert _bin_label(0.0, 25.0) == "0–25"


# ---------------------------------------------------------------------------
# plot_ablation
# ---------------------------------------------------------------------------
def _write_ablation(path: Path) -> None:
    """Two head_features at the default pooling over two contexts, one pooling variant,
    and one feature set that exists ONLY at a non-default pooling (the skip branch)."""
    for feat in ("emb", "emb+locscale"):
        for ctx in (12, 24):
            for seed in (0, 1):
                append_result_row(path, {
                    "model": "chronos-2_mlp", "dataset": "FD001", "n_units": 8,
                    "seed": seed, "loss": "mse", "tsfm_context_length": ctx,
                    "head_features": feat, "pooling": "forecast_token",
                    "rmse_clipped": 20.0 - ctx * 0.1 + seed})
    for seed in (0, 1):     # pooling variant at the best cell
        append_result_row(path, {
            "model": "chronos-2_mlp", "dataset": "FD001", "n_units": 8, "seed": seed,
            "loss": "mse", "tsfm_context_length": 24, "head_features": "emb+locscale",
            "pooling": "mean", "rmse_clipped": 15.0 + seed})
        # a feature set that never ran at the default pooling -> skipped in phase 1
        append_result_row(path, {
            "model": "chronos-2_mlp", "dataset": "FD001", "n_units": 8, "seed": seed,
            "loss": "mse", "tsfm_context_length": 24, "head_features": "emb+locscale+raw",
            "pooling": "last_content", "rmse_clipped": 16.0 + seed})


def test_plot_ablation(tmp_path):
    csv_path = tmp_path / "ablation.csv"
    _write_ablation(csv_path)
    saved = plot_ablation(csv_path, tmp_path / "figs", show=False, prefix="exp_")
    names = {p.name for p in saved}
    assert names == {"exp_ablation_rmse_clipped.png", "exp_ablation_rmse_clipped.pdf"}
    assert all(p.exists() for p in saved)


# ---------------------------------------------------------------------------
# plot_horizon
# ---------------------------------------------------------------------------
def _write_horizon(path: Path, max_rul: int, bins, dataset: str = "FD001") -> None:
    for model, loss in (("chronos-2_mlp", "mse"), ("gbm", "native")):
        for seed in (0, 1):
            for lo, hi in bins:
                append_result_row(path, {
                    "schema_version": 2, "model": model, "n_units": 8, "seed": seed,
                    "loss": loss, "dataset": dataset, "max_rul": max_rul,
                    "window_size": 12, "tsfm_context_length": 12,
                    "head_features": "emb", "pooling": "mean",
                    "bin_lo": lo, "bin_hi": hi, "n_bin": 5,
                    "rmse_clipped": 10.0 + seed, "mae_clipped": 8.0 + seed,
                    "bias": -1.0 + seed, "nasa_mean": 2.0})
            append_result_row(path, {                    # the 'all' row is filtered out
                "schema_version": 2, "model": model, "n_units": 8, "seed": seed,
                "loss": loss, "dataset": dataset, "max_rul": max_rul,
                "window_size": 12, "tsfm_context_length": 12, "head_features": "emb",
                "pooling": "mean", "bin_lo": "all", "bin_hi": "all", "n_bin": 20,
                "rmse_clipped": 10.0, "mae_clipped": 8.0, "bias": -1.0, "nasa_mean": 2.0})


def test_plot_horizon_saturation_and_closed_bins(tmp_path):
    """One arm ends in the >=max_rul saturation bin (shaded), the other is fully
    closed -- one figure per (dataset, cap, n_units) arm."""
    csv_path = tmp_path / "horizon.csv"
    _write_horizon(csv_path, 40, [(0.0, 20.0), (20.0, "inf")])
    _write_horizon(csv_path, 60, [(0.0, 20.0), (20.0, 40.0)])
    saved = plot_horizon(csv_path, tmp_path / "figs", show=False)
    names = {p.name for p in saved}
    assert "horizon_FD001_mr40_n8.png" in names
    assert "horizon_FD001_mr60_n8.pdf" in names
    assert all(p.exists() for p in saved)


# ---------------------------------------------------------------------------
# plot_horizon_trajectories
# ---------------------------------------------------------------------------
PRED_FIELDS = ["model", "dataset", "max_rul", "n_units", "seed", "loss", "unit",
               "true_rul", "pred"]


def _write_preds(path: Path, rows: list[dict], fields=None) -> Path:
    fields = fields or PRED_FIELDS
    with open(path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def _traj_rows(dataset="FD001", max_rul=40, n_units=8, seed=0, models=("chronos-2_mlp",),
               units=(1, 2, 3), loss="mse"):
    rows = []
    for model in models:
        for unit in units:
            for true in range(30, 0, -3):
                rows.append({"model": model, "dataset": dataset, "max_rul": max_rul,
                             "n_units": n_units, "seed": seed, "loss": loss,
                             "unit": unit, "true_rul": float(true),
                             "pred": float(min(true + 2, max_rul))})
    return rows


def test_plot_horizon_trajectories(tmp_path):
    """Happy path: an explicit cap draws the cap line, `models=` filters the arms, and
    a model missing from one unit is skipped rather than crashing."""
    rows = _traj_rows(models=("chronos-2_mlp",))
    rows += _traj_rows(models=("gbm",), units=(1,), loss="native")   # unit 2/3: no pts
    csv_path = _write_preds(tmp_path / "preds.csv", rows)
    saved = plot_horizon_trajectories(csv_path, tmp_path / "figs", max_rul=40,
                                      n_units=8, seed=0, max_units_shown=3,
                                      models=["chronos-2_mlp", "gbm"], show=False)
    assert {p.suffix for p in saved} == {".png", ".pdf"}
    assert all(p.exists() for p in saved)
    assert any("horizon_trajectories_FD001_mr40_n8_seed0" in p.name for p in saved)


def test_plot_horizon_trajectories_without_dataset_or_cap_columns(tmp_path):
    """A minimal predictions file (no dataset / no cap) still renders: no cap line, no
    dataset tag, and the unit count defaults to the largest available."""
    rows = [{k: v for k, v in r.items() if k not in ("dataset", "max_rul")}
            for r in _traj_rows()]
    csv_path = _write_preds(tmp_path / "bare.csv", rows,
                            fields=[f for f in PRED_FIELDS
                                    if f not in ("dataset", "max_rul")])
    saved = plot_horizon_trajectories(csv_path, tmp_path / "figs", show=False)
    assert all(p.exists() for p in saved)
    assert any(p.name.startswith("horizon_trajectories_n8_seed0") for p in saved)


def test_plot_horizon_trajectories_seed_fallback(tmp_path, capsys):
    """A missing seed falls back to an available one WITH a printed explanation
    (archived predictions, CHANGES.md §20) instead of raising."""
    csv_path = _write_preds(tmp_path / "preds.csv", _traj_rows(seed=3))
    plot_horizon_trajectories(csv_path, tmp_path / "figs", seed=0, show=False)
    assert "using seed 3" in capsys.readouterr().out


def test_plot_horizon_trajectories_selects_and_guards(tmp_path):
    mixed = _traj_rows(dataset="FD001") + _traj_rows(dataset="FD003")
    csv_path = _write_preds(tmp_path / "mixed.csv", mixed)
    with pytest.raises(ValueError, match="mixes datasets"):
        plot_horizon_trajectories(csv_path, tmp_path / "figs", show=False)
    with pytest.raises(ValueError, match="no rows for dataset"):
        plot_horizon_trajectories(csv_path, tmp_path / "figs", dataset="FD002", show=False)
    saved = plot_horizon_trajectories(csv_path, tmp_path / "figs", dataset="FD003",
                                      show=False)
    assert any("_FD003_" in p.name for p in saved)

    caps = _traj_rows(max_rul=40) + _traj_rows(max_rul=60)
    caps_csv = _write_preds(tmp_path / "caps.csv", caps)
    with pytest.raises(ValueError, match="mixes label caps"):
        plot_horizon_trajectories(caps_csv, tmp_path / "figs", show=False)
    with pytest.raises(ValueError, match="no rows with max_rul"):
        plot_horizon_trajectories(caps_csv, tmp_path / "figs", max_rul=99, show=False)

    one = _write_preds(tmp_path / "one.csv", _traj_rows())
    with pytest.raises(ValueError, match="no predictions for n_units"):
        plot_horizon_trajectories(one, tmp_path / "figs", n_units=99, show=False)


def test_plot_horizon_trajectories_empty_file_raises(tmp_path):
    """A header-only predictions file fails loud instead of erroring inside max()."""
    empty = _write_preds(tmp_path / "empty.csv", [])
    with pytest.raises(ValueError, match="no prediction rows"):
        plot_horizon_trajectories(empty, tmp_path / "figs", show=False)


# ---------------------------------------------------------------------------
# plot_transfer
# ---------------------------------------------------------------------------
def _write_transfer(path: Path, modes=("zero_shot", "target_only", "source+target")):
    for mode in modes:
        ks = [0] if mode == "zero_shot" else [2, 4]
        for k in ks:
            for seed in (0, 1):
                for model, loss in (("chronos-2_mlp", "mse"), ("gbm", "native")):
                    append_result_row(path, {
                        "schema_version": 2, "model": model, "mode": mode,
                        "source_dataset": "FD001", "target_dataset": "FD003",
                        "n_target_units": k, "seed": seed, "loss": loss,
                        "max_rul": 40, "window_size": 12, "tsfm_context_length": 12,
                        "head_features": "emb", "pooling": "mean",
                        "rmse_clipped": 20.0 - k + seed, "mae_clipped": 15.0,
                        "nasa_clipped": 300.0, "rmse_unclipped": 22.0,
                        "mae_unclipped": 17.0, "nasa_unclipped": 400.0, "n": 5})


def test_plot_transfer_all_modes(tmp_path):
    csv_path = tmp_path / "transfer.csv"
    _write_transfer(csv_path)
    saved = plot_transfer(csv_path, tmp_path / "figs", show=False, prefix="exp_")
    assert any(p.name == "exp_transfer_FD001_to_FD003_rmse_clipped.png" for p in saved)
    assert all(p.exists() for p in saved)


def test_plot_transfer_zero_shot_only_and_empty(tmp_path):
    """Only zero-shot arms => no shot axis to draw; an empty file still renders a
    (labelled '?') figure rather than raising."""
    zs = tmp_path / "zs.csv"
    _write_transfer(zs, modes=("zero_shot",))
    assert all(p.exists() for p in plot_transfer(zs, tmp_path / "figs", show=False))

    empty = tmp_path / "empty_transfer.csv"
    with open(empty, "w", newline="") as f:
        _csv.writer(f).writerow(["model", "loss", "mode", "n_target_units",
                                 "source_dataset", "target_dataset", "rmse_clipped"])
    saved = plot_transfer(empty, tmp_path / "figs", show=False)
    assert any("transfer_?_to_?_" in p.name for p in saved)


# ---------------------------------------------------------------------------
# plot_learning_curves
# ---------------------------------------------------------------------------
def _write_curve(path: Path, n_epochs: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["step", "epoch", "metric", "value"])
        for e in range(n_epochs):
            w.writerow([e, e, "train_loss", 1.0 / (e + 1)])
            w.writerow(["", e, "val_loss", 0.9 / (e + 1)])     # no step -> epoch is x
            w.writerow(["", e, "val_rmse", 20.0 - e])


def test_plot_learning_curves(tmp_path):
    """One panel per loss arm, curves shaded by unit count; a (loss, n) combination
    that never ran is simply absent from its panel."""
    curves = tmp_path / "curves"
    for n in (2, 4):
        for seed in (0, 1):
            _write_curve(curves / f"chronos-2_mlp_n{n}_seed{seed}_mse.csv")
    _write_curve(curves / "chronos-2_mlp_n4_seed0_corn.csv")   # corn only at n=4
    (curves / "not_a_curve.csv").write_text("x\n")             # unparseable stem: ignored

    saved = plot_learning_curves(curves, tmp_path / "figs", show=False)
    assert any(p.name == "learning_curves_val_rmse.png" for p in saved)
    assert all(p.exists() for p in saved)
    # an explicit loss list restricts the panels
    saved2 = plot_learning_curves(curves, tmp_path / "figs", metric="train_loss",
                                  losses=["mse"], show=False)
    assert any("learning_curves_train_loss" in p.name for p in saved2)


def test_plot_learning_curves_requires_parseable_files(tmp_path):
    empty = tmp_path / "curves_empty"
    empty.mkdir()
    (empty / "whatever.csv").write_text("x\n")
    with pytest.raises(FileNotFoundError, match="no parseable learning-curve CSVs"):
        plot_learning_curves(empty, tmp_path / "figs", show=False)


# ---------------------------------------------------------------------------
# plot_success_map: explicit condition field (the auto-detect branch's twin)
# ---------------------------------------------------------------------------
def test_plot_success_map_explicit_condition_field(tmp_path):
    table = [{"dataset": "", "model": m, "arm": arm, "verdict": v}
             for m, arm, v in (("chronos-2_zeroshot · nasa_clipped", "FD001", "win"),
                               ("chronos-2_zeroshot · rmse_clipped", "FD001", "tie"))]
    saved = plot_success_map(table, tmp_path / "figs", condition_field="arm",
                             prefix="rq_z_", show=False)
    assert any(p.name == "rq_z_success_map_arm.png" for p in saved)
    assert all(p.exists() for p in saved)
    assert np.isfinite(len(saved))
