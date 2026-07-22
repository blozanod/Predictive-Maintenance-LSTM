"""The v2 reporting figures (§35-§37): each must render PNG+PDF from tiny synthetic
inputs without displaying (Agg backend, show=False)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pytest

from src.evaluate import append_result_row
from src.plots import (plot_success_map, plot_earliness, plot_cost_curve,
                       plot_cross_tsfm)


def test_plot_success_map_facets_by_dataset(tmp_path):
    table = []
    for ds in ("FD001", "FD003"):
        for model in ("chronos-2_mlp", "moirai-2_mlp"):
            for n_units, verdict in ((10, "loss"), (50, "tie"), (100, "win")):
                table.append({"dataset": ds, "model": model, "n_units": n_units,
                              "verdict": verdict})
    table.append({"dataset": "FD001", "model": "chronos-2_mlp", "n_units": 25,
                  "verdict": "hollow"})
    saved = plot_success_map(table, tmp_path / "figs", show=False)
    names = {p.name for p in saved}
    assert "success_map_FD001_n_units.png" in names
    assert "success_map_FD003_n_units.pdf" in names
    assert all(p.exists() for p in saved)


def test_plot_success_map_probe_condition_is_level(tmp_path):
    table = [{"dataset": "FD001", "model": "chronos-2_mlp", "factor": "channels",
              "level": lv, "n_units": 100, "verdict": v}
             for lv, v in (("all", "win"), ("few", "tie"))]
    saved = plot_success_map(table, tmp_path / "figs", show=False)
    assert any("success_map_level" in p.name for p in saved)


def test_plot_success_map_empty_raises(tmp_path):
    with pytest.raises(ValueError, match="empty success table"):
        plot_success_map([], tmp_path / "figs", show=False)


def _write_earliness(path: Path):
    for model in ("chronos-2_mlp", "gbm"):
        for seed in (0, 1):
            for lo, hi, side in ((-25, 0, "early"), (0, 25, "late")):
                append_result_row(path, {
                    "model": model, "dataset": "FD001", "max_rul": 125,
                    "n_units": 100, "seed": seed, "loss": "mse",
                    "lo": lo, "hi": hi, "side": side, "n_bin": 5, "frac": 0.5,
                    "frac_late": 0.4, "frac_early": 0.6, "mean_signed_error": -1.0})


def test_plot_earliness(tmp_path):
    csv_path = tmp_path / "earliness.csv"
    _write_earliness(csv_path)
    saved = plot_earliness(csv_path, tmp_path / "figs", show=False)
    assert {p.suffix for p in saved} == {".png", ".pdf"}
    assert all(p.exists() for p in saved)


def test_plot_cost_curve(tmp_path):
    csv_path = tmp_path / "cost_curve.csv"
    for model in ("chronos-2_mlp", "gbm"):
        for seed in (0, 1):
            for ratio in (1.0, 10.0, 100.0):
                append_result_row(csv_path, {
                    "model": model, "dataset": "FD001", "max_rul": 125, "n_units": 100,
                    "seed": seed, "loss": "mse", "cost_ratio": ratio,
                    "cost": ratio * 10 + seed})
    saved = plot_cost_curve(csv_path, tmp_path / "figs", show=False)
    assert all(p.exists() for p in saved)


def test_plot_cross_tsfm(tmp_path):
    csv_path = tmp_path / "representation_fairness.csv"
    for model in ("chronos-2_mlp", "moment-1-large_mlp"):
        for mode in ("native", "common"):
            for seed in (0, 1, 2):
                append_result_row(csv_path, {
                    "schema_version": 2, "model": model, "n_units": 100, "seed": seed,
                    "loss": "mse", "dataset": "FD001", "max_rul": 125, "window_size": 30,
                    "tsfm_context_length": 30, "head_features": "emb",
                    "pooling": "mean", "baseline_window": "",
                    "rmse_clipped": 12.0 + seed, "mae_clipped": 9.0, "nasa_clipped": 300.0,
                    "rmse_unclipped": 15.0, "mae_unclipped": 11.0, "nasa_unclipped": 400.0,
                    "n": 100, "mode": mode, "channel_aggregation":
                    "concat" if mode == "native" else "mean"})
    saved = plot_cross_tsfm(csv_path, tmp_path / "figs", show=False)
    assert any("cross_tsfm_rmse_clipped" in p.name for p in saved)
    assert all(p.exists() for p in saved)
