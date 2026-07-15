"""CPU tests for the a-c follow-ups (CHANGES.md §18-19): adaptive horizon bins +
dual label-cap arms, the paired seed t-test, the CSV schema guard, and the
fairness baselines (cycle_reg floor + gbm_age)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from src.config import Config
from src import horizon as H
from src.sweep import run_fairness_baselines
from src.embeddings import build_embedding_cache
from src.evaluate import ensure_csv_schema, paired_seed_ttest, append_result_row
from tests.synthetic import write_synthetic_cmapss, MockEmbedder


def _cfg(tmp_path: Path, **over) -> Config:
    base = dict(
        dataset="FD001",
        data_dir=str(tmp_path / "CMAPSSData"),
        cache_dir=str(tmp_path / "cache"),
        results_dir=str(tmp_path / "results"),
        window_size=12,
        sensor_columns=["s_2", "s_3", "s_4", "s_7", "s_9"],
        max_rul=40,
        num_bins=8,
        data_unit_counts=[2, 4],
        sweep_seeds=[0, 1],
        head_hidden_dim=16,
        head_batch_size=32,
        head_max_epochs=3,
        head_early_stopping_patience=2,
        baseline_max_epochs=2,
        baseline_early_stopping_patience=1,
        losses=["mse"],
    )
    base.update(over)
    return Config(**base)


def test_default_bin_edges():
    assert H.default_bin_edges(125) == H.DEFAULT_BIN_EDGES
    e200 = H.default_bin_edges(200)
    assert e200[:6] == H.DEFAULT_BIN_EDGES[:6]          # <=125 edges identical
    assert e200[-4:-1] == (150.0, 175.0, 200.0) and np.isinf(e200[-1])
    assert H.default_bin_edges(30, width=20) == (0.0, 20.0, 30.0, float("inf"))


def test_ensure_csv_schema_guard(tmp_path):
    p = tmp_path / "x.csv"
    ensure_csv_schema(p, ["a", "b"])                     # missing file: fine
    append_result_row(p, {"a": 1, "b": 2})
    ensure_csv_schema(p, ["a", "b"])                     # matching: fine
    with pytest.raises(ValueError, match="archive"):
        ensure_csv_schema(p, ["a", "b", "max_rul"])      # changed schema: loud


def test_paired_seed_ttest(tmp_path):
    p = tmp_path / "horizon.csv"
    # corn beats mse by ~1.0 (varying slightly by seed) in bin 0-20; ties in 20-inf
    deltas = [0.8, 0.9, 1.1, 1.2]
    for seed in range(4):
        for loss, m0, m1 in (("corn", 6.0 + seed - deltas[seed], 7.0),
                             ("mse", 6.0 + seed, 7.0)):
            for lo, hi, mval in (("0.0", "20.0", m0), ("20.0", "inf", m1)):
                append_result_row(p, {"model": "chronos-2_mlp", "max_rul": 40,
                                      "n_units": 4, "seed": seed, "loss": loss,
                                      "bin_lo": lo, "bin_hi": hi, "mae_clipped": mval})
    rows = paired_seed_ttest(p, loss_a="corn", loss_b="mse", metric="mae_clipped")
    by_bin = {r["bin_lo"]: r for r in rows}
    r = by_bin["0.0"]
    assert r["n_seeds"] == 4 and abs(r["mean_delta"] - (-1.0)) < 1e-12
    assert np.isfinite(r["t"]) and r["p"] < 0.01         # consistent large delta
    assert np.isnan(by_bin["20.0"]["t"])                 # zero-variance -> nan


def test_fairness_baselines_rows_and_restart(tmp_path):
    cfg = _cfg(tmp_path)
    write_synthetic_cmapss(Path(cfg.data_dir), n_train_units=8, n_test_units=5)
    out = run_fairness_baselines(cfg)
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    models = {r["model"] for r in rows}
    assert models == {"cycle_reg", "gbm_age"}
    # grid = requested counts below the fleet + the auto-appended full-fleet cell (§29);
    # 8 train units, data_unit_counts=[2,4] -> {2,4,8}.
    from src.sweep import resolve_unit_counts
    n_cells = len(resolve_unit_counts(cfg.data_unit_counts, 8))
    assert len(rows) == 2 * n_cells * len(cfg.sweep_seeds)
    # predictions respected the clip: metrics finite, floor beats nothing crazy
    assert all(0 < float(r["rmse_clipped"]) < 200 for r in rows)
    run_fairness_baselines(cfg)                          # restart: adds nothing
    with open(out, newline="") as f:
        assert len(list(csv.DictReader(f))) == len(rows)


def test_horizon_dual_cap_arms_share_csv(tmp_path):
    cfg = _cfg(tmp_path)
    write_synthetic_cmapss(Path(cfg.data_dir), n_train_units=6, n_test_units=4,
                           min_cycles=20, max_cycles=45)
    embedder = MockEmbedder(feature_dim=16)
    for mr in (40, 60):
        c = cfg.replace(max_rul=mr)
        build_embedding_cache(c, embedder=embedder)
        H.build_horizon_cache(c, embedder=embedder)
        out = H.run_horizon_eval(c, n_units_list=[4], seeds=[0],
                                 baseline_names=[])
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    assert {r["max_rul"] for r in rows} == {"40", "60"}
    # 60-cap arm got the extra bin(s): edges differ between arms
    bins40 = {r["bin_lo"] for r in rows if r["max_rul"] == "40"}
    bins60 = {r["bin_lo"] for r in rows if r["max_rul"] == "60"}
    assert bins60 - bins40
    # restart with both arms adds nothing (max_rul is in the cell key)
    n = len(rows)
    for mr in (40, 60):
        H.run_horizon_eval(cfg.replace(max_rul=mr), n_units_list=[4], seeds=[0],
                           baseline_names=[])
    with open(out, newline="") as f:
        assert len(list(csv.DictReader(f))) == n
    # preds carry the cap column
    with open(Path(cfg.results_dir) / "horizon_predictions.csv", newline="") as f:
        pr = list(csv.DictReader(f))
    assert {r["max_rul"] for r in pr} == {"40", "60"}
