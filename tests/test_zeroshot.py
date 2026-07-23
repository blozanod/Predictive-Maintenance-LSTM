"""Zero-shot health-index forecasting (RQ-Z; §39): threshold-crossing logic + the
run wired through a mock forecaster (no backbone)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from src.config import Config
from src import zeroshot as Z
from tests.synthetic import write_synthetic_cmapss


class _FixedForecaster:
    """Returns a fixed forecast trajectory (the CPU-test forecaster seam)."""
    def __init__(self, trajectory):
        self.trajectory = np.asarray(trajectory, np.float64)

    def forecast(self, series, horizon):
        return self.trajectory[:horizon]


class _RampForecaster:
    """Continues the series upward at a fixed slope, so the index always crosses."""
    def __init__(self, slope=1.0):
        self.slope = slope

    def forecast(self, series, horizon):
        return series[-1] + self.slope * np.arange(1, horizon + 1)


# ---------------------------------------------------------------------------
# threshold-crossing logic (closed form)
# ---------------------------------------------------------------------------
def test_crossing_returns_first_step_over_threshold():
    rul = Z.threshold_crossing_rul([0.0, 1.0, 2.0], _FixedForecaster([3, 4, 5, 6]),
                                   threshold=5.0, horizon=4)
    assert rul == 3.0                       # forecast hits 5 at the 3rd step


def test_crossing_already_past_threshold_is_zero():
    rul = Z.threshold_crossing_rul([1.0, 5.0, 10.0], _FixedForecaster([11, 12]),
                                   threshold=5.0, horizon=2)
    assert rul == 0.0                       # last observed value already >= threshold


def test_crossing_never_reached_returns_horizon():
    rul = Z.threshold_crossing_rul([0.0, 1.0], _FixedForecaster([2, 3, 4]),
                                   threshold=99.0, horizon=3)
    assert rul == 3.0                       # capped at the horizon


# ---------------------------------------------------------------------------
# health index transform (unsupervised, oriented toward failure)
# ---------------------------------------------------------------------------
def test_health_index_increases_toward_failure():
    # a single degrading channel: value rises with cycle
    n = 40
    X = np.linspace(0, 10, n)[:, None] + np.random.default_rng(0).normal(0, 0.1, (n, 1))
    t = np.arange(1, n + 1, dtype=float)
    tf = Z.build_health_index(X, t)
    idx = Z.health_index(X, tf)
    assert np.corrcoef(idx, t)[0, 1] > 0.9   # oriented to increase with elapsed cycles


# ---------------------------------------------------------------------------
# run_zeroshot end-to-end (mock forecaster)
# ---------------------------------------------------------------------------
def _cfg(tmp_path: Path) -> Config:
    return Config(dataset="FD001", data_dir=str(tmp_path / "CMAPSSData"),
                  cache_dir=str(tmp_path / "cache"), results_dir=str(tmp_path / "results"),
                  window_size=12, sensor_columns=["s_2", "s_3", "s_4", "s_7", "s_9"],
                  max_rul=40)


def test_run_zeroshot_writes_rows_and_floors(tmp_path):
    cfg = _cfg(tmp_path)
    write_synthetic_cmapss(Path(cfg.data_dir), n_train_units=8, n_test_units=6)
    out = Z.run_zeroshot(cfg, forecaster_factory=lambda c: _RampForecaster(slope=0.2))
    assert out.name == "zeroshot.csv"
    rows = list(csv.DictReader(open(out)))
    models = {r["model"] for r in rows}
    assert "chronos-2_zeroshot" in models
    assert {"predict_mean", "cycle_reg"} <= models        # scored vs the floors
    # one row per (model, seed): 3 models x the default 5 seeds
    assert {int(r["seed"]) for r in rows} == {0, 1, 2, 3, 4}
    assert len(rows) == 3 * 5
    for r in rows:
        assert int(r["n_units"]) == 0                     # the 0-failures endpoint
        assert np.isfinite(float(r["rmse_clipped"]))
        assert np.isfinite(float(r["nasa_clipped"]))
        assert 0.0 <= float(r["mae_clipped"])


def test_run_zeroshot_seeds_vary_and_average(tmp_path):
    """The bootstrap makes the arm genuinely stochastic across seeds, so the seed-mean
    is a real average of distinct draws (not a repeated single number)."""
    cfg = _cfg(tmp_path)
    write_synthetic_cmapss(Path(cfg.data_dir), n_train_units=12, n_test_units=6)
    out = Z.run_zeroshot(cfg, seeds=[0, 1, 2],
                         forecaster_factory=lambda c: _RampForecaster(slope=0.2))
    rows = [r for r in csv.DictReader(open(out)) if r["model"] == "chronos-2_zeroshot"]
    assert {int(r["seed"]) for r in rows} == {0, 1, 2}
    rmses = {float(r["rmse_clipped"]) for r in rows}
    assert len(rmses) > 1                                 # seeds genuinely differ


def test_run_zeroshot_restartable(tmp_path):
    cfg = _cfg(tmp_path)
    write_synthetic_cmapss(Path(cfg.data_dir), n_train_units=8, n_test_units=6)
    out = Z.run_zeroshot(cfg, forecaster_factory=lambda c: _RampForecaster(0.2))
    n1 = len(list(csv.DictReader(open(out))))
    Z.run_zeroshot(cfg, forecaster_factory=lambda c: _RampForecaster(0.2))
    assert len(list(csv.DictReader(open(out)))) == n1     # all models already done


def test_chronos_forecaster_constructs_without_backbone():
    """The default forecaster is cheap to build (the backbone load lives in forecast,
    the pragma boundary) -- constructing it must not import chronos."""
    f = Z.ChronosForecaster(Config(dataset="FD001"))
    assert f.model_name == "amazon/chronos-2" and f._pipeline is None


def test_zeroshot_arm_is_scoreable_by_the_win_rule(tmp_path):
    """The plan (IMPLEMENTATION_PLAN §4.5) scores the zero-shot arm with the win-rule
    vs the floors. Before §40 this yielded ZERO rows (the ``_zeroshot`` tag was not
    recognized as a TSFM); it must now produce a scored row against a floor."""
    from src import scoring as SC
    cfg = _cfg(tmp_path)
    write_synthetic_cmapss(Path(cfg.data_dir), n_train_units=8, n_test_units=6)
    out = Z.run_zeroshot(cfg, forecaster_factory=lambda c: _RampForecaster(slope=0.2))
    assert SC.success_map(out) == []                        # unscoreable as a core cell
    table = SC.success_map(out, compare_to_floors=True)     # the zero-shot scoring path
    assert len(table) == 1
    row = table[0]
    assert row["model"] == "chronos-2_zeroshot"
    assert row["verdict"] in ("win", "tie", "loss")         # a real verdict, not skipped
    assert row["best_baseline"] in ("predict_mean", "cycle_reg")
    assert np.isfinite(row["margin"])


def test_run_zeroshot_predictions_within_cap(tmp_path):
    cfg = _cfg(tmp_path)
    write_synthetic_cmapss(Path(cfg.data_dir), n_train_units=8, n_test_units=6)
    # a forecaster that never crosses -> every unit's RUL is capped at the horizon
    out = Z.run_zeroshot(cfg, seeds=[0], horizon=25,
                         forecaster_factory=lambda c: _RampForecaster(slope=-1.0))
    z_rows = [r for r in csv.DictReader(open(out)) if r["model"] == "chronos-2_zeroshot"]
    assert len(z_rows) == 1
    # predictions were clipped into [0, max_rul]; metrics stay finite
    assert np.isfinite(float(z_rows[0]["rmse_clipped"]))
