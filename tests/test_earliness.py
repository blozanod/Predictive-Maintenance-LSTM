"""Earliness histograms + cost curves (§37): closed-form metric checks + the runner
that emits earliness.csv / cost_curve.csv from horizon_predictions.csv."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from src.config import Config
from src.evaluate import earliness_histogram, cost_curve, append_result_row
from src.horizon import run_earliness


# ---------------------------------------------------------------------------
# Closed-form: d = pred - true, d>=0 LATE (dangerous), d<0 EARLY (wasteful)
# ---------------------------------------------------------------------------
def test_earliness_histogram_split_and_bins():
    y_true = np.array([100.0, 100.0])
    y_pred = np.array([110.0, 90.0])        # d = +10 (late), -10 (early)
    h = earliness_histogram(y_true, y_pred, [-50, -25, -10, 0, 10, 25, 50])
    assert h["n"] == 2
    assert h["frac_late"] == 0.5 and h["frac_early"] == 0.5
    assert h["mean_signed_error"] == pytest.approx(0.0)
    late = [b for b in h["bins"] if b["side"] == "late" and b["n_bin"]]
    early = [b for b in h["bins"] if b["side"] == "early" and b["n_bin"]]
    assert late[0]["lo"] == 10.0 and late[0]["hi"] == 25.0    # +10 lands in [10, 25)
    assert early[0]["lo"] == -10.0 and early[0]["hi"] == 0.0  # -10 lands in [-10, 0)
    assert sum(b["frac"] for b in h["bins"]) == pytest.approx(1.0)


def test_earliness_all_late_when_overpredicting():
    h = earliness_histogram([10.0, 20.0, 30.0], [40.0, 40.0, 40.0], [0.0])
    assert h["frac_late"] == 1.0 and h["frac_early"] == 0.0


def test_earliness_empty_raises():
    with pytest.raises(ValueError, match="at least one"):
        earliness_histogram([], [], [0.0])


def test_cost_curve_closed_form():
    y_true = np.array([100.0, 100.0])
    y_pred = np.array([110.0, 90.0])        # late total = 10, early total = 10
    cc = cost_curve(y_true, y_pred, [1.0, 5.0, 100.0])
    assert cc[1.0] == pytest.approx(20.0)          # 10 early + 1*10 late
    assert cc[5.0] == pytest.approx(60.0)          # 10 early + 5*10 late
    assert cc[100.0] == pytest.approx(1010.0)      # lateness dominates as ratio grows


def test_cost_curve_monotone_in_ratio():
    rng = np.random.default_rng(0)
    y_true = rng.uniform(0, 125, 50)
    y_pred = rng.uniform(0, 125, 50)
    cc = cost_curve(y_true, y_pred, [1, 2, 5, 10, 100])
    costs = [cc[float(r)] for r in [1, 2, 5, 10, 100]]
    assert all(b >= a for a, b in zip(costs, costs[1:]))   # non-decreasing in late:early


# ---------------------------------------------------------------------------
# Runner: horizon_predictions.csv -> earliness.csv + cost_curve.csv
# ---------------------------------------------------------------------------
def _write_preds(path: Path):
    # two cells (two models), each with 4 per-cycle predictions
    for model, preds in (("chronos-2_mlp", [(100, 110), (80, 60), (50, 55), (20, 10)]),
                         ("gbm", [(100, 90), (80, 82), (50, 40), (20, 30)])):
        for unit, (true, pred) in enumerate(preds):
            append_result_row(path, {
                "model": model, "dataset": "FD001", "max_rul": 125,
                "n_units": 100, "seed": 0, "loss": "mse" if "mlp" in model else "native",
                "unit": unit, "true_rul": float(true), "pred": float(pred)})


def _cfg(tmp_path: Path) -> Config:
    return Config(dataset="FD001", results_dir=str(tmp_path / "results"),
                  cost_ratios=[1.0, 10.0], earliness_bin_edges=[-25.0, 0.0, 25.0])


def test_run_earliness_emits_both_csvs(tmp_path):
    cfg = _cfg(tmp_path)
    preds = cfg.results_path("horizon_predictions.csv")
    _write_preds(preds)
    e_csv, c_csv = run_earliness(cfg)
    assert e_csv.exists() and c_csv.exists()

    e_rows = list(csv.DictReader(open(e_csv)))
    models = {r["model"] for r in e_rows}
    assert models == {"chronos-2_mlp", "gbm"}
    # each cell emits one row per bin (edges [-25,0,25] -> 4 bins with -inf/inf)
    per_model = [r for r in e_rows if r["model"] == "gbm"]
    assert len({(r["lo"], r["hi"]) for r in per_model}) == 4
    assert all(r["side"] in ("late", "early") for r in per_model)

    c_rows = list(csv.DictReader(open(c_csv)))
    ratios = {float(r["cost_ratio"]) for r in c_rows}
    assert ratios == {1.0, 10.0}
    assert all(float(r["cost"]) >= 0 for r in c_rows)


def test_run_earliness_restartable(tmp_path):
    cfg = _cfg(tmp_path)
    _write_preds(cfg.results_path("horizon_predictions.csv"))
    e_csv, c_csv = run_earliness(cfg)
    n_e, n_c = len(list(csv.DictReader(open(e_csv)))), len(list(csv.DictReader(open(c_csv))))
    run_earliness(cfg)                    # rerun: both cells already done -> no new rows
    assert len(list(csv.DictReader(open(e_csv)))) == n_e
    assert len(list(csv.DictReader(open(c_csv)))) == n_c


def test_run_earliness_missing_preds_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="predictions file"):
        run_earliness(_cfg(tmp_path))
