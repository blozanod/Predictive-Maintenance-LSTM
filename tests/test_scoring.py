"""Win-rule / success-map scoring (§36). Synthetic CSVs with hand-set seed-means
exercise strongest-baseline selection and every verdict (win/tie/loss/hollow)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.config import Config
from src import scoring as SC
from src.evaluate import append_result_row, paired_ttest


def _rows(model, dataset, n_units, nasa_by_seed, rmse_by_seed=None, **extra):
    """One row per seed with hand-set nasa_clipped (+ optional rmse_clipped)."""
    out = []
    for seed, nasa in nasa_by_seed.items():
        r = {"model": model, "dataset": dataset, "n_units": n_units, "seed": seed,
             "loss": "mse" if SC.is_tsfm_model(model) else "native",
             "nasa_clipped": nasa}
        if rmse_by_seed is not None:
            r["rmse_clipped"] = rmse_by_seed[seed]
        r.update(extra)
        out.append(r)
    return out


def _seeds(vals):
    return {i: v for i, v in enumerate(vals)}


# ---------------------------------------------------------------------------
# paired-test core (generalized, nan-safe)
# ---------------------------------------------------------------------------
def test_paired_ttest_nan_safe():
    t, p = paired_ttest([1.0], [2.0])                 # <2 pairs
    assert np.isnan(t) and np.isnan(p)
    t, p = paired_ttest([1.0, 1.0, 1.0], [2.0, 2.0, 2.0])   # constant diff
    assert np.isnan(t) and np.isnan(p)
    t, p = paired_ttest([10, 11, 9, 10, 10], [20, 19, 22, 21, 18])
    assert np.isfinite(t) and p < 0.05                # clear, consistent separation


# ---------------------------------------------------------------------------
# strongest baseline = the toughest COMPETITOR (floors excluded)
# ---------------------------------------------------------------------------
def test_strongest_baseline_is_min_competitor_excluding_floors():
    rows = (_rows("gbm", "FD001", 100, _seeds([20, 20, 20]))
            + _rows("cnn", "FD001", 100, _seeds([15, 15, 15]))       # best competitor
            + _rows("minirocket", "FD001", 100, _seeds([25, 25, 25]))  # worse than cnn
            + _rows("predict_mean", "FD001", 100, _seeds([9, 9, 9])) # floor, lower but excluded
            + _rows("chronos-2_mlp", "FD001", 100, _seeds([12, 12, 12])))
    strongest = SC.strongest_baseline_per_cell(rows)
    assert strongest[("FD001", 100)] == ("cnn", pytest.approx(15.0))
    assert SC.is_competitor_baseline("gbm") and not SC.is_competitor_baseline("predict_mean")
    assert not SC.is_competitor_baseline("chronos-2_mlp")


def test_strongest_baseline_none_when_only_floors():
    rows = (_rows("predict_mean", "FD001", 100, _seeds([9, 9, 9]))
            + _rows("chronos-2_mlp", "FD001", 100, _seeds([12, 12, 12])))
    assert SC.strongest_baseline_per_cell(rows) == {}


# ---------------------------------------------------------------------------
# the four verdicts
# ---------------------------------------------------------------------------
def _all_scenarios():
    # Differences must be NON-constant (a constant paired diff has no variance and
    # yields nan p -> tie): the gbm arm varies while staying on its side of the TSFM.
    # win: TSFM ~10 beats gbm ~20 significantly, well under the predict-mean floor 40.
    win = (_rows("chronos-2_mlp", "FD001", 100, _seeds([10, 11, 9, 10, 10]))
           + _rows("gbm", "FD001", 100, _seeds([20, 19, 22, 21, 18]))
           + _rows("predict_mean", "FD001", 100, _seeds([40, 40, 40, 40, 40])))
    # loss: TSFM ~30 clearly worse than gbm ~10.
    loss = (_rows("chronos-2_mlp", "FD001", 50, _seeds([30, 31, 29, 30, 30]))
            + _rows("gbm", "FD001", 50, _seeds([10, 9, 12, 11, 8]))
            + _rows("predict_mean", "FD001", 50, _seeds([40, 40, 40, 40, 40])))
    # tie: TSFM and gbm within noise (tiny, non-significant difference).
    tie = (_rows("chronos-2_mlp", "FD001", 25, _seeds([20, 15, 25, 10, 30]))
           + _rows("gbm", "FD001", 25, _seeds([21, 14, 26, 9, 31]))
           + _rows("predict_mean", "FD001", 25, _seeds([40, 40, 40, 40, 40])))
    # hollow: TSFM (15) beats gbm (20) significantly, but the predict-mean floor (14)
    # is BETTER than the TSFM -> everything real fails, so the "win" is hollow.
    hollow = (_rows("chronos-2_mlp", "FD001", 10, _seeds([15, 16, 14, 15, 15]))
              + _rows("gbm", "FD001", 10, _seeds([20, 19, 22, 21, 18]))
              + _rows("predict_mean", "FD001", 10, _seeds([14, 14, 14, 14, 14])))
    return win, loss, tie, hollow


def test_win_verdict_covers_all_four():
    win, loss, tie, hollow = _all_scenarios()
    v = SC.win_verdict(win + loss + tie + hollow, Config())
    assert v[("FD001", 100, "chronos-2_mlp")]["verdict"] == "win"
    assert v[("FD001", 50, "chronos-2_mlp")]["verdict"] == "loss"
    assert v[("FD001", 25, "chronos-2_mlp")]["verdict"] == "tie"
    assert v[("FD001", 10, "chronos-2_mlp")]["verdict"] == "hollow"
    # win carries a positive margin and the chosen baseline
    w = v[("FD001", 100, "chronos-2_mlp")]
    assert w["margin"] == pytest.approx(10.0) and w["best_baseline"] == "gbm"
    assert w["p"] < 0.05 and w["n_seeds"] == 5


def test_win_margin_and_alpha_shift_verdicts():
    win, _, _, _ = _all_scenarios()
    # a huge required margin turns the 10-cycle win into a tie
    strict = Config(win_margin=100.0)
    assert SC.win_verdict(win, strict)[("FD001", 100, "chronos-2_mlp")]["verdict"] == "tie"
    # an impossibly strict alpha also demotes it (significance never reached)
    strict_a = Config(win_alpha=1e-12)
    assert SC.win_verdict(win, strict_a)[("FD001", 100, "chronos-2_mlp")]["verdict"] == "tie"


def test_win_verdict_skips_rows_missing_the_metric():
    """A row without the primary metric (mixed-schema CSV) is skipped, not crashed on."""
    win, *_ = _all_scenarios()
    bad = _rows("cnn", "FD001", 100, _seeds([np.nan]))  # placeholder, overwritten below
    bad[0]["nasa_clipped"] = ""                          # missing metric value
    v = SC.win_verdict(win + bad, Config())
    assert v[("FD001", 100, "chronos-2_mlp")]["verdict"] == "win"


def test_win_verdict_skips_cells_without_competitor():
    rows = (_rows("predict_mean", "FD001", 100, _seeds([9, 9, 9]))
            + _rows("chronos-2_mlp", "FD001", 100, _seeds([12, 12, 12])))
    assert SC.win_verdict(rows, Config()) == {}


# ---------------------------------------------------------------------------
# zero-shot arm: recognized as a TSFM + scored against the floors (RQ-Z, §4.5)
# ---------------------------------------------------------------------------
def test_is_tsfm_recognizes_mlp_and_zeroshot():
    assert SC.is_tsfm_model("chronos-2_mlp")
    assert SC.is_tsfm_model("moirai-2_zeroshot")           # the RQ-Z arm is a TSFM too
    assert not SC.is_competitor_baseline("chronos-2_zeroshot")   # never a competitor bar
    assert not SC.is_tsfm_model("gbm")


def test_win_verdict_compare_to_floors_scores_zeroshot():
    # the zero-shot arm carries only the TSFM + the two floors (no competitor baseline).
    rows = (_rows("chronos-2_zeroshot", "FD001", 0, _seeds([10, 11, 9, 10, 10]))
            + _rows("predict_mean", "FD001", 0, _seeds([40, 41, 39, 42, 38]))
            + _rows("cycle_reg", "FD001", 0, _seeds([30, 32, 28, 31, 29])))
    # default path: floors are excluded => no competitor => no verdict (the old gap)
    assert SC.win_verdict(rows, Config()) == {}
    # floors-as-bar: scored against the TOUGHER (lower) floor, cycle_reg
    v = SC.win_verdict(rows, Config(), compare_to_floors=True)
    key = ("FD001", 0, "chronos-2_zeroshot")
    assert v[key]["verdict"] == "win"
    assert v[key]["best_baseline"] == "cycle_reg"
    assert v[key]["margin"] == pytest.approx(20.0) and v[key]["n_seeds"] == 5


def test_win_verdict_compare_to_floors_loss_no_hollow():
    # worse than both floors => loss; the hollow guard never fires when floors are the bar.
    rows = (_rows("chronos-2_zeroshot", "FD001", 0, _seeds([50, 51, 49, 50, 50]))
            + _rows("predict_mean", "FD001", 0, _seeds([30, 32, 28, 31, 29]))
            + _rows("cycle_reg", "FD001", 0, _seeds([40, 41, 39, 42, 38])))
    v = SC.win_verdict(rows, Config(), compare_to_floors=True)
    assert v[("FD001", 0, "chronos-2_zeroshot")]["verdict"] == "loss"


def test_success_map_compare_to_floors(tmp_path):
    rows = (_rows("chronos-2_zeroshot", "FD001", 0, _seeds([10, 11, 9, 10, 10]),
                  rmse_by_seed=_seeds([7, 7, 7, 7, 7]))
            + _rows("predict_mean", "FD001", 0, _seeds([40, 41, 39, 42, 38]))
            + _rows("cycle_reg", "FD001", 0, _seeds([30, 32, 28, 31, 29])))
    csv_path = tmp_path / "combo_zeroshot.csv"
    for r in rows:
        append_result_row(csv_path, r)
    assert SC.success_map(csv_path) == []                       # default: unscoreable
    table = SC.success_map(csv_path, compare_to_floors=True)    # the plan's zero-shot path
    assert len(table) == 1
    assert table[0]["model"] == "chronos-2_zeroshot"
    assert table[0]["verdict"] == "win"
    assert np.isfinite(table[0]["tsfm_rmse_clipped"])


# ---------------------------------------------------------------------------
# success_map: reads per-combo CSVs and returns the table
# ---------------------------------------------------------------------------
def test_success_map_from_directory(tmp_path):
    win, loss, tie, hollow = _all_scenarios()
    # add rmse-alongside so the secondary column is populated
    def _with_rmse(rows):
        for r in rows:
            r["rmse_clipped"] = r["nasa_clipped"] / 2.0
        return rows
    # write one CSV per "combo" into a directory
    d = tmp_path / "results"
    for i, rows in enumerate((win, loss, tie, hollow)):
        csv_path = d / f"combo_{i}_results_v2.csv"
        for r in _with_rmse(rows):
            append_result_row(csv_path, r)
    table = SC.success_map(d)
    assert len(table) == 4
    verdicts = {(row["n_units"], row["verdict"]) for row in table}
    assert verdicts == {(100, "win"), (50, "loss"), (25, "tie"), (10, "hollow")}
    win_row = next(r for r in table if r["n_units"] == 100)
    assert win_row["model"] == "chronos-2_mlp" and win_row["metric"] == "nasa_clipped"
    assert np.isfinite(win_row["tsfm_rmse_clipped"])      # RMSE reported alongside
    assert win_row["best_baseline"] == "gbm"


def test_success_map_single_csv_and_glob(tmp_path):
    win, *_ = _all_scenarios()
    csv_path = tmp_path / "one_results_v2.csv"
    for r in win:
        append_result_row(csv_path, r)
    assert len(SC.success_map(csv_path)) == 1                    # single file
    assert len(SC.success_map(str(tmp_path / "*_results_v2.csv"))) == 1   # glob


def test_success_map_probe_cell_fields(tmp_path):
    """A probe-style table keys cells on (dataset, factor, level, n_units)."""
    rows = (_rows("chronos-2_mlp", "FD001", 100, _seeds([10, 11, 9, 10, 10]),
                  factor="channels", level="subset_a")
            + _rows("gbm", "FD001", 100, _seeds([20, 19, 22, 21, 18]),
                    factor="channels", level="subset_a"))
    cell_fields = ("dataset", "factor", "level", "n_units")
    # win_verdict on the richer cell
    v = SC.win_verdict(rows, Config(), cell_fields=cell_fields)
    assert v[("FD001", "channels", "subset_a", 100, "chronos-2_mlp")]["verdict"] == "win"
    # success_map unpacks the cell tuple back into named columns
    csv_path = tmp_path / "probe_channels.csv"
    for r in rows:
        append_result_row(csv_path, r)
    table = SC.success_map(csv_path, cell_fields=cell_fields)
    assert len(table) == 1
    row = table[0]
    assert row["factor"] == "channels" and row["level"] == "subset_a"
    assert row["verdict"] == "win"
