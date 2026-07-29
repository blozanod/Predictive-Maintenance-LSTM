"""CPU tests for the censoring machinery and the binary alarm arm (CHANGES.md §54).

The chapter's load-bearing claim is that a mostly-healthy fleet can be learned from
WITHOUT inventing failure dates. Concretely:

  * a right-censored survivor observed past the horizon is a genuine NEGATIVE and must
    train; a survivor whose observation ends INSIDE the horizon is UNKNOWABLE and must be
    dropped, not labelled 0 (the classic censoring bug -- it teaches "healthy" from
    absence of evidence and inflates precision);
  * the alarm arm predicts a PROBABILITY, so it needs its own loss, its own decode, its
    own early-stopping criterion, its own metric block, its own baselines, its own CSV,
    and a win-rule that knows the metric direction is reversed;
  * none of that may perturb the RUL path by even one byte.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.config import Config
from src import baselines as B
from src import data as D
from src import evaluate as E
from src import heads as H
from src import scoring as S
from src import sweep as SW
from src import train as T
from tests.synthetic import write_synthetic_cmapss, MockEmbedder


def _frame(rul_by_unit: dict[int, list[float]], observed: dict[int, int]) -> pd.DataFrame:
    rows = []
    for unit, ruls in rul_by_unit.items():
        for i, r in enumerate(ruls, start=1):
            rows.append({"unit_number": unit, "time_cycles": i, "actual_rul": r,
                         D.EVENT_OBSERVED_COLUMN: observed[unit]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# (a) the label
# ---------------------------------------------------------------------------
def test_alarm_label_truth_table():
    """All four rows of the §54 table, in one assertion each."""
    cfg = Config(dataset="FD001", max_rul=125, alarm_horizon=10)
    df = _frame({1: [30.0, 5.0], 2: [30.0, 5.0]}, {1: 1, 2: 0})
    out = D.add_alarm_label(df, cfg)[D.ALARM_LABEL_COLUMN].tolist()
    assert out[0] == 0.0          # observed, r > H  -> a real negative
    assert out[1] == 1.0          # observed, r <= H -> a real positive
    assert out[2] == 0.0          # censored, r > H  -> provably survived the horizon
    assert np.isnan(out[3])       # censored, r <= H -> UNKNOWABLE, never 0


def test_censored_rows_inside_the_horizon_are_dropped_not_guessed():
    cfg = Config(dataset="FD001", max_rul=125, alarm_horizon=10)
    df = _frame({1: [30.0, 20.0, 5.0, 1.0]}, {1: 0})
    labelled = D.add_alarm_label(df, cfg)
    assert labelled[D.ALARM_LABEL_COLUMN].isna().sum() == 2
    kept = D.drop_unlabeled_rows(labelled, D.ALARM_LABEL_COLUMN)
    assert len(kept) == 2
    assert (kept[D.ALARM_LABEL_COLUMN] == 0.0).all()
    assert list(kept.index) == [0, 1], "index must be reset after dropping"


def test_a_frame_without_the_event_column_is_all_observed():
    """Run-to-failure families never emit ``event_observed``; their runs all end in a
    failure, so absence must mean "observed", not "censored"."""
    cfg = Config(dataset="FD001", max_rul=125, alarm_horizon=10)
    df = pd.DataFrame({"unit_number": [1, 1], "time_cycles": [1, 2],
                       "actual_rul": [30.0, 5.0]})
    out = D.add_alarm_label(df, cfg)
    assert out[D.ALARM_LABEL_COLUMN].tolist() == [0.0, 1.0]
    assert not out[D.ALARM_LABEL_COLUMN].isna().any()


def test_alarm_label_is_a_noop_without_a_horizon():
    cfg = Config(dataset="FD001")
    df = _frame({1: [5.0]}, {1: 1})
    assert D.add_alarm_label(df, cfg) is df
    assert D.ALARM_LABEL_COLUMN not in D.add_alarm_label(df, cfg).columns


def test_drop_unlabeled_is_a_noop_when_nothing_is_missing():
    df = pd.DataFrame({"x": [1.0, 2.0]})
    assert D.drop_unlabeled_rows(df, "absent") is df       # column not present
    assert D.drop_unlabeled_rows(df, "x") is df            # fully labelled


def test_load_prepared_attaches_and_prunes_the_alarm_label(tmp_path):
    """The ONE loading path must carry the alarm target end to end."""
    data_dir = tmp_path / "CMAPSSData"
    write_synthetic_cmapss(data_dir, dataset="FD001", n_train_units=6, n_test_units=4,
                           min_cycles=25, max_cycles=40, seed=1)
    cfg = Config(dataset="FD001", data_dir=str(data_dir), max_rul=60, alarm_horizon=10,
                 window_size=6)
    df_train, df_test = D.load_prepared(cfg)
    for frame in (df_train, df_test):
        assert D.ALARM_LABEL_COLUMN in frame.columns
        assert not frame[D.ALARM_LABEL_COLUMN].isna().any()
        assert set(frame[D.ALARM_LABEL_COLUMN].unique()) <= {0.0, 1.0}


def test_alarm_horizon_keys_only_when_set():
    base = Config(dataset="FD001")
    assert "alarm_horizon" not in base._window_key_fields()
    assert base.window_cache_key() == "windows_FD001_1da313c871251cec"
    withh = Config(dataset="FD001", alarm_horizon=30)
    assert withh._window_key_fields()["alarm_horizon"] == 30
    assert withh.window_cache_key() != base.window_cache_key()
    assert (Config(dataset="FD001", alarm_horizon=31).window_cache_key()
            != withh.window_cache_key())


def test_config_guards_the_horizon_against_the_clip_point():
    with pytest.raises(ValueError, match="must be < max_rul"):
        Config(dataset="FD001", max_rul=125, alarm_horizon=125)
    with pytest.raises(ValueError, match="alarm_horizon must be >= 1"):
        Config(dataset="FD001", alarm_horizon=0)
    with pytest.raises(ValueError, match="alarm_threshold"):
        Config(dataset="FD001", alarm_threshold=0.0)
    Config(dataset="FD001", max_rul=125, alarm_horizon=124)   # the boundary is allowed


# ---------------------------------------------------------------------------
# (b) the arm
# ---------------------------------------------------------------------------
def test_alarm_head_emits_one_logit_and_decodes_to_a_probability():
    cfg = Config(dataset="FD001", max_rul=125, alarm_horizon=30)
    assert H.head_output_dim(H.ALARM_LOSS, cfg) == 1
    assert H.is_alarm_loss(H.ALARM_LOSS) and not H.is_alarm_loss("mse")
    head = H.build_head(6, H.ALARM_LOSS, cfg)
    out = head(torch.randn(9, 6))
    assert out.shape == (9, 1)
    prob = H.decode(out, H.ALARM_LOSS, cfg)
    assert prob.shape == (9,) and prob.min() >= 0.0 and prob.max() <= 1.0
    # ... and it is NOT clipped into RUL units like every other arm
    assert prob.max() <= 1.0 < cfg.max_rul


def test_alarm_targets_read_the_binary_label_off_the_rul_tensor():
    cfg = Config(dataset="FD001", max_rul=125, alarm_horizon=30)
    rul = torch.tensor([0.0, 30.0, 30.5, 100.0])
    assert H.alarm_targets(rul, cfg).tolist() == [1.0, 1.0, 0.0, 0.0]


def test_alarm_loss_requires_a_horizon():
    cfg = Config(dataset="FD001")
    with pytest.raises(ValueError, match="requires config.alarm_horizon"):
        H.alarm_targets(torch.tensor([1.0]), cfg)
    with pytest.raises(ValueError, match="requires config.alarm_horizon"):
        H.compute_loss(torch.zeros(1, 1), torch.tensor([1.0]), H.ALARM_LOSS, cfg)


def test_alarm_loss_is_bce_and_rewards_the_right_direction():
    cfg = Config(dataset="FD001", max_rul=125, alarm_horizon=30)
    rul = torch.tensor([1.0, 100.0])          # -> targets [1, 0]
    confident_right = torch.tensor([[5.0], [-5.0]])
    confident_wrong = torch.tensor([[-5.0], [5.0]])
    good = float(H.compute_loss(confident_right, rul, H.ALARM_LOSS, cfg))
    bad = float(H.compute_loss(confident_wrong, rul, H.ALARM_LOSS, cfg))
    assert good < 0.05 < bad
    # matches torch's own BCE on the derived targets
    expected = float(torch.nn.functional.binary_cross_entropy_with_logits(
        confident_right.squeeze(-1), torch.tensor([1.0, 0.0])))
    assert good == pytest.approx(expected)


def test_train_head_early_stops_on_bce_for_the_alarm_arm():
    """An RMSE against RUL labels is meaningless for a probability, so the alarm arm
    early-stops on the validation BCE -- and ``val_rmse`` is nan rather than a fake."""
    cfg = Config(dataset="FD001", max_rul=125, alarm_horizon=30, head_hidden_dim=8,
                 head_max_epochs=3, head_early_stopping_patience=2, head_batch_size=8)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5)).astype(np.float32)
    y = np.where(X[:, 0] > 0, 5.0, 100.0).astype(np.float32)   # learnable alarm signal
    _model, hist = T.train_head(X, y, X, y, H.ALARM_LOSS, cfg, seed=0)
    assert all(np.isnan(v) for v in hist["val_rmse"])
    assert hist["val_score"] == hist["val_loss"]
    assert np.isfinite(hist["best_val_rmse"])


def test_regression_arms_keep_val_score_identical_to_val_rmse():
    """The new ``val_score`` must not change a single recorded regression result."""
    cfg = Config(dataset="FD001", max_rul=125, head_hidden_dim=8, head_max_epochs=3,
                 head_early_stopping_patience=2, head_batch_size=8)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 4)).astype(np.float32)
    y = rng.uniform(0, 125, size=30).astype(np.float32)
    _m, hist = T.train_head(X, y, X, y, "mse", cfg, seed=0)
    assert hist["val_score"] == hist["val_rmse"]
    assert hist["best_val_rmse"] == min(hist["val_rmse"])


def test_history_csv_carries_the_new_val_score_metric(tmp_path):
    cfg = Config(dataset="FD001", max_rul=125, head_hidden_dim=8, head_max_epochs=2,
                 head_early_stopping_patience=2, head_batch_size=8)
    X = np.random.default_rng(2).normal(size=(20, 3)).astype(np.float32)
    y = np.full(20, 50.0, np.float32)
    path = tmp_path / "curve.csv"
    T.train_head(X, y, X, y, "mse", cfg, seed=0, log_csv_path=path)
    metrics = {r["metric"] for r in csv.DictReader(open(path))}
    assert {"train_loss", "val_loss", "val_rmse", "val_score"} == metrics
    assert set(E.load_learning_curve(path)) >= {"val_score"}


# ---------------------------------------------------------------------------
# (c) the metric
# ---------------------------------------------------------------------------
def test_alarm_metrics_closed_form():
    y = [1, 1, 0, 0]
    p = [0.9, 0.4, 0.8, 0.1]          # at 0.5: TP=1, FN=1, FP=1, TN=1
    m = E.alarm_metrics(y, p, 0.5)
    assert (m["alarm_tp"], m["alarm_fp"], m["alarm_fn"], m["alarm_tn"]) == (1, 1, 1, 1)
    assert m["alarm_precision"] == pytest.approx(0.5)
    assert m["alarm_recall"] == pytest.approx(0.5)
    assert m["alarm_f1"] == pytest.approx(0.5)
    assert m["alarm_specificity"] == pytest.approx(0.5)
    # AUROC is threshold-free: of the 4 (pos, neg) score pairs, 3 are ranked correctly
    # ((.9,.8), (.9,.1), (.4,.1)) and one is not ((.4,.8)) -> 0.75.
    assert m["alarm_auroc"] == pytest.approx(0.75)
    assert m["alarm_brier"] == pytest.approx(np.mean((np.array(p) - np.array(y)) ** 2))
    assert m["n"] == 4 and m["n_positive"] == 2


def test_alarm_metrics_degenerate_cases_are_nan_not_zero():
    """Under 1-in-23,500 imbalance a cell with one class is routine -- it must report
    nan (unmeasurable), never a misleading 0."""
    allneg = E.alarm_metrics([0, 0], [0.1, 0.2], 0.5)
    assert np.isnan(allneg["alarm_recall"])          # no positives to recall
    assert np.isnan(allneg["alarm_auroc"])           # one class -> undefined ranking
    assert np.isnan(allneg["alarm_ap"])
    assert np.isnan(allneg["alarm_precision"])       # nothing flagged at 0.5
    assert allneg["alarm_specificity"] == pytest.approx(1.0)
    nothing_flagged = E.alarm_metrics([1, 0], [0.1, 0.2], 0.9)
    assert np.isnan(nothing_flagged["alarm_precision"])
    assert np.isnan(nothing_flagged["alarm_f1"])
    with pytest.raises(ValueError, match="at least one prediction"):
        E.alarm_metrics([], [], 0.5)


def test_alarm_lead_times_measure_only_the_caught_events():
    y = [1, 1, 0]
    p = [0.9, 0.2, 0.95]
    rul = [12.0, 3.0, 80.0]
    lt = E.alarm_lead_times(y, p, rul, 0.5)
    assert lt["n_caught"] == 1          # only the first row is a caught true positive
    assert lt["alarm_mean_lead_time"] == pytest.approx(12.0)
    assert lt["alarm_median_lead_time"] == pytest.approx(12.0)
    assert lt["alarm_min_lead_time"] == pytest.approx(12.0)
    missed = E.alarm_lead_times([1], [0.1], [5.0], 0.5)
    assert missed["n_caught"] == 0 and np.isnan(missed["alarm_mean_lead_time"])


def test_alarm_threshold_sweep_is_monotone_in_recall():
    y = [1, 1, 0, 0]
    p = [0.9, 0.6, 0.4, 0.1]
    sweep = E.alarm_threshold_sweep(y, p, [0.2, 0.5, 0.8])
    recalls = [r["alarm_recall"] for r in sweep]
    assert recalls == sorted(recalls, reverse=True)   # a higher bar can only lose recall
    assert [r["threshold"] for r in sweep] == [0.2, 0.5, 0.8]


def test_evaluate_alarm_predictions_columns_are_disjoint_from_the_rul_ones():
    """Structural enforcement of the non-comparability rule: an alarm row and a RUL row
    must share NO metric column, so they can never be averaged into one table."""
    row = E.evaluate_alarm_predictions([1, 0], [0.9, 0.1], [4.0, 90.0], 0.5)
    assert set(E.ALARM_METRIC_FIELDS) <= set(row)
    assert set(E.ALARM_METRIC_FIELDS).isdisjoint(E.METRIC_FIELDS)
    assert set(row).isdisjoint(set(E.METRIC_FIELDS))


# ---------------------------------------------------------------------------
# (d) the competitors
# ---------------------------------------------------------------------------
def test_alarm_base_rate_predicts_the_training_prevalence():
    cfg = Config(dataset="FD001", alarm_horizon=10)
    bl = B.make_baseline("alarm_base_rate", cfg, seed=0)
    windows = np.zeros((6, 4, 2), np.float32)
    bl.fit(windows, np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]))
    pred = bl.predict(np.zeros((3, 4, 2), np.float32))
    assert np.allclose(pred, 1 / 3) and pred.shape == (3,)


@pytest.mark.parametrize("name", ["alarm_gbm", "alarm_catch22_gbm"])
def test_alarm_classifier_baselines_emit_probabilities(name):
    pytest.importorskip("lightgbm")
    if name == "alarm_catch22_gbm":
        pytest.importorskip("pycatch22")
    cfg = Config(dataset="FD001", alarm_horizon=10)
    rng = np.random.default_rng(0)
    n, w = 80, 10
    y = (rng.random(n) < 0.4).astype(np.float64)
    # A TEMPORAL signal (a rising ramp on the positives), not a mean shift: catch22
    # z-scores each series for most of its features, so a pure level difference is
    # invisible to it while window_statistics reads it off `mean`. Ramping makes the
    # signal legible to BOTH banks, which is what this test is actually about.
    ramp = np.linspace(0.0, 6.0, w)[None, :, None]
    windows = (rng.normal(size=(n, w, 2)) + y[:, None, None] * ramp).astype(np.float32)
    bl = B.make_baseline(name, cfg, seed=0)
    bl.fit(windows[:50], y[:50], windows[50:], y[50:])
    p = bl.predict(windows)
    assert p.shape == (n,) and p.min() >= 0.0 and p.max() <= 1.0
    assert E.alarm_metrics(y, p, 0.5)["alarm_auroc"] > 0.8   # the signal is learnable


def test_alarm_classifier_degrades_gracefully_on_a_single_class_draw():
    """A low-data cell can draw only healthy units; that must not kill the sweep."""
    pytest.importorskip("lightgbm")
    cfg = Config(dataset="FD001", alarm_horizon=10)
    bl = B.make_baseline("alarm_gbm", cfg, seed=0)
    windows = np.random.default_rng(0).normal(size=(10, 6, 2)).astype(np.float32)
    bl.fit(windows, np.zeros(10))
    assert np.allclose(bl.predict(windows), 0.0)
    empty = B.make_baseline("alarm_gbm", cfg, seed=0).fit(windows[:0], np.zeros(0))
    assert np.allclose(empty.predict(windows), 0.0)


def test_alarm_classifier_skips_a_single_class_eval_set():
    pytest.importorskip("lightgbm")
    cfg = Config(dataset="FD001", alarm_horizon=10)
    rng = np.random.default_rng(3)
    windows = rng.normal(size=(30, 6, 2)).astype(np.float32)
    y = np.r_[np.ones(15), np.zeros(15)]
    bl = B.make_baseline("alarm_gbm", cfg, seed=0)
    bl.fit(windows, y, windows[:5], np.ones(5))       # val holds one class only
    assert bl.predict(windows).shape == (30,)


def test_alarm_baselines_are_registered_and_named():
    assert B.ALARM_BASELINES == {"alarm_base_rate", "alarm_gbm", "alarm_catch22_gbm"}
    assert B.ALARM_BASELINES <= set(B.BASELINES)


# ---------------------------------------------------------------------------
# (e) the win-rule direction
# ---------------------------------------------------------------------------
def test_metric_direction_is_resolved_in_one_place():
    assert S.metric_is_higher_better("alarm_ap")
    assert S.metric_is_higher_better("macro_f1")
    assert not S.metric_is_higher_better("nasa_clipped")
    assert not S.metric_is_higher_better("alarm_brier")     # an error, not a skill score


def _write(path: Path, tsfm, base, floor, metric):
    rows = []
    for seed, (t, b) in enumerate(zip(tsfm, base)):
        rows.append({"dataset": "M", "n_units": 4, "model": "chronos-2_mlp",
                     "seed": seed, "loss": "alarm", metric: t})
        rows.append({"dataset": "M", "n_units": 4, "model": "alarm_gbm",
                     "seed": seed, "loss": "native", metric: b})
        rows.append({"dataset": "M", "n_units": 4, "model": "alarm_base_rate",
                     "seed": seed, "loss": "native", metric: floor})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


def test_win_tie_loss_hollow_all_fire_on_a_higher_is_better_metric(tmp_path):
    better = [0.70, 0.74, 0.69, 0.76, 0.72]
    worse = [0.50, 0.58, 0.47, 0.55, 0.52]
    cfg = Config(dataset="FD001")

    win = S.success_map(_write(tmp_path / "w.csv", better, worse, 0.10, "alarm_ap"),
                        cfg, metric="alarm_ap")
    assert win[0]["verdict"] == "win" and win[0]["margin"] > 0

    loss = S.success_map(_write(tmp_path / "l.csv", worse, better, 0.05, "alarm_ap"),
                         cfg, metric="alarm_ap")
    assert loss[0]["verdict"] == "loss" and loss[0]["margin"] < 0

    # beats the competitor but not the base-rate floor -> hollow (guard direction flips)
    hollow = S.success_map(_write(tmp_path / "h.csv", better, worse, 0.95, "alarm_ap"),
                           cfg, metric="alarm_ap")
    assert hollow[0]["verdict"] == "hollow"

    tie = S.success_map(_write(tmp_path / "t.csv", better, better, 0.1, "alarm_ap"),
                        cfg, metric="alarm_ap")
    assert tie[0]["verdict"] == "tie"


def test_alarm_base_rate_is_a_floor_not_a_competitor():
    assert "alarm_base_rate" in S.FLOOR_MODELS
    assert not S.is_competitor_baseline("alarm_base_rate")
    assert S.is_competitor_baseline("alarm_gbm")
    assert S.is_tsfm_model("chronos-2_probe")          # the RQ-F rows


# ---------------------------------------------------------------------------
# (f) the sweep + campaign routing, end to end on synthetic C-MAPSS
# ---------------------------------------------------------------------------
def _alarm_cfg(tmp_path: Path, **over) -> Config:
    data_dir = tmp_path / "CMAPSSData"
    write_synthetic_cmapss(data_dir, dataset="FD001", n_train_units=6, n_test_units=4,
                           min_cycles=25, max_cycles=40, seed=11)
    base = dict(dataset="FD001", data_dir=str(data_dir),
                cache_dir=str(tmp_path / "cache"), results_dir=str(tmp_path / "results"),
                window_size=6, max_rul=60, alarm_horizon=12,
                sensor_columns=["s_2", "s_3", "s_4"],
                data_unit_counts=[2], sweep_seeds=[0, 1], head_hidden_dim=8,
                head_batch_size=16, head_max_epochs=2, head_early_stopping_patience=1)
    base.update(over)
    return Config(**base)


def test_alarm_labels_from_rul_matches_the_head_target(tmp_path):
    cfg = _alarm_cfg(tmp_path)
    rul = np.array([0.0, 12.0, 12.5, 59.0])
    from_sweep = SW.alarm_labels_from_rul(rul, cfg)
    from_head = H.alarm_targets(torch.tensor(rul), cfg).numpy()
    assert np.array_equal(from_sweep, from_head), "head and baselines must agree"


def test_run_alarm_sweep_end_to_end(tmp_path):
    from src.embeddings import build_embedding_cache
    cfg = _alarm_cfg(tmp_path)
    build_embedding_cache(cfg, embedder=MockEmbedder(feature_dim=8), verbose=False)
    out = SW.run_alarm_sweep(cfg, device="cpu",
                             baseline_names=["alarm_base_rate", "alarm_gbm"])
    rows = list(csv.DictReader(open(out)))
    assert out.name == "alarm_results.csv"
    assert {r["model"] for r in rows} == {"chronos-2_mlp", "alarm_base_rate", "alarm_gbm"}
    assert {r["loss"] for r in rows} == {H.ALARM_LOSS, "native"}
    for r in rows:
        assert int(r["alarm_horizon"]) == 12
        assert set(E.ALARM_METRIC_FIELDS) <= set(r)
        assert not (set(E.METRIC_FIELDS) & set(r)), "no RUL column may appear"
    # restartable: a rerun adds nothing
    before = len(rows)
    SW.run_alarm_sweep(cfg, device="cpu",
                       baseline_names=["alarm_base_rate", "alarm_gbm"])
    assert len(list(csv.DictReader(open(out)))) == before


def test_run_alarm_sweep_guards_its_inputs(tmp_path):
    from src.embeddings import build_embedding_cache
    no_horizon = _alarm_cfg(tmp_path, alarm_horizon=None)
    with pytest.raises(ValueError, match="requires config.alarm_horizon"):
        SW.run_alarm_sweep(no_horizon, cache={}, device="cpu")
    cfg = _alarm_cfg(tmp_path)
    build_embedding_cache(cfg, embedder=MockEmbedder(feature_dim=8), verbose=False)
    with pytest.raises(ValueError, match="ALARM baselines"):
        SW.run_alarm_sweep(cfg, device="cpu", baseline_names=["gbm"])


def test_campaign_skips_rul_only_stages_on_a_censored_fleet(tmp_path, capsys):
    """A censored dataset must route to the alarm sweep and SKIP fairness/horizon with a
    notice -- cycle_reg has no RUL to regress and horizon bins are RUL bands."""
    from src import campaign as C
    cfg = _alarm_cfg(tmp_path)
    # Exercise the routing directly: mark this config object as a censored fleet.
    censored = cfg.replace()
    object.__setattr__(censored, "is_censored_dataset", lambda: True)
    artifacts = C._run_stages(censored, ["cache", "sweep", "fairness", "horizon",
                                         "figures"], "cpu",
                              lambda c: MockEmbedder(feature_dim=8), None)
    assert "alarm_csv" in artifacts and "results_csv" not in artifacts
    assert "fairness_csv" not in artifacts and "horizon_csv" not in artifacts
    assert "figures" in artifacts, "the figures stage must still run"
    assert "skipping RUL-only stage" in capsys.readouterr().out


def test_plot_alarm_scaling_renders_and_skips_degenerate_metrics(tmp_path):
    """The alarm figure is a SEPARATE renderer reading a SEPARATE file. A metric that is
    nan everywhere (a degenerate cell under heavy imbalance) is skipped, not plotted."""
    import matplotlib
    matplotlib.use("Agg")
    from src import plots as P
    path = tmp_path / "alarm_results.csv"
    rows = []
    for n_units in (2, 4):
        for seed in range(3):
            for model, ap in (("chronos-2_mlp", 0.7), ("alarm_gbm", 0.5),
                              ("alarm_base_rate", 0.1)):
                rows.append({"dataset": "MetroPT-3", "n_units": n_units, "seed": seed,
                             "model": model, "loss": "native",
                             "alarm_ap": ap + 0.01 * seed, "alarm_auroc": 0.8,
                             "alarm_recall": 0.6,
                             "alarm_mean_lead_time": float("nan")})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    saved = P.plot_alarm_scaling(path, tmp_path / "figs", show=False)
    names = {p.stem for p in saved}
    assert "alarm_scaling_alarm_ap" in names
    assert "alarm_scaling_alarm_auroc" in names
    # every value of the lead-time metric is nan -> no figure for it
    assert not any("mean_lead_time" in n for n in names)
    assert all(p.exists() for p in saved)


def test_plot_alarm_threshold_curve_renders(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from src import plots as P
    sweep = E.alarm_threshold_sweep([1, 1, 0, 0], [0.9, 0.6, 0.4, 0.1],
                                    [0.2, 0.5, 0.8])
    saved = P.plot_alarm_threshold_curve(sweep, tmp_path / "figs", show=False)
    assert saved and all(p.exists() for p in saved)


# ---------------------------------------------------------------------------
# Phase-B family routing + the family-scoped key blocks (§54-§56)
# ---------------------------------------------------------------------------
def test_phase_b_families_route_and_declare_censoring():
    kinds = {"MetroPT-3": ("metropt", True), "Hydraulic": ("hydraulic", False),
             "Backblaze": ("backblaze", True)}
    for ds, (kind, censored) in kinds.items():
        cfg = Config(dataset=ds)
        assert cfg.dataset_kind() == kind
        assert cfg.is_censored_dataset() is censored
        assert cfg.sensor_columns == list(cfg.default_sensor_columns())
        assert len(cfg.sensor_columns) > 0
    for ds in ("FD001", "XJTU-SY", "DS02"):
        assert Config(dataset=ds).is_censored_dataset() is False


def test_unknown_dataset_names_the_new_families():
    with pytest.raises(ValueError, match="MetroPT-3"):
        Config(dataset="NotADataset")


def test_backblaze_smart_columns_are_the_channel_set():
    """Fleet-scale RQ-C: which SMART attributes you record IS the channel choice."""
    cfg = Config(dataset="Backblaze", backblaze_smart_columns=["smart_5_raw",
                                                               "smart_9_raw"])
    assert cfg.sensor_columns == ["smart_5_raw", "smart_9_raw"]
    assert cfg.num_channels() == 2


def test_hydraulic_channel_set_follows_its_stat_set():
    assert len(Config(dataset="Hydraulic").sensor_columns) == 17 * 2
    rich = Config(dataset="Hydraulic", ncmapss_agg_stats="mean_std",
                  hydraulic_agg_stats="mean_std_minmax_slope")
    assert len(rich.sensor_columns) == 17 * 5


@pytest.mark.parametrize("ds,field,value", [
    ("MetroPT-3", "metropt_cycle_minutes", 30),
    ("MetroPT-3", "metropt_min_samples_per_cycle", 5),
    ("MetroPT-3", "metropt_test_runs", [3]),
    ("MetroPT-3", "metropt_test_truncation", 0.5),
    ("Hydraulic", "hydraulic_drop_unstable", False),
    ("Hydraulic", "hydraulic_agg_stats", "mean_std_minmax_slope"),
    ("Hydraulic", "hydraulic_test_fraction", 0.4),
    ("Backblaze", "backblaze_models", ["ST4000DM000"]),
    ("Backblaze", "backblaze_start_date", "2024-01-01"),
    ("Backblaze", "backblaze_end_date", "2024-06-30"),
    ("Backblaze", "backblaze_min_days", 10),
    ("Backblaze", "backblaze_max_survivors_per_model", 50),
    ("Backblaze", "backblaze_test_fraction", 0.5),
    ("Backblaze", "seed", 99),
])
def test_every_phase_b_data_shaping_field_rekeys_its_own_family(ds, field, value):
    """Each field that changes what the loader produces MUST change that family's
    window key -- otherwise two different datasets would share one cache."""
    base = Config(dataset=ds, sensor_columns=None)
    changed = Config(dataset=ds, sensor_columns=None, **{field: value})
    assert changed.window_cache_key() != base.window_cache_key(), field


def test_phase_b_fields_never_rekey_the_recorded_families():
    """Family-scoped by construction: setting every Phase-B knob at once must leave
    FD001 / XJTU-SY / DS02 byte-identical."""
    over = dict(metropt_cycle_minutes=15, metropt_min_samples_per_cycle=2,
                metropt_test_runs=[1], metropt_test_truncation=0.4,
                hydraulic_drop_unstable=False, hydraulic_agg_stats="mean_std_minmax_slope",
                hydraulic_test_fraction=0.5, backblaze_models=["X"],
                backblaze_start_date="2020-01-01", backblaze_min_days=5,
                backblaze_max_survivors_per_model=None, backblaze_test_fraction=0.5)
    for ds, expected in (("FD001", "windows_FD001_1da313c871251cec"),
                         ("XJTU-SY", "windows_XJTU-SY_97e96700cc2670b4"),
                         ("DS02", "windows_DS02_ba4dfa4567c86cba")):
        assert Config(dataset=ds, **over).window_cache_key() == expected


def test_config_rejects_bad_phase_b_values():
    with pytest.raises(ValueError, match="hydraulic_agg_stats"):
        Config(dataset="Hydraulic", hydraulic_agg_stats="median")
    with pytest.raises(ValueError, match="hydraulic_taxonomy_component"):
        Config(dataset="Hydraulic", hydraulic_taxonomy_component="turbine")
    with pytest.raises(ValueError, match="metropt_cycle_minutes"):
        Config(dataset="MetroPT-3", metropt_cycle_minutes=0)


def test_classifier_baseline_abstract_features_raise():
    """The shared classifier base declares its feature hook abstract; a subclass that
    forgets to supply one must fail loudly, not silently score nothing."""
    with pytest.raises(NotImplementedError):
        B._ClassifierBaseline._features(np.zeros((1, 2, 1), np.float32))


def test_run_alarm_sweep_accepts_a_preloaded_cache(tmp_path):
    """The ``cache is not None`` branch: a caller that already holds the Stage-A cache
    must not trigger a reload."""
    from src.embeddings import build_embedding_cache, load_embedding_cache
    cfg = _alarm_cfg(tmp_path)
    build_embedding_cache(cfg, embedder=MockEmbedder(feature_dim=8), verbose=False)
    cache = load_embedding_cache(cfg)
    out = SW.run_alarm_sweep(cfg, cache=cache, device="cpu",
                             baseline_names=["alarm_base_rate"],
                             results_csv=tmp_path / "preloaded.csv",
                             run_dir=tmp_path / "preloaded_runs")
    assert out.name == "preloaded.csv"
    assert list(csv.DictReader(open(out)))


def test_metropt_bin_coverage_is_a_validated_fraction():
    """The gap defence is expressed as a FRACTION of a bin (so it is invariant under the
    RQ-G bin-width sweep), which only means anything inside [0, 1]."""
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="metropt_min_bin_coverage"):
            Config(dataset="MetroPT-3", metropt_min_bin_coverage=bad)
    for ok in (0.0, 0.5, 1.0):
        assert Config(dataset="MetroPT-3",
                      metropt_min_bin_coverage=ok).metropt_min_bin_coverage == ok
    # and it re-keys the MetroPT aggregate/window cache, since it changes which bins exist
    a = Config(dataset="MetroPT-3", metropt_min_bin_coverage=0.5)
    b = Config(dataset="MetroPT-3", metropt_min_bin_coverage=0.9)
    assert a.window_cache_key() != b.window_cache_key()
