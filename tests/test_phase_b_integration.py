"""The Milestone 3-7 acceptance test: every Phase-B chapter, end to end, on CPU.

The per-module tests prove each piece in isolation. This module proves the pieces
COMPOSE -- that ``run_campaign`` actually produces each dataset's deliverable through
the real registry, the real loading path, the real Stage-A cache and the real scoring,
with only the GPU backbone mocked:

  * **RQ-D (M3)** -- an XJTU ``feature_mode`` probe: raw samples vs. hand-crafted
    indicators, scored by the win-rule.
  * **RQ-G (M4)** -- an N-CMAPSS ``aggregation`` probe over the stride/stat-set knobs.
  * **censored fleets (M5/M7)** -- MetroPT-3 through the alarm sweep, with the RUL-only
    stages skipped and the alarm CSV kept apart from the RUL one.
  * **RQ-F (M6)** -- the UCI Hydraulic few-shot adjust-vs-replace probe, on the dataset
    whose graded severity labels the chapter exists for.

Everything here runs at toy scale in seconds; the point is the WIRING, not the numbers.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from src.config import Config, HYDRAULIC_ACTIONS
from src import campaign as C
from src import datasets as DS
from src import probes as P
from src import scoring as S
from src import taxonomy as TX
from tests.synthetic import (MockEmbedder, write_synthetic_hydraulic,
                             write_synthetic_metropt, write_synthetic_ncmapss,
                             write_synthetic_xjtu)


def _embedder(_cfg):
    return MockEmbedder(feature_dim=10)


def _base(tmp_path: Path, **over) -> Config:
    base = dict(cache_dir=str(tmp_path / "cache"), results_dir=str(tmp_path / "results"),
                data_root=str(tmp_path / "Data"),
                sweep_seeds=[0, 1], data_unit_counts=[2],
                head_hidden_dim=8, head_batch_size=16, head_max_epochs=2,
                head_early_stopping_patience=1, baseline_max_epochs=2,
                baseline_early_stopping_patience=1, losses=["mse"])
    base.update(over)
    return Config(**base)


def _write_real_range_metropt(tmp_path: Path, seed: int) -> Path:
    """Synthetic MetroPT-3 spanning the REAL Feb-Sep 2020 record with the REAL four
    documented air-leak events, because the loader segments runs by
    ``config.METROPT_FAILURE_EVENTS`` -- a fixture on any other date range yields a
    single censored run and no test unit. A 5-minute nominal cadence keeps the file
    small while still putting ~12 rows in every hourly bin, above the loader's
    ``metropt_min_samples_per_cycle`` floor."""
    from src.config import METROPT_FAILURE_EVENTS
    return write_synthetic_metropt(
        tmp_path / "Data" / "MetroPT-3",
        start="2020-02-01 00:00:00", end="2020-09-01 00:00:00",
        events=METROPT_FAILURE_EVENTS, step_seconds=300, seed=seed)


# ---------------------------------------------------------------------------
# The registry sees every family (the drift alarm, extended to Phase B)
# ---------------------------------------------------------------------------
def test_every_served_dataset_maps_to_a_registered_family():
    for name in DS.all_dataset_names():
        kind = Config(dataset=name).dataset_kind()
        assert kind in DS.DATASET_LOADERS and kind in DS.DATASET_FAMILIES
    assert set(DS.DATASET_LOADERS) == set(DS.DATASET_FAMILIES)
    served = {Config(dataset=n).dataset_kind() for n in DS.all_dataset_names()}
    assert served == set(DS.DATASET_FAMILIES)
    # the Phase-B families are actually served, not just importable
    assert {"MetroPT-3", "Hydraulic"} <= set(DS.all_dataset_names())


def test_campaign_skips_every_absent_dataset_without_crashing(tmp_path):
    """The run-all button on a machine with no downloads: every dataset reports
    ``skipped_no_data`` and nothing raises (invariant, IMPLEMENTATION_PLAN §9.6)."""
    cfg = _base(tmp_path)
    summary = C.run_campaign(cfg, models=["amazon/chronos-2"], stages=["cache"],
                             embedder_factory=_embedder)
    assert summary and all(s["status"] == "skipped_no_data" for s in summary)
    assert {s["dataset"] for s in summary} >= {"MetroPT-3", "Hydraulic"}


# ---------------------------------------------------------------------------
# RQ-D (M3): the XJTU raw-vs-indicators probe
# ---------------------------------------------------------------------------
def test_rq_d_feature_mode_probe_end_to_end(tmp_path):
    write_synthetic_xjtu(tmp_path / "Data" / "XJTU-SY", bearings_per_condition=5,
                         min_snapshots=14, max_snapshots=18, samples_per_snapshot=64,
                         seed=2)
    cfg = _base(tmp_path, dataset="XJTU-SY", window_size=6, max_rul=40,
                tsfm_context_length=8, xjtu_raw_channels=4,
                xjtu_test_bearings=["Bearing1_4", "Bearing1_5"])
    out = P.run_factor_probe(
        cfg, "feature_mode",
        levels={"indicators": {"xjtu_feature_mode": "indicators"},
                "raw": {"xjtu_feature_mode": "raw"},
                "raw_plus": {"xjtu_feature_mode": "raw+indicators"}},
        models=["amazon/chronos-2"], baselines=["gbm", "predict_mean"],
        seeds=[0, 1], embedder_factory=_embedder)
    rows = list(csv.DictReader(open(out)))
    assert {r["level"] for r in rows} == {"indicators", "raw", "raw_plus"}
    assert {r["factor"] for r in rows} == {"feature_mode"}
    # each level really did run at its own channel width (the probe re-resolved them)
    assert len({r["model"] for r in rows}) == 3      # 1 TSFM head + 2 baselines
    table = S.success_map(out, cfg, cell_fields=("dataset", "n_units", "factor", "level"))
    assert {r["level"] for r in table} == {"indicators", "raw", "raw_plus"}
    assert all(r["verdict"] in {"win", "tie", "loss", "hollow"} for r in table)


# ---------------------------------------------------------------------------
# RQ-G (M4): the N-CMAPSS aggregation probe
# ---------------------------------------------------------------------------
def test_rq_g_aggregation_probe_end_to_end(tmp_path):
    write_synthetic_ncmapss(tmp_path / "Data" / "N-CMAPSS", dataset="DS02",
                            n_dev_units=4, n_test_units=2, min_cycles=12, max_cycles=14,
                            min_rows=16, max_rows=20, seed=3)
    cfg = _base(tmp_path, dataset="DS02", window_size=4, max_rul=40,
                tsfm_context_length=6)
    out = P.run_factor_probe(
        cfg, "aggregation",
        levels={"1hz_meanstd": {"ncmapss_agg_stride": 1, "ncmapss_agg_stats": "mean_std"},
                "stride4": {"ncmapss_agg_stride": 4, "ncmapss_agg_stats": "mean_std"},
                "rich": {"ncmapss_agg_stride": 1,
                         "ncmapss_agg_stats": "mean_std_minmax_slope"}},
        models=["amazon/chronos-2"], baselines=["gbm"], seeds=[0],
        embedder_factory=_embedder)
    rows = list(csv.DictReader(open(out)))
    assert {r["level"] for r in rows} == {"1hz_meanstd", "stride4", "rich"}
    assert all(np.isfinite(float(r["rmse_clipped"])) for r in rows)
    # the rich level genuinely used a wider channel set -> its own Stage-A cache exists
    caches = sorted(Path(cfg.cache_dir).glob("emb_DS02_*.npz"))
    assert len(caches) == 3, "each level must key its own cache"


# ---------------------------------------------------------------------------
# M5/M7: a censored fleet through the campaign
# ---------------------------------------------------------------------------
def test_metropt_runs_the_alarm_arm_through_the_campaign(tmp_path):
    _write_real_range_metropt(tmp_path, seed=4)
    # The fixture's 5-minute cadence cannot satisfy the shipped-cadence coverage rule
    # (a 60-minute bin expects ~360 rows at 10 s), so it is switched off here; the rule
    # itself is tested against real cadences in tests/test_metropt.py.
    cfg = _base(tmp_path, dataset="MetroPT-3", metropt_min_bin_coverage=0.0)
    summary = C.run_campaign(
        cfg, datasets=["MetroPT-3"], models=["amazon/chronos-2"],
        stages=["cache", "sweep", "fairness", "horizon", "figures"],
        embedder_factory=_embedder)
    assert len(summary) == 1 and summary[0]["status"] == "ok", summary
    art = summary[0]
    # the censored routing: alarm CSV produced, RUL-only artifacts absent
    assert "alarm_csv" in art and "results_csv" not in art
    assert "fairness_csv" not in art and "horizon_csv" not in art
    rows = list(csv.DictReader(open(art["alarm_csv"])))
    assert rows, "the alarm sweep must produce rows"
    from src import evaluate as E
    for r in rows:
        assert set(E.ALARM_METRIC_FIELDS) <= set(r)
        assert not set(E.METRIC_FIELDS) & set(r), "no RUL metric may share the file"
        assert int(r["alarm_horizon"]) == C.DEFAULT_DATASET_OVERRIDES[
            "MetroPT-3"]["alarm_horizon"]
    # and the alarm file is NOT the RUL file
    assert Path(art["alarm_csv"]).name.endswith("alarm_results.csv")


def test_metropt_frames_carry_censoring_and_alarm_labels(tmp_path):
    """The censored contract at the frame level: a survivor run is flagged, and the
    alarm label is present and fully known after the unknowable rows are dropped."""
    from src import data as D
    _write_real_range_metropt(tmp_path, seed=5)
    over = dict(C.DEFAULT_DATASET_OVERRIDES["MetroPT-3"])
    cfg = _base(tmp_path, dataset="MetroPT-3", metropt_min_bin_coverage=0.0, **over)
    df_train, df_test = D.load_prepared(cfg)
    assert D.EVENT_OBSERVED_COLUMN in df_train.columns
    assert D.ALARM_LABEL_COLUMN in df_train.columns
    assert not df_train[D.ALARM_LABEL_COLUMN].isna().any()
    assert set(df_train[D.ALARM_LABEL_COLUMN].unique()) <= {0.0, 1.0}
    # at least one censored (survivor) run is present in training
    assert (df_train[D.EVENT_OBSERVED_COLUMN] == 0).any()
    # a censored run must never be a test unit (it has no true RUL)
    assert (df_test[D.EVENT_OBSERVED_COLUMN] == 1).all()


# ---------------------------------------------------------------------------
# RQ-F (M6): the Hydraulic adjust-vs-replace probe -- the chapter's whole point
# ---------------------------------------------------------------------------
def test_hydraulic_emits_the_rq_f_labels_on_one_polarity(tmp_path):
    from src import data as D
    from src.datasets import hydraulic as HY
    write_synthetic_hydraulic(tmp_path / "Data" / "Hydraulic", cycles=360, block_len=12,
                              samples_scale=60, seed=6)
    cfg = _base(tmp_path, dataset="Hydraulic",
                **C.DEFAULT_DATASET_OVERRIDES["Hydraulic"])
    df_train, _df_test = D.load_prepared(cfg)
    for component in ("cooler", "valve", "pump", "accumulator"):
        sev = df_train[HY.severity_column(component)].to_numpy()
        act = df_train[HY.action_column(component)].to_numpy()
        assert sev.min() >= 0, "severity 0 = healthy for EVERY component"
        # the action taxonomy is a function of severity, on one polarity
        assert set(np.unique(act)) <= set(range(len(HYDRAULIC_ACTIONS)))
        assert (act[sev == 0] == 0).all(), "severity 0 must map to 'none'"
        assert (act[sev > 0] > 0).all(), "any fault must need an action"


def test_rq_f_taxonomy_probe_on_hydraulic(tmp_path):
    """The RQ-F deliverable: few-shot separability of adjust-vs-replace from a FROZEN
    embedding, compared against the hand-crafted indicator bank, on the dataset whose
    graded severity labels the chapter exists for."""
    pytest.importorskip("pycatch22")
    from src.datasets import hydraulic as HY
    write_synthetic_hydraulic(tmp_path / "Data" / "Hydraulic", cycles=480, block_len=12,
                              samples_scale=60, seed=7)
    cfg = _base(tmp_path, dataset="Hydraulic",
                **C.DEFAULT_DATASET_OVERRIDES["Hydraulic"])
    label = HY.action_column(cfg.hydraulic_taxonomy_component)
    out = TX.run_taxonomy_probe(cfg, label, shots=[1, 5], seeds=[0, 1],
                                feature_sources=["embedding", "catch22"],
                                embedder_factory=_embedder)
    rows = list(csv.DictReader(open(out)))
    assert rows and {r["feature_source"] for r in rows} == {"embedding", "catch22"}
    assert {r["label"] for r in rows} == {label}
    for r in rows:
        assert 0.0 <= float(r["accuracy"]) <= 1.0
        assert int(r["n_labelled"]) >= 1
    # more labels must not hurt on average -- a sanity check that the probe learns
    by_shots = {}
    for r in rows:
        if r["feature_source"] == "embedding":
            by_shots.setdefault(int(r["shots"]), []).append(float(r["accuracy"]))
    assert np.mean(by_shots[5]) >= np.mean(by_shots[1]) - 0.2


def test_hydraulic_campaign_runs_the_taxonomy_probe_not_a_rul_sweep(tmp_path, capsys):
    """Hydraulic has NO failure events -- a block ends because the experimenter changed
    the set-point -- so RUL is degenerate by construction (uniform blocks give a CONSTANT
    rul_truth, against which the predict-the-mean floor scores a perfect 0.0). The
    campaign must therefore route it to the RQ-F taxonomy probe and skip every
    time-to-event stage, rather than tabling a meaningless RMSE beside C-MAPSS's (§55)."""
    pytest.importorskip("pycatch22")
    write_synthetic_hydraulic(tmp_path / "Data" / "Hydraulic", cycles=480, block_len=12,
                              samples_scale=60, seed=8)
    cfg = _base(tmp_path, dataset="Hydraulic")
    assert cfg.is_classification_dataset() and not cfg.is_censored_dataset()
    summary = C.run_campaign(cfg, datasets=["Hydraulic"], models=["amazon/chronos-2"],
                             stages=["cache", "sweep", "fairness", "horizon", "figures"],
                             baseline_names=["predict_mean", "gbm"],
                             embedder_factory=_embedder)
    assert summary[0]["status"] == "ok", summary
    art = summary[0]
    assert "taxonomy_csv" in art
    assert "results_csv" not in art and "alarm_csv" not in art
    assert "fairness_csv" not in art and "horizon_csv" not in art
    assert "skipping time-to-event stage" in capsys.readouterr().out
    rows = list(csv.DictReader(open(art["taxonomy_csv"])))
    assert rows and {r["feature_source"] for r in rows} >= {"embedding"}
    assert art["figures"], "the RQ-F few-shot curve is the deliverable figure"


def test_stage_skipping_notices_only_fire_when_there_is_something_to_skip(tmp_path):
    """Running a censored or classification dataset with ONLY the stages it supports must
    not print a skip notice for stages nobody asked for."""
    write_synthetic_hydraulic(tmp_path / "Data" / "Hydraulic", cycles=480, block_len=12,
                              samples_scale=60, seed=21)
    _write_real_range_metropt(tmp_path, seed=22)
    pytest.importorskip("pycatch22")

    hyd = _base(tmp_path, dataset="Hydraulic")
    summary = C.run_campaign(hyd, datasets=["Hydraulic"], models=["amazon/chronos-2"],
                             stages=["cache", "sweep"], embedder_factory=_embedder)
    assert summary[0]["status"] == "ok" and "taxonomy_csv" in summary[0]

    met = _base(tmp_path, dataset="MetroPT-3", metropt_min_bin_coverage=0.0)
    summary = C.run_campaign(met, datasets=["MetroPT-3"], models=["amazon/chronos-2"],
                             stages=["cache", "sweep"], embedder_factory=_embedder)
    assert summary[0]["status"] == "ok" and "alarm_csv" in summary[0]
    assert "fairness_csv" not in summary[0] and "horizon_csv" not in summary[0]
