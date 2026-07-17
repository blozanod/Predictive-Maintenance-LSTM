"""CPU test for the run-all campaign (src/campaign.py, CHANGES.md §24):
datasets x models cross product, per-combo experiment naming, missing-data
skips, and failure isolation -- on synthetic data + MockEmbedder."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import pytest

from src.config import Config
from src.campaign import (run_campaign, campaign_experiment_name,
                         merge_dataset_overrides, DEFAULT_DATASET_OVERRIDES)
from tests.synthetic import write_synthetic_cmapss, MockEmbedder


def _base(tmp_path: Path) -> Config:
    return Config(
        dataset="FD001",
        data_root=str(tmp_path / "Data"),
        cache_dir=str(tmp_path / "cache"),
        results_dir=str(tmp_path / "results"),
        window_size=12,
        max_rul=40,
        num_bins=8,
        data_unit_counts=[2, 4],
        sweep_seeds=[0],
        head_hidden_dim=16,
        head_batch_size=32,
        head_max_epochs=2,
        head_early_stopping_patience=1,
        baseline_max_epochs=2,
        baseline_early_stopping_patience=1,
        losses=["mse"],
    )


def test_campaign_names_and_cross_product(tmp_path):
    base = _base(tmp_path)
    cmapss_dir = Path(base.data_root) / "CMAPSSData"
    for ds in ("FD001", "FD003"):
        write_synthetic_cmapss(cmapss_dir, dataset=ds, n_train_units=6,
                               n_test_units=4, seed=5)
    embedders: dict[str, MockEmbedder] = {}
    factory = lambda c: embedders.setdefault(c.experiment_name, MockEmbedder(16))

    summary = run_campaign(
        base, datasets=["FD001", "FD003", "XJTU-SY"],   # XJTU data absent
        stages=("cache", "sweep", "fairness", "figures"),
        embedder_factory=factory, baseline_names=["predict_mean"])

    by = {(s["dataset"], s["model"]): s for s in summary}
    assert by[("XJTU-SY", None)]["status"] == "skipped_no_data"
    for ds in ("FD001", "FD003"):
        s = by[(ds, "amazon/chronos-2")]
        assert s["status"] == "ok"
        # file naming carries dataset + TSFM tag
        exp = campaign_experiment_name(base, ds, "amazon/chronos-2")
        assert exp == f"{ds}_chronos-2"
        results = Path(s["results_csv"])
        assert results.name == f"{exp}_results_v2.csv" and results.exists()
        with open(results, newline="") as f:
            rows = list(csv.DictReader(f))
        assert {r["dataset"] for r in rows} == {ds}
        assert any(r["model"] == "cycle_reg" for r in rows)      # fairness stage ran
        figs = [Path(f).name for f in s["figures"]]
        assert any(f.startswith(f"{exp}_data_scaling_") for f in figs)

    # restartable: second run adds no rows (all cells complete)
    n_before = sum(1 for _ in open(by[("FD001", "amazon/chronos-2")]["results_csv"]))
    run_campaign(base, datasets=["FD001"], stages=("cache", "sweep", "fairness"),
                 embedder_factory=factory, baseline_names=["predict_mean"])
    n_after = sum(1 for _ in open(by[("FD001", "amazon/chronos-2")]["results_csv"]))
    assert n_after == n_before


def test_default_dataset_overrides_merge():
    """§30: None -> the recorded defaults; a user dict merges over them per key
    (user wins); an XJTU key without max_rul keeps the default max_rul."""
    # None -> a copy of the defaults (XJTU protocol pinned; DSALL members pinned)
    d = merge_dataset_overrides(None)
    assert d["XJTU-SY"] == DEFAULT_DATASET_OVERRIDES["XJTU-SY"]
    assert d["DSALL"]["dsall_datasets"][0] == "DS01"
    # user wins per key; unspecified keys keep the default
    merged = merge_dataset_overrides({"XJTU-SY": {"max_rul": 60},
                                      "FD002": {"condition_norm": True}})
    assert merged["XJTU-SY"]["max_rul"] == 60                 # user override
    assert merged["XJTU-SY"]["window_size"] == 30            # default preserved
    assert merged["FD002"] == {"condition_norm": True}       # user-only dataset kept
    # mutating the result never mutates the module constant
    merged["XJTU-SY"]["window_size"] = 999
    assert DEFAULT_DATASET_OVERRIDES["XJTU-SY"]["window_size"] == 30


def test_campaign_combo_config_applies_default_overrides(tmp_path):
    """The per-combo config carries the recorded XJTU protocol (default overrides),
    and sensor_columns resolves to the dataset default regardless."""
    from src.campaign import _combo_config, merge_dataset_overrides
    from src.config import XJTU_FEATURE_COLUMNS
    base = _base(tmp_path)
    over = merge_dataset_overrides(None)
    cfg = _combo_config(base, "XJTU-SY", "amazon/chronos-2", over)
    assert cfg.max_rul == 125 and cfg.window_size == 30
    assert cfg.tsfm_context_length == 256
    assert cfg.sensor_columns == list(XJTU_FEATURE_COLUMNS)   # dataset default
    # DSALL combo pins the member list (deterministic cache key); DS08d is excluded
    # by default because it is unreliably obtainable (CHANGES §31).
    dsall = _combo_config(base, "DSALL", "amazon/chronos-2", over)
    assert dsall.dsall_datasets[0] == "DS01" and len(dsall.dsall_datasets) == 9
    assert "DS08d" not in dsall.dsall_datasets


def test_campaign_isolates_failures_but_raises_when_all_fail(tmp_path):
    base = _base(tmp_path)
    cmapss_dir = Path(base.data_root) / "CMAPSSData"
    write_synthetic_cmapss(cmapss_dir, dataset="FD001", n_train_units=6,
                           n_test_units=4, seed=6)

    class BrokenEmbedder(MockEmbedder):
        def embed_windows(self, contexts):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="every campaign combo failed"):
        run_campaign(base, datasets=["FD001"], stages=("cache", "sweep"),
                     embedder_factory=lambda c: BrokenEmbedder(16),
                     baseline_names=["predict_mean"])

    # one broken combo among good ones is reported, not fatal
    embedders: dict[str, MockEmbedder] = {}

    def mixed(c: Config):
        if c.dataset == "FD003":
            return BrokenEmbedder(16)
        return embedders.setdefault(c.experiment_name, MockEmbedder(16))

    write_synthetic_cmapss(cmapss_dir, dataset="FD003", n_train_units=6,
                           n_test_units=4, seed=6)
    summary = run_campaign(base, datasets=["FD001", "FD003"],
                           stages=("cache", "sweep"), embedder_factory=mixed,
                           baseline_names=["predict_mean"])
    status = {s["dataset"]: s["status"] for s in summary}
    assert status == {"FD001": "ok", "FD003": "failed"}
