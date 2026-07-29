"""Horizon (Stage A-H / B-H) and campaign-orchestration branches.

The horizon sidecar cache has its own registry-resolved embedder path and its own
metrics/predictions file-sync guard (CHANGES.md §20); the campaign runs a partial stage
list per combo. Both are exercised here on synthetic C-MAPSS with a mock embedder --
CPU-only, no downloads, no backbone import.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

from src.config import Config
from src import horizon as H
from src.campaign import run_campaign
from src.embeddings import build_embedding_cache, load_embedding_cache
from tests.synthetic import write_synthetic_cmapss, write_synthetic_xjtu, MockEmbedder


class _ThroughputMock(MockEmbedder):
    """MockEmbedder reporting a throughput, so Stage A-H's progress print is exercised."""
    def embed_windows(self, contexts):
        out = super().embed_windows(contexts)
        self.last_throughput = 42.0
        return out


def _cfg(tmp_path: Path, **over) -> Config:
    base = dict(
        dataset="FD001",
        data_dir=str(tmp_path / "CMAPSSData"),
        cache_dir=str(tmp_path / "cache"),
        results_dir=str(tmp_path / "results"),
        window_size=12,
        sensor_columns=["s_2", "s_3", "s_4", "s_7", "s_9"],
        max_rul=40, num_bins=8,
        data_unit_counts=[2], sweep_seeds=[0], losses=["mse"],
        head_hidden_dim=16, head_batch_size=32, head_max_epochs=2,
        head_early_stopping_patience=1,
        baseline_max_epochs=2, baseline_early_stopping_patience=1,
    )
    base.update(over)
    return Config(**base)


# ---------------------------------------------------------------------------
# horizon: Stage A-H registry path, sync guard, pre-loaded caches
# ---------------------------------------------------------------------------
def test_build_horizon_cache_uses_the_registry_when_no_embedder_is_injected(
        tmp_path, monkeypatch, capsys):
    """embedder=None resolves through models.make_embedder (the production Stage A-H
    path); the registry is monkeypatched to a CPU mock so no backbone is imported."""
    import src.models as models_mod
    cfg = _cfg(tmp_path)
    write_synthetic_cmapss(Path(cfg.data_dir), n_train_units=5, n_test_units=4,
                           min_cycles=20, max_cycles=30)
    monkeypatch.setattr(models_mod, "make_embedder",
                        lambda c, device=None: _ThroughputMock(feature_dim=12))
    path = H.build_horizon_cache(cfg, verbose=True)
    assert path.exists() and path.with_suffix(".json").exists()
    assert "test-all-cycles embed throughput" in capsys.readouterr().out


def test_load_horizon_cache_requires_stage_a_h(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run build_horizon_cache first"):
        H.load_horizon_cache(_cfg(tmp_path))


def test_run_horizon_eval_accepts_preloaded_caches_and_skips_oversized_counts(tmp_path):
    """Both caches can be handed in (the campaign path), and a unit count above the
    fleet is skipped rather than sampled."""
    cfg = _cfg(tmp_path)
    write_synthetic_cmapss(Path(cfg.data_dir), n_train_units=5, n_test_units=4,
                           min_cycles=20, max_cycles=30)
    embedder = MockEmbedder(feature_dim=12)
    build_embedding_cache(cfg, embedder=embedder)
    H.build_horizon_cache(cfg, embedder=embedder)
    out = H.run_horizon_eval(cfg, cache=load_embedding_cache(cfg),
                             hcache=H.load_horizon_cache(cfg),
                             n_units_list=[4, 999], seeds=[0], baseline_names=[],
                             bin_edges=(0, 20, float("inf")))
    counts = {int(r["n_units"]) for r in csv.DictReader(open(out))}
    assert counts == {4}                       # 999 > 5 available units -> skipped


def test_run_horizon_eval_refuses_metrics_without_their_predictions(tmp_path):
    """§20: horizon.csv and horizon_predictions.csv must be archived TOGETHER. If the
    metrics file keeps cells whose predictions were deleted, a rerun would silently skip
    them (leaving trajectory plots broken) -- so it is a hard error naming the remedy."""
    cfg = _cfg(tmp_path)
    write_synthetic_cmapss(Path(cfg.data_dir), n_train_units=5, n_test_units=4,
                           min_cycles=20, max_cycles=30)
    embedder = MockEmbedder(feature_dim=12)
    build_embedding_cache(cfg, embedder=embedder)
    H.build_horizon_cache(cfg, embedder=embedder)
    H.run_horizon_eval(cfg, n_units_list=[4], seeds=[0], baseline_names=[],
                       bin_edges=(0, 20, float("inf")))
    cfg.results_path("horizon_predictions.csv").unlink()      # archive only one file
    with pytest.raises(ValueError, match="out of sync"):
        H.run_horizon_eval(cfg, n_units_list=[4], seeds=[0], baseline_names=[],
                           bin_edges=(0, 20, float("inf")))


# ---------------------------------------------------------------------------
# campaign: partial stage lists + user dataset overrides
# ---------------------------------------------------------------------------
def test_campaign_runs_the_horizon_stage_without_a_sweep(tmp_path):
    """Stage selection is a-la-carte: a second pass can run only `horizon` + `figures`
    over caches an earlier `cache` pass wrote, and the figures stage renders whatever
    artifacts that pass produced (here: horizon only, no data-scaling curve)."""
    base = _cfg(tmp_path, data_root=str(tmp_path / "Data"), data_dir=None)
    write_synthetic_cmapss(Path(base.data_root) / "CMAPSSData", dataset="FD001",
                           n_train_units=5, n_test_units=4, min_cycles=20, max_cycles=30)
    embedders: dict[str, MockEmbedder] = {}
    factory = lambda c: embedders.setdefault(c.experiment_name, MockEmbedder(12))

    run_campaign(base, datasets=["FD001"], models=["amazon/chronos-2"],
                 stages=("cache",), embedder_factory=factory)
    summary = run_campaign(base, datasets=["FD001"], models=["amazon/chronos-2"],
                           stages=("horizon", "figures"), embedder_factory=factory)
    combo = summary[0]
    assert combo["status"] == "ok"
    assert "results_csv" not in combo and "horizon_csv" in combo
    assert Path(combo["horizon_csv"]).exists()
    figs = [Path(f).name for f in combo["figures"]]
    assert figs and all(f.startswith("FD001_chronos-2_horizon_") for f in figs)


def test_campaign_merges_user_dataset_overrides(tmp_path):
    """A non-empty `dataset_overrides` is merged OVER the recorded defaults, so the
    per-combo config carries the user's value and every artifact is named for it."""
    base = _cfg(tmp_path, data_root=str(tmp_path / "Data"), data_dir=None)
    write_synthetic_cmapss(Path(base.data_root) / "CMAPSSData", dataset="FD001",
                           n_train_units=5, n_test_units=4)
    embedders: dict[str, MockEmbedder] = {}
    factory = lambda c: embedders.setdefault(c.experiment_name, MockEmbedder(12))
    summary = run_campaign(base, datasets=["FD001"], models=["amazon/chronos-2"],
                           stages=("cache", "sweep"),
                           dataset_overrides={"FD001": {"max_rul": 30}},
                           embedder_factory=factory, baseline_names=["predict_mean"])
    rows = list(csv.DictReader(open(summary[0]["results_csv"])))
    assert {r["max_rul"] for r in rows} == {"30"}          # the user override took effect
    assert all(np.isfinite(float(r["rmse_clipped"])) for r in rows)


def test_campaign_empty_dataset_overrides_opts_out_of_the_recorded_defaults(tmp_path):
    """`dataset_overrides={}` is the explicit opt-out (CHANGES.md §30): even XJTU-SY,
    which HAS recorded defaults (max_rul=125, window_size=30, context 256), runs at the
    base config's own protocol instead."""
    base = _cfg(tmp_path, dataset="XJTU-SY", data_root=str(tmp_path / "Data"),
                data_dir=None, sensor_columns=None, max_rul=40, window_size=12,
                xjtu_test_bearings=["Bearing1_3", "Bearing2_3", "Bearing3_3"])
    write_synthetic_xjtu(Path(base.data_root) / "XJTU-SY", bearings_per_condition=3,
                         min_snapshots=18, max_snapshots=24, samples_per_snapshot=64)
    embedders: dict[str, MockEmbedder] = {}
    factory = lambda c: embedders.setdefault(c.experiment_name, MockEmbedder(12))
    summary = run_campaign(base, datasets=["XJTU-SY"], models=["amazon/chronos-2"],
                           stages=("cache", "sweep"), dataset_overrides={},
                           embedder_factory=factory, baseline_names=["predict_mean"])
    assert summary[0]["status"] == "ok"
    rows = list(csv.DictReader(open(summary[0]["results_csv"])))
    assert {r["max_rul"] for r in rows} == {"40"}          # NOT the recorded 125
    assert {r["window_size"] for r in rows} == {"12"}      # NOT the recorded 30
