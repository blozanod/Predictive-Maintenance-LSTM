"""Sweep / feature-builder / transfer branches the happy-path smoke tests skip.

Covers the per-baseline window override (Task 1.5), the default baseline roster, the
pre-loaded-cache entry point, restart skipping in the ablation, the baseline-window
comparison runner, the head-feature builder's fit/transform contract, and transfer's
multi-condition warning + oversized-shot skip. CPU-only, mock embedder, tiny synthetic
C-MAPSS.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from src.config import Config
from src import sweep as S
from src import transfer as X
from src.embeddings import build_embedding_cache, load_embedding_cache
from src.features import HeadFeatureBuilder, raw_last_cycle
from tests.synthetic import write_synthetic_cmapss, MockEmbedder


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
# HeadFeatureBuilder
# ---------------------------------------------------------------------------
def test_head_feature_builder_requires_fit_before_transform():
    b = HeadFeatureBuilder(Config(head_features="emb+locscale"))
    with pytest.raises(RuntimeError, match="called before fit"):
        b.transform(torch.zeros(2, 4), torch.zeros(2, 3, 2), torch.zeros(2, 3))


def test_head_feature_builder_output_dim_matches_the_assembled_width():
    """output_dim is the head's declared input width; assert it equals what transform
    actually produces for each head_features mode."""
    emb = torch.randn(6, 5)
    locscale = torch.randn(6, 3, 2)
    raw = torch.randn(6, 3)
    for mode, expected in (("emb", 5), ("emb+locscale", 5 + 6), ("emb+locscale+raw", 5 + 6 + 3)):
        b = HeadFeatureBuilder(Config(head_features=mode)).fit(locscale, raw)
        assert b.output_dim(emb_dim=5, n_variates=3, n_channels=3) == expected
        assert b.transform(emb, locscale, raw).shape == (6, expected)
    assert raw_last_cycle(np.zeros((4, 12, 3))).shape == (4, 3)


# ---------------------------------------------------------------------------
# run_sweep: per-baseline window override, pre-loaded cache, default roster
# ---------------------------------------------------------------------------
def test_run_sweep_rewindows_only_the_overriding_baseline(tmp_path):
    """`baseline_windows` re-windows the RAW series for the named baseline only (equal-
    tuning-budget fairness); every other model keeps the cached window."""
    cfg = _cfg(tmp_path, baseline_windows={"predict_mean": 6})
    write_synthetic_cmapss(Path(cfg.data_dir), n_train_units=5, n_test_units=4)
    build_embedding_cache(cfg, embedder=MockEmbedder(feature_dim=12))
    out = S.run_sweep(cfg, baseline_names=["predict_mean"], device="cpu")
    rows = list(csv.DictReader(open(out)))
    windows = {r["baseline_window"] for r in rows if r["model"] == "predict_mean"}
    assert windows == {"6"}                       # the override, not window_size=12
    assert {r["baseline_window"] for r in rows if r["model"].endswith("_mlp")} == {""}


def test_run_sweep_accepts_a_preloaded_cache_and_the_default_roster(tmp_path):
    """Passing `cache=` skips the disk load (the campaign path), and omitting
    `baseline_names` runs the recorded default roster."""
    pytest.importorskip("lightgbm")
    pytest.importorskip("sktime")
    cfg = _cfg(tmp_path)
    write_synthetic_cmapss(Path(cfg.data_dir), n_train_units=5, n_test_units=4)
    build_embedding_cache(cfg, embedder=MockEmbedder(feature_dim=12))
    cache = load_embedding_cache(cfg)
    out = S.run_sweep(cfg, cache=cache, device="cpu")
    models = {r["model"] for r in csv.DictReader(open(out))}
    assert {"predict_mean", "gbm", "minirocket", "cnn", "lstm"} <= models
    assert any(m.endswith("_mlp") for m in models)


# ---------------------------------------------------------------------------
# run_ablation restart + select_best guard
# ---------------------------------------------------------------------------
def test_run_ablation_restart_never_recomputes_a_completed_cell(tmp_path):
    """Restart safety for the ablation: a rerun may ADD cells (phase 2 re-picks the best
    cell from the now-fuller CSV) but must never re-emit one it already has, so no
    (context, features, pooling, seed) row is ever duplicated."""
    cfg = _cfg(tmp_path)
    write_synthetic_cmapss(Path(cfg.data_dir), n_train_units=5, n_test_units=4)
    kw = dict(device="cpu", contexts=[12], feature_sets=["emb"], pooling_variants=["mean"],
              seeds=[0], embedder_factory=lambda c: MockEmbedder(feature_dim=12))
    out = S.run_ablation(cfg, **kw)
    first = [tuple(r.values()) for r in csv.DictReader(open(out))]
    S.run_ablation(cfg, **kw)                     # rerun: completed cells are skipped
    rows = list(csv.DictReader(open(out)))
    cells = [(r["tsfm_context_length"], r["head_features"], r["pooling"], r["seed"])
             for r in rows]
    assert len(cells) == len(set(cells))          # nothing recomputed
    assert len(rows) >= len(first)


def test_select_best_ablation_cell_requires_rows(tmp_path):
    with pytest.raises(ValueError, match="no ablation rows"):
        S.select_best_ablation_cell(tmp_path / "ablation.csv")


# ---------------------------------------------------------------------------
# run_baseline_window_comparison
# ---------------------------------------------------------------------------
def test_run_baseline_window_comparison_is_restartable(tmp_path):
    """Each baseline is rerun at every requested window straight from the raw series
    (no embedding needed), keyed on (model, dataset, window, seed)."""
    cfg = _cfg(tmp_path)
    write_synthetic_cmapss(Path(cfg.data_dir), n_train_units=5, n_test_units=4)
    out = S.run_baseline_window_comparison(cfg, windows=[12, 6],
                                           baseline_names=["predict_mean"], seeds=[0])
    rows = list(csv.DictReader(open(out)))
    assert {r["baseline_window"] for r in rows} == {"12", "6"}
    assert all(np.isfinite(float(r["rmse_clipped"])) for r in rows)
    S.run_baseline_window_comparison(cfg, windows=[12, 6],
                                     baseline_names=["predict_mean"], seeds=[0])
    assert len(list(csv.DictReader(open(out)))) == len(rows)      # restart adds nothing


# ---------------------------------------------------------------------------
# run_fairness_baselines: unit counts above the fleet are skipped
# ---------------------------------------------------------------------------
def test_fairness_baselines_skip_unit_counts_above_the_fleet(tmp_path):
    cfg = _cfg(tmp_path)
    write_synthetic_cmapss(Path(cfg.data_dir), n_train_units=5, n_test_units=4)
    out = S.run_fairness_baselines(cfg, n_units_list=[2, 999], seeds=[0])
    counts = {int(r["n_units"]) for r in csv.DictReader(open(out))}
    assert counts == {2}                          # 999 > 5 available units -> skipped


# ---------------------------------------------------------------------------
# transfer: the multi-condition warning + oversized shot counts
# ---------------------------------------------------------------------------
def test_transfer_warns_on_unnormalized_multi_condition_source(tmp_path, capsys):
    """FD002 has several operating conditions; running it with condition-wise
    normalization explicitly OFF must print the exploratory-only warning, and a shot
    count larger than the target fleet is skipped rather than crashing."""
    cfg = _cfg(tmp_path, dataset="FD002", condition_norm=False)
    write_synthetic_cmapss(Path(cfg.data_dir), dataset="FD002", n_train_units=5,
                           n_test_units=4, n_conditions=3, seed=1)
    write_synthetic_cmapss(Path(cfg.data_dir), dataset="FD003", n_train_units=5,
                           n_test_units=4, seed=2)
    embedders: dict[str, MockEmbedder] = {}
    factory = lambda c: embedders.setdefault(c.dataset, MockEmbedder(feature_dim=12))
    out = X.run_transfer_eval(cfg, source_dataset="FD002", target_dataset="FD003",
                              shots=[2, 999], seeds=[0], losses=["mse"],
                              baseline_names=[], embedder_factory=factory)
    assert "multiple operating conditions" in capsys.readouterr().out
    ks = {int(r["n_target_units"]) for r in csv.DictReader(open(out))}
    assert ks == {0, 2}                           # 999 > 5 target units -> skipped
