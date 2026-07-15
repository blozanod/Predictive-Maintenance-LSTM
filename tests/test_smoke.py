"""CPU-only end-to-end smoke test (Task 2.7).

Synthetic multivariate C-MAPSS -> windowing -> (mock) embedding -> MLP head trained
a few steps -> metrics -> full sweep with baselines. Runs without a GPU and without
downloading C-MAPSS, so the pipeline can be validated before spending Colab hours.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from src.config import Config
from src import embeddings as E
from src import train as T
from src import data as D
from src.evaluate import evaluate_predictions
from tests.synthetic import write_synthetic_cmapss, MockEmbedder


def _smoke_config(tmp_path: Path) -> Config:
    return Config(
        dataset="FD001",
        data_dir=str(tmp_path / "CMAPSSData"),
        cache_dir=str(tmp_path / "cache"),
        results_dir=str(tmp_path / "results"),
        window_size=12,  # >=9 so the sktime MiniRocket baseline is exercisable
        sensor_columns=["s_2", "s_3", "s_4", "s_7", "s_9"],
        max_rul=40,
        num_bins=8,
        data_unit_counts=[2, 4],
        sweep_seeds=[0, 1],
        head_hidden_dim=16,
        head_batch_size=32,
        head_max_epochs=4,
        head_early_stopping_patience=2,
        baseline_max_epochs=3,
        baseline_early_stopping_patience=2,
        losses=["mse", "corn"],
    )


def test_direct_embed_train_metrics(tmp_path):
    """Windowing -> mock embed -> train head a few steps -> metrics, no files."""
    cfg = _smoke_config(tmp_path)
    write_synthetic_cmapss(Path(cfg.data_dir), n_train_units=6, n_test_units=4)
    df_train, df_test, rul = D.load_cmapss(cfg)
    df_train = D.add_train_rul(df_train, cfg)

    win, lab, units = D.make_windows(df_train, cfg.sensor_columns, cfg.window_size)
    assert win.shape[1:] == (cfg.window_size, len(cfg.sensor_columns))

    emb, loc_scale = MockEmbedder(feature_dim=16).embed_windows(win)
    assert emb.shape == (len(win), 16)
    assert loc_scale.shape == (len(win), len(cfg.sensor_columns), 2)

    # tiny train/val split by row is fine here (this is the head, not the sweep)
    n = len(emb); k = int(n * 0.8)
    model, hist = T.train_head(emb[:k], lab[:k], emb[k:], lab[k:], "mse", cfg, seed=0)
    assert len(hist["train_loss"]) > 0          # per-step logging populated
    assert np.isfinite(hist["best_val_rmse"])
    pred = T.predict_head(model, emb[k:], "mse", cfg)
    assert pred.min() >= 0 and pred.max() <= cfg.max_rul
    m = evaluate_predictions(lab[k:], pred, cfg.max_rul)
    assert np.isfinite(m["rmse_clipped"]) and np.isfinite(m["nasa_unclipped"])


def test_cache_is_idempotent_and_sweep_never_reembeds(tmp_path):
    cfg = _smoke_config(tmp_path)
    write_synthetic_cmapss(Path(cfg.data_dir), n_train_units=8, n_test_units=6)
    embedder = MockEmbedder(feature_dim=32)

    # Stage A: build once (train + test => 2 embed calls)
    path = E.build_embedding_cache(cfg, embedder=embedder)
    assert path.exists()
    assert path.with_suffix(".json").exists()
    assert embedder.n_calls == 2

    # Idempotent: same key => no recompute
    E.build_embedding_cache(cfg, embedder=embedder)
    assert embedder.n_calls == 2

    cache = E.load_embedding_cache(cfg)
    for key in ["train_emb", "train_windows", "train_labels", "train_units",
                "train_locscale", "test_emb", "test_windows", "test_labels",
                "test_units", "test_locscale"]:
        assert key in cache
    assert cache["train_emb"].shape[0] == cache["train_windows"].shape[0]
    assert cache["train_emb"].shape[1] == 32
    assert cache["train_locscale"].shape == (cache["train_emb"].shape[0],
                                             len(cfg.sensor_columns), 2)
    assert cache["test_emb"].shape[0] == cache["test_windows"].shape[0]

    # Stage B: sweep consumes the cache only; embedder must not be touched again.
    from src import sweep as S
    results_csv = S.run_sweep(
        cfg, baseline_names=["predict_mean", "cnn", "lstm"], device="cpu",
    )
    assert embedder.n_calls == 2  # NO re-embedding during the sweep (Task 3)
    assert results_csv.name == "results_v2.csv"  # v2 schema file (Task 1.4)

    rows = list(csv.DictReader(open(results_csv)))
    assert len(rows) > 0
    models = {r["model"] for r in rows}
    losses = {r["loss"] for r in rows}
    assert any("_mlp" in m for m in models)         # TSFM head present
    assert {"predict_mean", "cnn", "lstm"} <= models
    assert {"mse", "corn"} <= losses                # both loss arms ran
    # both-protocol metric columns present and finite (Task 1.4)
    for col in ("rmse_clipped", "mae_clipped", "nasa_clipped",
                "rmse_unclipped", "mae_unclipped", "nasa_unclipped"):
        assert col in rows[0]
    for r in rows:
        # {2,4} requested; 8 = the auto-appended full-fleet cell (8 train units, §29)
        assert int(r["n_units"]) in (2, 4, 8)
        assert int(r["schema_version"]) == 2
        assert np.isfinite(float(r["rmse_clipped"]))
        assert np.isfinite(float(r["rmse_unclipped"]))
        assert np.isfinite(float(r["nasa_clipped"]))
        assert float(r["mae_clipped"]) >= 0

    # sampled unit IDs saved per cell (Task 2.3)
    run_dir = Path(cfg.results_dir) / "runs"
    assert list(run_dir.glob("units_n*_seed*.json"))
    assert list((run_dir / "learning_curves").glob("*.csv"))
    assert (run_dir / "run_metadata.json").exists()

    # Restart-safety: rerunning skips completed cells => row count unchanged.
    n_before = len(rows)
    S.run_sweep(cfg, baseline_names=["predict_mean", "cnn", "lstm"], device="cpu")
    n_after = len(list(csv.DictReader(open(results_csv))))
    assert n_after == n_before


def test_resolve_unit_counts_appends_full_fleet(tmp_path):
    """§29: the grid always includes the full-fleet cell; FD-size datasets unchanged."""
    from src.sweep import resolve_unit_counts
    # small dataset (9 units): counts above the fleet size collapse to the full cell
    assert resolve_unit_counts([2, 5, 10, 25, 50, 100], 9) == [2, 5, 9]
    assert resolve_unit_counts([2, 5], 6) == [2, 5, 6]        # DS02 dev fleet
    # full-size dataset: exactly the requested grid (no new cells -> keys unchanged)
    assert resolve_unit_counts([2, 5, 10, 25, 50, 100], 100) == [2, 5, 10, 25, 50, 100]
    # the max count already equals the fleet -> idempotent
    assert resolve_unit_counts([2, 4], 4) == [2, 4]


def test_sweep_gets_full_fleet_cell_on_small_dataset(tmp_path):
    """A 5-train-unit dataset with data_unit_counts=[2,50] sweeps exactly {2,5}: the
    requested 2, plus the auto-appended full-fleet cell (5), and NOT 50."""
    from src import sweep as S
    cfg = _smoke_config(tmp_path).replace(data_unit_counts=[2, 50], sweep_seeds=[0],
                                          losses=["mse"])
    write_synthetic_cmapss(Path(cfg.data_dir), n_train_units=5, n_test_units=4)
    E.build_embedding_cache(cfg, embedder=MockEmbedder(feature_dim=16))
    results_csv = S.run_sweep(cfg, baseline_names=["predict_mean"], device="cpu")
    rows = list(csv.DictReader(open(results_csv)))
    assert {int(r["n_units"]) for r in rows} == {2, 5}


def test_ablation_runs_and_selects_cell(tmp_path):
    """run_ablation orchestrates Stage A (per context/pooling) + head training over
    the grid on CPU with a mock embedder, and select_best returns a valid cell."""
    from src import sweep as S
    cfg = _smoke_config(tmp_path)
    write_synthetic_cmapss(Path(cfg.data_dir), n_train_units=8, n_test_units=6)

    abl_csv = S.run_ablation(
        cfg, device="cpu",
        contexts=[12], feature_sets=["emb", "emb+locscale"],
        pooling_variants=["mean"], seeds=[0],
        embedder_factory=lambda c: MockEmbedder(feature_dim=24),
    )
    rows = list(csv.DictReader(open(abl_csv)))
    assert len(rows) >= 3  # emb, emb+locscale, +raw arm, + mean-pooling variant
    got = {(r["tsfm_context_length"], r["head_features"], r["pooling"]) for r in rows}
    assert ("12", "emb", "forecast_token") in got
    assert ("12", "emb+locscale", "forecast_token") in got
    assert ("12", "emb+locscale+raw", "forecast_token") in got  # raw-fusion arm ran
    assert any(p == "mean" for (_, _, p) in got)                # pooling variant ran

    best = S.select_best_ablation_cell(abl_csv)
    assert best["head_features"] in ("emb", "emb+locscale")
    assert best["tsfm_context_length"] == 12
    assert np.isfinite(best["mean_rmse_clipped"])


@pytest.mark.parametrize("name", ["gbm", "minirocket"])
def test_optional_library_baselines(tmp_path, name):
    """GBM/MiniRocket reuse lightgbm/sktime; skipped when not installed."""
    pytest.importorskip("lightgbm" if name == "gbm" else "sktime")
    cfg = _smoke_config(tmp_path)
    write_synthetic_cmapss(Path(cfg.data_dir), n_train_units=6, n_test_units=4)
    df_train, df_test, rul = D.load_cmapss(cfg)
    df_train = D.add_train_rul(df_train, cfg)
    df_test = D.add_test_rul(df_test, rul, cfg)
    tr_w, tr_y, _ = D.make_windows(df_train, cfg.sensor_columns, cfg.window_size)
    te_w, te_y, _ = D.make_test_last_windows(df_test, cfg.sensor_columns, cfg.window_size)

    from src import baselines as B
    bl = B.make_baseline(name, cfg, seed=0).fit(tr_w, tr_y)
    pred = bl.predict(te_w)
    assert pred.shape == (len(te_w),)
    assert pred.min() >= 0 and pred.max() <= cfg.max_rul
