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

    emb = MockEmbedder(feature_dim=16).embed_windows(win)
    assert emb.shape == (len(win), 16)

    # tiny train/val split by row is fine here (this is the head, not the sweep)
    n = len(emb); k = int(n * 0.8)
    model, hist = T.train_head(emb[:k], lab[:k], emb[k:], lab[k:], "mse", cfg, seed=0)
    assert len(hist["train_loss"]) > 0          # per-step logging populated
    assert np.isfinite(hist["best_val_rmse"])
    pred = T.predict_head(model, emb[k:], "mse", cfg)
    assert pred.min() >= 0 and pred.max() <= cfg.max_rul
    m = evaluate_predictions(lab[k:], pred)
    assert np.isfinite(m["rmse"]) and np.isfinite(m["nasa_score"])


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
                "test_emb", "test_windows", "test_labels", "test_units"]:
        assert key in cache
    assert cache["train_emb"].shape[0] == cache["train_windows"].shape[0]
    assert cache["train_emb"].shape[1] == 32
    assert cache["test_emb"].shape[0] == cache["test_windows"].shape[0]

    # Stage B: sweep consumes the cache only; embedder must not be touched again.
    from src import sweep as S
    results_csv = S.run_sweep(
        cfg, baseline_names=["predict_mean", "cnn", "lstm"], device="cpu",
    )
    assert embedder.n_calls == 2  # NO re-embedding during the sweep (Task 3)

    rows = list(csv.DictReader(open(results_csv)))
    assert len(rows) > 0
    models = {r["model"] for r in rows}
    losses = {r["loss"] for r in rows}
    assert any("_mlp" in m for m in models)         # TSFM head present
    assert {"predict_mean", "cnn", "lstm"} <= models
    assert {"mse", "corn"} <= losses                # both loss arms ran
    for r in rows:
        assert int(r["n_units"]) in (2, 4)
        assert np.isfinite(float(r["rmse"]))
        assert np.isfinite(float(r["nasa_score"]))
        assert float(r["mae"]) >= 0

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
