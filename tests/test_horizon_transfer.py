"""CPU smoke tests for the horizon-stratified eval (src/horizon.py) and the
cold-start transfer eval (src/transfer.py), on synthetic C-MAPSS + MockEmbedder."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.config import Config
from src import horizon as H
from src import transfer as X
from src.embeddings import build_embedding_cache
from tests.synthetic import write_synthetic_cmapss, MockEmbedder


def _cfg(tmp_path: Path, dataset: str = "FD001") -> Config:
    return Config(
        dataset=dataset,
        data_dir=str(tmp_path / "CMAPSSData"),
        cache_dir=str(tmp_path / "cache"),
        results_dir=str(tmp_path / "results"),
        window_size=12,
        sensor_columns=["s_2", "s_3", "s_4", "s_7", "s_9"],
        max_rul=40,
        num_bins=8,
        sweep_seeds=[0, 1],
        head_hidden_dim=16,
        head_batch_size=32,
        head_max_epochs=3,
        head_early_stopping_patience=2,
        baseline_max_epochs=2,
        baseline_early_stopping_patience=1,
        losses=["mse"],
    )


def test_horizon_bin_rows_stratify_and_total():
    y_true = np.array([5.0, 20.0, 45.0, 90.0])   # unclipped; 90 > max_rul=40
    y_pred = np.array([10.0, 15.0, 40.0, 40.0])
    rows = H.horizon_bin_rows(y_true, y_pred, max_rul=40,
                              bin_edges=(0, 25, 50, float("inf")))
    by_lo = {r["bin_lo"]: r for r in rows}
    assert by_lo[0]["n_bin"] == 2 and by_lo[25]["n_bin"] == 1
    assert by_lo[50]["bin_hi"] == "inf" and by_lo[50]["n_bin"] == 1
    # saturation bin: truth clips to 40, prediction at the cap => zero error
    assert by_lo[50]["mae_clipped"] == 0.0
    assert by_lo["all"]["n_bin"] == 4
    # bias sign: first bin predictions average (10-5) + (15-20) = 0
    assert abs(by_lo[0]["bias"]) < 1e-9


def test_horizon_cache_and_eval_end_to_end(tmp_path):
    cfg = _cfg(tmp_path)
    write_synthetic_cmapss(Path(cfg.data_dir), n_train_units=8, n_test_units=6,
                           min_cycles=20, max_cycles=45)
    embedder = MockEmbedder(feature_dim=16)
    build_embedding_cache(cfg, embedder=embedder)

    hpath = H.build_horizon_cache(cfg, embedder=embedder)
    assert hpath.exists() and hpath.with_suffix(".json").exists()
    calls_after_build = embedder.n_calls
    H.build_horizon_cache(cfg, embedder=embedder)          # idempotent
    assert embedder.n_calls == calls_after_build

    hcache = H.load_horizon_cache(cfg)
    # every test cycle >= window_size is a row (short units excluded)
    assert hcache["test_emb"].shape[0] == hcache["test_labels"].shape[0] > 6

    out = H.run_horizon_eval(cfg, n_units_list=[4], seeds=[0],
                             baseline_names=["gbm"],
                             bin_edges=(0, 20, float("inf")))
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    models = {r["model"] for r in rows}
    assert "chronos-2_mlp" in models and "gbm" in models
    assert {r["bin_lo"] for r in rows} >= {"0", "all"}
    preds = Path(cfg.results_dir) / "horizon_predictions.csv"
    assert preds.exists()
    # restartability: rerun adds nothing
    n_rows = len(rows)
    H.run_horizon_eval(cfg, n_units_list=[4], seeds=[0], baseline_names=["gbm"],
                       bin_edges=(0, 20, float("inf")))
    with open(out, newline="") as f:
        assert len(list(csv.DictReader(f))) == n_rows


def test_transfer_zero_and_few_shot(tmp_path):
    cfg = _cfg(tmp_path)
    write_synthetic_cmapss(Path(cfg.data_dir), dataset="FD001",
                           n_train_units=8, n_test_units=5, seed=0)
    write_synthetic_cmapss(Path(cfg.data_dir), dataset="FD003",
                           n_train_units=6, n_test_units=5, seed=1)
    embedders: dict[str, MockEmbedder] = {}

    def factory(c: Config) -> MockEmbedder:
        return embedders.setdefault(c.dataset, MockEmbedder(feature_dim=16))

    out = X.run_transfer_eval(cfg, source_dataset="FD001", target_dataset="FD003",
                              shots=[2, 4], seeds=[0], losses=["mse"],
                              baseline_names=["gbm"], embedder_factory=factory)
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    modes = {(r["model"], r["mode"], r["n_target_units"]) for r in rows}
    assert ("chronos-2_mlp", "zero_shot", "0") in modes
    assert ("chronos-2_mlp", "target_only", "2") in modes
    assert ("chronos-2_mlp", "source+target", "4") in modes
    assert ("gbm", "zero_shot", "0") in modes
    # every row evaluates the TARGET test set (5 units) under both protocols
    assert all(r["n"] == "5" for r in rows)
    assert all(float(r["rmse_clipped"]) > 0 for r in rows)
    # restartability
    X.run_transfer_eval(cfg, source_dataset="FD001", target_dataset="FD003",
                        shots=[2, 4], seeds=[0], losses=["mse"],
                        baseline_names=["gbm"], embedder_factory=factory)
    with open(out, newline="") as f:
        assert len(list(csv.DictReader(f))) == len(rows)


def test_transfer_rejects_single_shot(tmp_path):
    cfg = _cfg(tmp_path)
    try:
        X.run_transfer_eval(cfg, shots=[1])
        assert False, "expected ValueError for shots < 2"
    except ValueError:
        pass
