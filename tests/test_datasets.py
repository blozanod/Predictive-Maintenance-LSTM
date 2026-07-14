"""CPU tests for the multi-dataset layer (CHANGES.md §21-22): condition-wise
normalization, the unified load_prepared path, multi-dataset restart keys, and
the XJTU-SY bearing loader."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src import data as D
from src.xjtu import XJTU_FEATURE_COLUMNS, load_xjtu
from tests.synthetic import write_synthetic_cmapss, write_synthetic_xjtu, MockEmbedder


def _cfg(tmp_path: Path, **over) -> Config:
    base = dict(
        dataset="FD002",
        data_dir=str(tmp_path / "CMAPSSData"),
        cache_dir=str(tmp_path / "cache"),
        results_dir=str(tmp_path / "results"),
        window_size=12,
        sensor_columns=["s_2", "s_3", "s_4", "s_7", "s_9"],
        max_rul=40,
        num_bins=8,
        data_unit_counts=[2, 4],
        sweep_seeds=[0],
        head_hidden_dim=16,
        head_batch_size=32,
        head_max_epochs=3,
        head_early_stopping_patience=2,
        baseline_max_epochs=2,
        baseline_early_stopping_patience=1,
        losses=["mse"],
    )
    base.update(over)
    return Config(**base)


# ---------------------------------------------------------------------------
# Condition-wise normalization
# ---------------------------------------------------------------------------
def test_condition_keys_and_norm(tmp_path):
    cfg = _cfg(tmp_path)
    write_synthetic_cmapss(Path(cfg.data_dir), dataset="FD002", n_train_units=8,
                           n_test_units=4, n_conditions=3, seed=1)
    df_train, df_test, rul = D.load_cmapss(cfg)
    assert len(np.unique(D.identify_conditions(df_train))) == 3
    # raw data: regime switching dominates -- per-sensor std is huge
    raw_std = df_train[cfg.sensor_columns].std().mean()
    assert raw_std > 20

    tr_n, te_n = D.condition_normalize(D.add_train_rul(df_train, cfg),
                                       D.add_test_rul(df_test, rul, cfg), cfg)
    keys = D.condition_keys(tr_n)
    for key in np.unique(keys, axis=0):
        block = tr_n.loc[np.all(keys == key, axis=1), cfg.sensor_columns]
        assert np.allclose(block.mean().to_numpy(), 0, atol=1e-9)
        assert np.allclose(block.std(ddof=0).to_numpy(), 1, atol=1e-6)
    # test normalized with TRAIN stats: roughly standardized (raw scale was ~500
    # with ±60 condition offsets) but NOT exactly zero-mean -- proof the stats
    # came from train, not from the test frame itself (no leakage).
    te_means = te_n[cfg.sensor_columns].mean().to_numpy()
    assert np.all(np.abs(te_means) < 5.0)
    assert not np.allclose(te_means, 0, atol=1e-9)
    # labels untouched
    assert (tr_n["clipped_rul"] <= cfg.max_rul).all()


def test_condition_norm_cross_frame_key_alignment(tmp_path):
    """Scaler keys are setting VALUES: a test frame missing one train condition
    must still normalize each row with ITS OWN condition's train stats."""
    cfg = _cfg(tmp_path)
    write_synthetic_cmapss(Path(cfg.data_dir), dataset="FD002", n_train_units=8,
                           n_test_units=4, n_conditions=3, seed=2)
    df_train, df_test, rul = D.load_cmapss(cfg)
    df_train, df_test = D.add_train_rul(df_train, cfg), D.add_test_rul(df_test, rul, cfg)
    # drop condition-0 rows from test => test's per-frame ranks would misalign
    te_keys = D.condition_keys(df_test)
    keep_key = np.unique(te_keys, axis=0)[0]
    df_test_sub = df_test.loc[~np.all(te_keys == keep_key, axis=1)].reset_index(drop=True)

    _, te_n = D.condition_normalize(df_train, df_test_sub, cfg)
    _, te_full = D.condition_normalize(df_train, df_test, cfg)
    # rows present in both must be normalized identically (value-keyed, not rank-keyed)
    sub_ids = df_test_sub[["unit_number", "time_cycles"]].apply(tuple, axis=1)
    full = te_full.set_index(["unit_number", "time_cycles"])
    sub = te_n.set_index(["unit_number", "time_cycles"])
    common = sub.index.intersection(full.index)
    assert len(common) == len(sub)
    assert np.allclose(sub.loc[common, cfg.sensor_columns].to_numpy(),
                       full.loc[common, cfg.sensor_columns].to_numpy())


def test_effective_condition_norm_auto():
    assert not Config(dataset="FD001").effective_condition_norm()
    assert not Config(dataset="FD003").effective_condition_norm()
    assert Config(dataset="FD002").effective_condition_norm()
    assert Config(dataset="FD004").effective_condition_norm()
    assert Config(dataset="XJTU-SY").effective_condition_norm()
    assert Config(dataset="FD002", condition_norm=False).effective_condition_norm() is False
    # toggling it re-keys the caches
    assert (Config(dataset="FD002").embedding_cache_key()
            != Config(dataset="FD002", condition_norm=False).embedding_cache_key())


def test_load_prepared_applies_norm_only_when_resolved(tmp_path):
    cfg1 = _cfg(tmp_path, dataset="FD001")
    write_synthetic_cmapss(Path(cfg1.data_dir), dataset="FD001", n_train_units=6,
                           n_test_units=4, seed=3)
    tr, _ = D.load_prepared(cfg1)   # FD001: OFF -> raw sensor scale preserved
    assert tr[cfg1.sensor_columns].abs().mean().mean() > 100
    cfg2 = _cfg(tmp_path, dataset="FD002")
    write_synthetic_cmapss(Path(cfg2.data_dir), dataset="FD002", n_train_units=6,
                           n_test_units=4, n_conditions=3, seed=3)
    tr2, _ = D.load_prepared(cfg2)  # FD002: ON -> per-condition standardized
    assert tr2[cfg2.sensor_columns].abs().mean().mean() < 2


# ---------------------------------------------------------------------------
# Multi-dataset restart keys
# ---------------------------------------------------------------------------
def test_sweep_cells_keyed_by_dataset(tmp_path):
    """Running FD003 after FD001 into the SAME results_v2.csv must add rows (the
    pre-§21 keys marked every FD003 cell 'done')."""
    from src import sweep as S
    from src.embeddings import build_embedding_cache

    n_rows = {}
    for ds in ("FD001", "FD003"):
        cfg = _cfg(tmp_path, dataset=ds)
        write_synthetic_cmapss(Path(cfg.data_dir), dataset=ds, n_train_units=6,
                               n_test_units=4, seed=4)
        build_embedding_cache(cfg, embedder=MockEmbedder(feature_dim=16))
        out = S.run_sweep(cfg, baseline_names=["predict_mean"], device="cpu")
        with open(out, newline="") as f:
            rows = list(csv.DictReader(f))
        n_rows[ds] = len(rows)
    assert n_rows["FD003"] == 2 * n_rows["FD001"]
    assert {r["dataset"] for r in rows} == {"FD001", "FD003"}


# ---------------------------------------------------------------------------
# XJTU-SY loader
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def xjtu_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("xjtu")
    write_synthetic_xjtu(root, bearings_per_condition=3, min_snapshots=18,
                         max_snapshots=30, samples_per_snapshot=128)
    return root


def _xjtu_cfg(root, tmp_path: Path) -> Config:
    return Config(
        dataset="XJTU-SY", data_dir=str(root),
        cache_dir=str(tmp_path / "cache"), results_dir=str(tmp_path / "results"),
        window_size=6, sensor_columns=list(XJTU_FEATURE_COLUMNS), max_rul=15,
        xjtu_test_bearings=["Bearing1_3", "Bearing2_3", "Bearing3_3"],
        xjtu_test_truncation=0.6,
    )


def test_xjtu_loader_schema_and_split(xjtu_root, tmp_path):
    cfg = _xjtu_cfg(xjtu_root, tmp_path)
    df_train, df_test, rul = load_xjtu(cfg)
    # canonical frame: 9 bearings total, 3 held out
    assert df_train["unit_number"].nunique() == 6
    assert df_test["unit_number"].nunique() == 3
    assert set(XJTU_FEATURE_COLUMNS) <= set(df_train.columns)
    assert not set(df_train["unit_number"]) & set(df_test["unit_number"])
    # 3 operating conditions encoded in setting_1 -> condition norm groups them
    assert df_train["setting_1"].nunique() == 3
    # test truncation: every test unit shorter than its full life, RUL > 0 matches
    for u, r in rul.items():
        n_kept = int(df_test.loc[df_test.unit_number == u, "time_cycles"].max())
        assert r > 0 and n_kept >= cfg.window_size
    # features finite, and the degradation signal exists (late rms > early rms)
    tr = df_train.sort_values(["unit_number", "time_cycles"])
    assert np.isfinite(tr[XJTU_FEATURE_COLUMNS].to_numpy()).all()
    one = tr[tr.unit_number == tr.unit_number.iloc[0]]
    assert one["h_rms"].iloc[-3:].mean() > one["h_rms"].iloc[:3].mean()


def test_xjtu_end_to_end_windows_and_cache(xjtu_root, tmp_path):
    cfg = _xjtu_cfg(xjtu_root, tmp_path)
    df_train, df_test = D.load_prepared(cfg)   # includes condition norm (auto ON)
    w, y, u = D.make_windows(df_train, cfg.sensor_columns, cfg.window_size,
                             target_col="clipped_rul")
    assert w.shape[1:] == (cfg.window_size, len(XJTU_FEATURE_COLUMNS))
    assert y.max() <= cfg.max_rul and len(np.unique(u)) == 6

    from src.embeddings import build_embedding_cache, load_embedding_cache
    build_embedding_cache(cfg, embedder=MockEmbedder(feature_dim=16))
    cache = load_embedding_cache(cfg)
    assert cache["train_emb"].shape[0] == cache["train_windows"].shape[0]
    assert cache["test_emb"].shape[0] == 3   # one last-context per test bearing


def test_xjtu_rejects_bad_split(xjtu_root, tmp_path):
    cfg = _xjtu_cfg(xjtu_root, tmp_path).replace(
        xjtu_test_bearings=["Bearing9_9"])
    with pytest.raises(ValueError, match="not on disk"):
        load_xjtu(cfg)
