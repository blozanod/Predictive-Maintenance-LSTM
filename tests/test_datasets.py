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
from src.datasets.xjtu import XJTU_FEATURE_COLUMNS, load_xjtu
from src.datasets.ncmapss import load_ncmapss
from tests.synthetic import (write_synthetic_cmapss, write_synthetic_xjtu,
                             write_synthetic_ncmapss, MockEmbedder)


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


# --- §25: condition-3 folder/force fix + unmatched-folder guard ---
def test_xjtu_condition3_folder_and_force():
    """Regression: condition 3 is "40Hz10kN" at 10 kN, NOT the pre-§25
    "40Hz12kN"/12 kN typo that silently hid the whole condition."""
    from src.datasets.xjtu import XJTU_CONDITIONS
    assert XJTU_CONDITIONS["40Hz10kN"] == (2, 40.0, 10.0)
    assert "40Hz12kN" not in XJTU_CONDITIONS
    # the loader must actually surface all 3 conditions now
    assert {c[0] for c in XJTU_CONDITIONS.values()} == {0, 1, 2}


def test_xjtu_unmatched_condition_folder_raises(xjtu_root, tmp_path):
    """A condition-looking folder that isn't recognized must fail loudly, naming
    both the offending folder and the expected set."""
    stray = xjtu_root / "45Hz9kN" / "Bearing4_1"
    stray.mkdir(parents=True, exist_ok=True)
    (stray / "1.csv").write_text(
        "Horizontal_vibration_signals,Vertical_vibration_signals\n0.1,0.1\n")
    try:
        cfg = _xjtu_cfg(xjtu_root, tmp_path)
        with pytest.raises(ValueError, match="45Hz9kN"):
            load_xjtu(cfg)
    finally:
        import shutil
        shutil.rmtree(xjtu_root / "45Hz9kN")


# --- §26: tolerant data-dir resolution (alternate subdir name + depth-1 nesting) ---
def _xjtu_cfg_rooted(data_root: Path, tmp_path: Path) -> Config:
    """XJTU config that resolves via data_root/subdir candidates (data_dir=None)."""
    return Config(
        dataset="XJTU-SY", data_root=str(data_root), data_dir=None,
        cache_dir=str(tmp_path / "cache"), results_dir=str(tmp_path / "results"),
        window_size=6, sensor_columns=list(XJTU_FEATURE_COLUMNS), max_rul=15,
        xjtu_test_bearings=["Bearing1_3", "Bearing2_3", "Bearing3_3"],
        xjtu_test_truncation=0.6,
    )


def test_xjtu_resolves_alternate_subdir_name(tmp_path):
    """The zip's own folder name ``XJTU-SY_Bearing_Datasets`` loads without renaming."""
    data_root = tmp_path / "Data"
    write_synthetic_xjtu(data_root / "XJTU-SY_Bearing_Datasets",
                         bearings_per_condition=3, min_snapshots=18, max_snapshots=30,
                         samples_per_snapshot=128)
    cfg = _xjtu_cfg_rooted(data_root, tmp_path)
    df_train, df_test, rul = load_xjtu(cfg)
    assert df_train["unit_number"].nunique() == 6 and df_test["unit_number"].nunique() == 3


def test_xjtu_resolves_one_level_nesting(tmp_path):
    """Zip-in-a-folder (condition dirs one level below the resolved root) loads."""
    data_root = tmp_path / "Data"
    write_synthetic_xjtu(data_root / "XJTU-SY" / "XJTU-SY_Bearing_Datasets",
                         bearings_per_condition=3, min_snapshots=18, max_snapshots=30,
                         samples_per_snapshot=128)
    cfg = _xjtu_cfg_rooted(data_root, tmp_path)
    from src import datasets as DS
    assert DS.is_available(cfg)
    df_train, _, _ = load_xjtu(cfg)
    assert df_train["unit_number"].nunique() == 6


# ---------------------------------------------------------------------------
# §27: N-CMAPSS loader (cycle-aggregated frames, truncation protocol)
# ---------------------------------------------------------------------------
def _ncmapss_cfg(root: Path, tmp_path: Path, dataset="DS02", **over) -> Config:
    base = dict(
        dataset=dataset, data_root=str(root), data_dir=None,
        cache_dir=str(tmp_path / "cache"), results_dir=str(tmp_path / "results"),
        window_size=4, max_rul=125, ncmapss_test_truncation=0.6,
        num_bins=5, data_unit_counts=[2], sweep_seeds=[0],
        head_hidden_dim=8, head_max_epochs=2, head_early_stopping_patience=1,
        baseline_max_epochs=2, baseline_early_stopping_patience=1, losses=["mse"],
    )
    base.update(over)
    return Config(**base)


def test_ncmapss_schema_and_split(tmp_path):
    from src.config import INDEX_COLUMNS, SETTING_COLUMNS, NCMAPSS_FEATURE_COLUMNS
    write_synthetic_ncmapss(tmp_path / "Data" / "N-CMAPSS", dataset="DS02",
                            n_dev_units=3, n_test_units=2, seed=1)
    cfg = _ncmapss_cfg(tmp_path / "Data", tmp_path)
    df_train, df_test, rul = load_ncmapss(cfg)
    # canonical frame: one row per (unit, cycle); exact column set/order
    assert list(df_train.columns) == (list(INDEX_COLUMNS) + list(SETTING_COLUMNS)
                                      + list(NCMAPSS_FEATURE_COLUMNS))
    assert len(NCMAPSS_FEATURE_COLUMNS) == 37
    # dev = train (full length), test disjoint
    assert df_train["unit_number"].nunique() == 3
    assert df_test["unit_number"].nunique() == 2
    assert not set(df_train.unit_number) & set(df_test.unit_number)
    # time_cycles consecutive from 1; setting_1 = Fc constant per unit ∈ {1,2,3}
    for _, u in df_train.groupby("unit_number"):
        cyc = u.sort_values("time_cycles").time_cycles.to_numpy()
        assert cyc[0] == 1 and np.array_equal(cyc, np.arange(1, len(cyc) + 1))
        assert u.setting_1.nunique() == 1 and u.setting_1.iloc[0] in (1, 2, 3)
    # truncation: each test unit kept to max(window, floor(0.6 n)); rul = n - keep
    assert (rul > 0).all() and np.isfinite(df_test[NCMAPSS_FEATURE_COLUMNS].to_numpy()).all()


def test_ncmapss_aggregate_values(tmp_path):
    """alt_mean / Wf_std / cycle_len_s match a hand computation on the raw arrays
    (pandas ddof=1 std)."""
    import h5py
    ncdir = tmp_path / "Data" / "N-CMAPSS"
    path = write_synthetic_ncmapss(ncdir, dataset="DS02", n_dev_units=2,
                                   n_test_units=2, seed=3)
    with h5py.File(path) as h:
        W, Xs, A = np.asarray(h["W_dev"]), np.asarray(h["X_s_dev"]), np.asarray(h["A_dev"])
        wv = [str(x).strip() for x in np.array(h["W_var"]).astype("U20").ravel()]
        xv = [str(x).strip() for x in np.array(h["X_s_var"]).astype("U20").ravel()]
    cfg = _ncmapss_cfg(tmp_path / "Data", tmp_path)
    df_train, _, _ = load_ncmapss(cfg)
    m = (A[:, 0] == 1) & (A[:, 1] == 1)
    row = df_train[(df_train.unit_number == 1) & (df_train.time_cycles == 1)].iloc[0]
    assert np.isclose(row.alt_mean, W[m, wv.index("alt")].mean(), atol=1e-4)
    assert np.isclose(row.Wf_std, Xs[m, xv.index("Wf")].std(ddof=1), atol=1e-4)
    assert row.cycle_len_s == m.sum()


def test_ncmapss_var_name_mismatch_raises(tmp_path):
    write_synthetic_ncmapss(tmp_path / "Data" / "N-CMAPSS", dataset="DS03",
                            rename_sensor="BOGUS", seed=4)
    cfg = _ncmapss_cfg(tmp_path / "Data", tmp_path, dataset="DS03")
    with pytest.raises(ValueError, match="do not match"):
        load_ncmapss(cfg)


def test_ncmapss_truncation_in_key_only_for_ncmapss(tmp_path):
    """ncmapss_test_truncation changes a DS0x window-cache key but NOT an FD001 key."""
    ds = Config(dataset="DS02", window_size=4)
    ds2 = ds.replace(ncmapss_test_truncation=0.5)
    assert ds.window_cache_key() != ds2.window_cache_key()
    fd = Config(dataset="FD001")
    fd2 = fd.replace(ncmapss_test_truncation=0.5)
    assert fd.window_cache_key() == fd2.window_cache_key()


def test_ncmapss_aggregate_cache_reused_and_versioned(tmp_path, monkeypatch):
    write_synthetic_ncmapss(tmp_path / "Data" / "N-CMAPSS", dataset="DS02", seed=1)
    cfg = _ncmapss_cfg(tmp_path / "Data", tmp_path)
    load_ncmapss(cfg)   # builds the aggregate cache
    # second load must NOT reopen the h5
    import src.datasets.ncmapss as NC
    monkeypatch.setattr(NC, "_read_and_aggregate",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("re-parsed h5")))
    load_ncmapss(cfg)   # served from cache -> no _read_and_aggregate call
    # bumping the aggregate version forces a rebuild (would call the patched raiser)
    monkeypatch.setattr(NC, "NCMAPSS_AGG_VERSION", NC.NCMAPSS_AGG_VERSION + 1)
    with pytest.raises(AssertionError, match="re-parsed h5"):
        load_ncmapss(cfg)


def test_ncmapss_end_to_end_cache_and_sweep(tmp_path):
    write_synthetic_ncmapss(tmp_path / "Data" / "N-CMAPSS", dataset="DS02",
                            n_dev_units=3, n_test_units=2, seed=7)
    cfg = _ncmapss_cfg(tmp_path / "Data", tmp_path)
    df_train, df_test = D.load_prepared(cfg)     # condition_norm auto-OFF
    assert not cfg.effective_condition_norm()
    w, y, u = D.make_windows(df_train, cfg.sensor_columns, cfg.window_size,
                             target_col="clipped_rul")
    assert w.shape[1:] == (cfg.window_size, 37)
    from src.embeddings import build_embedding_cache, load_embedding_cache
    from src.sweep import run_sweep
    build_embedding_cache(cfg, embedder=MockEmbedder(feature_dim=12))
    cache = load_embedding_cache(cfg)
    assert cache["test_emb"].shape[0] == 2   # one last-context per test unit
    csv = run_sweep(cfg, device="cpu", baseline_names=["predict_mean", "gbm"])
    assert Path(csv).exists()


def test_ncmapss_registry_and_defaults():
    from src.config import NCMAPSS_FEATURE_COLUMNS
    from src import datasets as DS
    assert Config(dataset="DS03").dataset_kind() == "ncmapss"
    assert Config(dataset="DS08d").dataset_kind() == "ncmapss"
    assert Config(dataset="DS02").sensor_columns == list(NCMAPSS_FEATURE_COLUMNS)
    assert "ncmapss" in DS.DATASET_LOADERS and "ncmapss" in DS.DATASET_FAMILIES


# ---------------------------------------------------------------------------
# §24 fixes: registry consistency, sensor-column defaults, experiment_name guard
# ---------------------------------------------------------------------------
def test_dataset_kind_and_registry_never_drift():
    """Every dataset name any family serves must round-trip through
    config.dataset_kind() into a registered loader family, and every registry
    kind must serve at least one name -- adding a dataset family requires
    touching both, and this test is the drift alarm."""
    from src import datasets as DS
    for name in DS.all_dataset_names():
        kind = Config(dataset=name).dataset_kind()
        assert kind in DS.DATASET_LOADERS and kind in DS.DATASET_FAMILIES
    assert set(DS.DATASET_LOADERS) == set(DS.DATASET_FAMILIES)
    served = {Config(dataset=n).dataset_kind() for n in DS.all_dataset_names()}
    assert served == set(DS.DATASET_FAMILIES)


def test_sensor_columns_default_resolution_and_key_stability():
    from src.config import (Config, FD001_NONCONSTANT_SENSORS,
                            XJTU_FEATURE_COLUMNS)
    c = Config()  # sensor_columns=None -> FD001 default
    assert c.sensor_columns == list(FD001_NONCONSTANT_SENSORS)
    # resolved default hashes identically to the previously-required explicit list
    assert (c.embedding_cache_key()
            == Config(sensor_columns=list(FD001_NONCONSTANT_SENSORS)).embedding_cache_key())
    # replace() with sensor_columns=None re-resolves for the NEW dataset
    x = c.replace(dataset="XJTU-SY", sensor_columns=None)
    assert x.sensor_columns == list(XJTU_FEATURE_COLUMNS)
    # a dataset switch WITHOUT resetting keeps the explicit list (caller's choice)
    kept = c.replace(dataset="XJTU-SY")
    assert kept.sensor_columns == list(FD001_NONCONSTANT_SENSORS)


def test_experiment_name_guard():
    Config(experiment_name="fd001_chronos-2.v1")           # allowed charset
    for bad in ("has space", "slash/y", "semi;colon"):
        with pytest.raises(ValueError, match="experiment_name"):
            Config(experiment_name=bad)


def test_plot_data_scaling_facets_by_dataset(tmp_path):
    """A results CSV holding two datasets must yield per-dataset figures, never
    one pooled curve (the pre-§24 silent-mixing bug)."""
    import matplotlib
    matplotlib.use("Agg")
    from src.evaluate import append_result_row
    from src.plots import plot_data_scaling

    csv_path = tmp_path / "results_v2.csv"
    for ds, base in (("FD001", 20.0), ("FD003", 40.0)):
        for n_units in (2, 4):
            for seed in (0, 1):
                append_result_row(csv_path, {
                    "model": "gbm", "dataset": ds, "n_units": n_units,
                    "seed": seed, "loss": "native",
                    "rmse_clipped": base - n_units + seed})
    saved = plot_data_scaling(csv_path, tmp_path / "figs", show=False,
                              metrics=[("rmse_clipped", "test RMSE")])
    names = {p.name for p in saved}
    assert "data_scaling_FD001_rmse_clipped.png" in names
    assert "data_scaling_FD003_rmse_clipped.png" in names
