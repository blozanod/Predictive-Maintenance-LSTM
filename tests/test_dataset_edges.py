"""Fail-loud guards and rarely-taken branches of the dataset loaders.

The XJTU-SY and N-CMAPSS loaders adapt real, messy directory layouts into the canonical
frame, so almost every guard here exists because a real layout can violate it: missing
or renamed folders, an unreadable snapshot, an ambiguous per-dataset file, dev/test unit
collisions, a trajectory too short to truncate. Repo invariant §7 says each of those must
raise with the expected AND the observed value, never adapt silently -- these tests hold
that line. CPU-only, synthetic fixtures, no downloads.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import Config, NCMAPSS_FEATURE_COLUMNS
from src.datasets import ncmapss as NC
from src.datasets import xjtu as XJ
from tests.synthetic import write_synthetic_ncmapss


# ---------------------------------------------------------------------------
# XJTU-SY
# ---------------------------------------------------------------------------
def _write_bearing(bdir: Path, n_snapshots: int, samples: int = 32, columns: int = 2):
    bdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    header = ",".join(["Horizontal_vibration_signals",
                       "Vertical_vibration_signals"][:columns] or ["only_one"])
    for i in range(1, n_snapshots + 1):
        x = rng.normal(0, 1 + i / n_snapshots, size=(samples, columns))
        np.savetxt(bdir / f"{i}.csv", x, delimiter=",", header=header,
                   comments="", fmt="%.5f")


def _xjtu_cfg(root: Path, tmp_path: Path, **over) -> Config:
    base = dict(dataset="XJTU-SY", data_dir=str(root),
                cache_dir=str(tmp_path / "cache"), results_dir=str(tmp_path / "results"),
                window_size=6, max_rul=15, xjtu_test_truncation=0.6,
                xjtu_test_bearings=["Bearing1_2"])
    base.update(over)
    return Config(**base)


def test_xjtu_bearing_folder_without_snapshots_raises(tmp_path):
    empty = tmp_path / "35Hz12kN" / "Bearing1_1"
    empty.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="no snapshot CSVs"):
        # _bearing_frame takes the config since §52 (the feature mode selects channels).
        XJ._bearing_frame(empty, 1, (0, 35.0, 12.0), Config(dataset="XJTU-SY"))


def test_xjtu_single_column_snapshot_raises(tmp_path):
    bdir = tmp_path / "35Hz12kN" / "Bearing1_1"
    _write_bearing(bdir, n_snapshots=2, columns=1)
    with pytest.raises(ValueError, match="expected 2 columns"):
        XJ._bearing_frame(bdir, 1, (0, 35.0, 12.0), Config(dataset="XJTU-SY"))


def test_xjtu_descend_gives_up_on_an_unrelated_tree(tmp_path):
    """Depth-1 descent scans immediate subdirectories only; when none holds condition
    folders the ORIGINAL root is returned so the caller's error names the documented
    path instead of some arbitrary child."""
    root = tmp_path / "Data" / "XJTU-SY"
    (root / "__MACOSX").mkdir(parents=True)
    (root / "docs" / "deeper").mkdir(parents=True)
    assert XJ._descend_to_conditions(root, verbose=False) == root


def test_xjtu_unmatched_condition_check_ignores_a_missing_root(tmp_path):
    XJ._check_unmatched_conditions(tmp_path / "does-not-exist")     # no raise


def test_xjtu_loads_a_partial_condition_set(tmp_path):
    """Only one of the three condition folders downloaded: the absent ones are skipped
    (not an error), and the split still resolves."""
    root = tmp_path / "XJTU-SY"
    for b in (1, 2):
        _write_bearing(root / "35Hz12kN" / f"Bearing1_{b}", n_snapshots=12)
    cfg = _xjtu_cfg(root, tmp_path)
    df_train, df_test, rul = XJ.load_xjtu(cfg)
    assert df_train["unit_number"].nunique() == 1
    assert df_test["unit_number"].nunique() == 1 and (rul > 0).all()
    assert set(XJ.XJTU_FEATURE_COLUMNS) <= set(df_train.columns)


def test_xjtu_without_any_condition_folder_raises(tmp_path):
    root = tmp_path / "XJTU-SY"
    root.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="no XJTU-SY condition folders"):
        XJ.load_xjtu(_xjtu_cfg(root, tmp_path, xjtu_test_bearings=[]))


def test_xjtu_bearing_too_short_to_truncate_raises(tmp_path):
    """window_size larger than the test bearing's life leaves no valid prefix -- raise
    naming the bearing and the window, rather than emitting a 0-window test unit."""
    root = tmp_path / "XJTU-SY"
    for b in (1, 2):
        _write_bearing(root / "35Hz12kN" / f"Bearing1_{b}", n_snapshots=4)
    cfg = _xjtu_cfg(root, tmp_path, window_size=6)
    with pytest.raises(ValueError, match="cannot truncate"):
        XJ.load_xjtu(cfg)


def test_xjtu_split_that_holds_out_everything_raises(tmp_path):
    root = tmp_path / "XJTU-SY"
    _write_bearing(root / "35Hz12kN" / "Bearing1_1", n_snapshots=12)
    cfg = _xjtu_cfg(root, tmp_path, xjtu_test_bearings=["Bearing1_1"])
    with pytest.raises(ValueError, match="empty train or test set"):
        XJ.load_xjtu(cfg)


def test_xjtu_is_available_is_false_without_data(tmp_path):
    assert not XJ.is_available(_xjtu_cfg(tmp_path / "missing", tmp_path))


# ---------------------------------------------------------------------------
# N-CMAPSS
# ---------------------------------------------------------------------------
def _nc_cfg(root: Path, tmp_path: Path, dataset="DS02", **over) -> Config:
    base = dict(dataset=dataset, data_root=str(root), data_dir=None,
                cache_dir=str(tmp_path / "cache"), results_dir=str(tmp_path / "results"),
                window_size=4, max_rul=125, ncmapss_test_truncation=0.6)
    base.update(over)
    return Config(**base)


def test_ncmapss_missing_and_ambiguous_files_raise(tmp_path):
    ncdir = tmp_path / "Data" / "N-CMAPSS"
    ncdir.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="no N-CMAPSS file matching"):
        NC._find_h5(ncdir, "DS02")
    write_synthetic_ncmapss(ncdir, dataset="DS02", seed=1, suffix="-000")
    write_synthetic_ncmapss(ncdir, dataset="DS02", seed=2, suffix="-001")
    with pytest.raises(ValueError, match="ambiguous N-CMAPSS files"):
        NC._find_h5(ncdir, "DS02")


def test_ncmapss_is_available_per_file_and_without_a_root(tmp_path):
    cfg = _nc_cfg(tmp_path / "Data", tmp_path)
    assert not NC.is_available(cfg)                          # root missing entirely
    write_synthetic_ncmapss(tmp_path / "Data" / "N-CMAPSS", dataset="DS02", seed=1)
    assert NC.is_available(cfg)                              # the per-file glob path
    assert not NC.is_available(cfg.replace(dataset="DS05"))


def test_ncmapss_dev_test_unit_collision_raises(tmp_path):
    """Dev/test unit ids must be disjoint WITHIN a file -- a collision would silently
    train and test on the same engine, so it is a hard error, not an assumption."""
    import h5py
    path = write_synthetic_ncmapss(tmp_path / "Data" / "N-CMAPSS", dataset="DS03",
                                   n_dev_units=2, n_test_units=1, seed=5)
    with h5py.File(path, "r+") as h:
        a_test = np.asarray(h["A_test"])
        a_test[:, 0] = 1                                     # a dev unit id
        del h["A_test"]
        h.create_dataset("A_test", data=a_test)
    with pytest.raises(ValueError, match="dev and test share unit ids"):
        NC._read_and_aggregate(path)


def test_ncmapss_aggregate_is_quiet_when_verbose_is_off(tmp_path, capsys):
    """Both the parse and the cache-hit path take a silent branch (the campaign runs
    them per combo; the notebooks want the notice)."""
    write_synthetic_ncmapss(tmp_path / "Data" / "N-CMAPSS", dataset="DS02", seed=1)
    cfg = _nc_cfg(tmp_path / "Data", tmp_path)
    NC._load_or_build_aggregate(cfg, "DS02", verbose=False)          # parses the h5
    NC._load_or_build_aggregate(cfg, "DS02", verbose=False)          # served from cache
    assert capsys.readouterr().out == ""


def test_ncmapss_test_unit_too_short_to_truncate_raises(tmp_path):
    cfg = _nc_cfg(tmp_path / "Data", tmp_path, window_size=10)
    df = pd.DataFrame({"unit_number": [100] * 5, "time_cycles": range(1, 6),
                       **{c: 0.0 for c in NCMAPSS_FEATURE_COLUMNS}})
    with pytest.raises(ValueError, match="cannot truncate"):
        NC._truncate_test(df, cfg)


def test_dsall_member_discovery_and_guards(tmp_path):
    cfg = _nc_cfg(tmp_path / "Data", tmp_path, dataset="DSALL")
    assert NC.dsall_members_on_disk(cfg) == []                # root does not exist
    with pytest.raises(FileNotFoundError, match=">= 2 N-CMAPSS_DS"):
        NC._resolve_dsall_members(cfg)                        # auto mode, nothing on disk
    with pytest.raises(ValueError, match="non-member name"):
        NC._resolve_dsall_members(cfg.replace(dsall_datasets=["DS02", "DSALL"]))
