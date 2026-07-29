"""CPU tests for the Phase-B FEATURE/AGGREGATION mode knobs (CHANGES.md §52-§53).

Two interventions that change what the LOADER emits rather than what the model does:
  * XJTU-SY ``xjtu_feature_mode`` -- raw samples vs hand-crafted indicators (RQ-D);
  * N-CMAPSS ``ncmapss_agg_stride`` / ``ncmapss_agg_stats`` -- how finely the 1 Hz
    within-flight rows are sampled and summarized (RQ-G).

The load-bearing properties tested here are (a) the channel set the loader emits equals
the channel set ``Config`` resolves, in the same ORDER, at every mode; (b) the raw
reductions emit real readings and fail loud rather than fabricating; (c) every new field
re-keys the cache only when non-default; and (d) the recorded default keys never drift.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import (Config, XJTU_FEATURE_COLUMNS, NCMAPSS_AGG_STAT_SETS,
                        ncmapss_feature_columns, xjtu_raw_columns)
from src import data as D
from src.datasets import ncmapss as NC
from src.datasets import xjtu as XJ
from tests.synthetic import write_synthetic_ncmapss, write_synthetic_xjtu


# ---------------------------------------------------------------------------
# §52 -- XJTU raw-vs-indicators
# ---------------------------------------------------------------------------
def _xjtu_cfg(tmp_path: Path, **over) -> Config:
    base = dict(dataset="XJTU-SY", data_dir=str(tmp_path / "XJTU-SY"),
                cache_dir=str(tmp_path / "cache"), results_dir=str(tmp_path / "results"),
                window_size=6, max_rul=40,
                xjtu_test_bearings=["Bearing1_4", "Bearing1_5"],
                xjtu_test_truncation=0.6)
    base.update(over)
    return Config(**base)


def test_snapshot_raw_decimate_keeps_real_samples():
    """``decimate`` must emit values that are actually IN the snapshot (first and last
    inclusive) -- it is a subtractive collection choice, not a transform."""
    x = np.arange(100, dtype=np.float64)
    vals = XJ.snapshot_raw(x, 5, "decimate")
    # np.linspace(0, 99, 5) -> 0, 24.75, 49.5, 74.25, 99 -> rounded to real indices
    assert vals == [0.0, 25.0, 50.0, 74.0, 99.0]      # evenly spaced, endpoints included
    assert all(v in set(x.tolist()) for v in vals)     # every value is a real reading
    assert len(XJ.snapshot_raw(x, 100, "decimate")) == 100   # exactly-enough samples


def test_snapshot_raw_segment_rms_preserves_full_rate_energy():
    """``segment_rms`` summarizes every sample, so a constant-magnitude signal keeps its
    amplitude even though the sample rate is coarsened (unlike decimation of an
    oscillating signal)."""
    x = np.array([3.0, -3.0] * 50)                     # RMS 3 everywhere
    vals = XJ.snapshot_raw(x, 4, "segment_rms")
    assert np.allclose(vals, 3.0)
    assert len(vals) == 4


def test_snapshot_raw_fails_loud_when_too_short():
    with pytest.raises(ValueError, match="xjtu_raw_channels"):
        XJ.snapshot_raw(np.arange(4.0), 8, "decimate")


@pytest.mark.parametrize("mode,expected", [
    ("indicators", 16), ("raw", 8), ("raw+indicators", 24),
])
def test_channel_columns_match_config_resolution(tmp_path, mode, expected):
    """The loader's emitted columns and ``Config.default_sensor_columns()`` must agree
    EXACTLY, order included -- they are two sides of one contract."""
    cfg = _xjtu_cfg(tmp_path, xjtu_feature_mode=mode, xjtu_raw_channels=4)
    assert XJ.xjtu_channel_columns(cfg) == list(cfg.default_sensor_columns())
    assert cfg.sensor_columns == XJ.xjtu_channel_columns(cfg)
    assert len(cfg.sensor_columns) == expected


def test_raw_block_precedes_indicators_in_combined_mode(tmp_path):
    cfg = _xjtu_cfg(tmp_path, xjtu_feature_mode="raw+indicators", xjtu_raw_channels=3)
    cols = cfg.sensor_columns
    assert cols[:6] == xjtu_raw_columns(3)
    assert cols[6:] == list(XJTU_FEATURE_COLUMNS)


@pytest.mark.parametrize("mode", ["indicators", "raw", "raw+indicators"])
@pytest.mark.parametrize("reduce", ["decimate", "segment_rms"])
def test_xjtu_loader_end_to_end_per_mode(tmp_path, mode, reduce):
    """Every mode loads into the canonical frame with exactly the resolved channels."""
    write_synthetic_xjtu(tmp_path / "XJTU-SY", bearings_per_condition=5,
                         min_snapshots=10, max_snapshots=14,
                         samples_per_snapshot=64, seed=3)
    cfg = _xjtu_cfg(tmp_path, xjtu_feature_mode=mode, xjtu_raw_channels=4,
                    xjtu_raw_reduce=reduce)
    df_train, df_test, rul = XJ.load_xjtu(cfg)
    for frame in (df_train, df_test):
        assert list(frame.columns) == (
            ["unit_number", "time_cycles", "setting_1", "setting_2", "setting_3"]
            + cfg.sensor_columns)
        assert frame[cfg.sensor_columns].notna().all().all()
    assert rul.name == "rul_truth" and rul.index.name == "unit_number"
    assert (rul > 0).all()


def test_xjtu_raw_mode_flows_through_load_prepared(tmp_path):
    """The ONE loading path must serve the raw arm too (labels + condition norm applied
    to the raw channels exactly as to the indicator ones)."""
    write_synthetic_xjtu(tmp_path / "XJTU-SY", bearings_per_condition=5,
                         min_snapshots=10, max_snapshots=14,
                         samples_per_snapshot=64, seed=4)
    cfg = _xjtu_cfg(tmp_path, xjtu_feature_mode="raw", xjtu_raw_channels=4)
    df_train, df_test = D.load_prepared(cfg)
    assert {"actual_rul", "clipped_rul"} <= set(df_train.columns)
    assert set(cfg.sensor_columns) <= set(df_train.columns)
    # condition_norm resolves auto-ON for XJTU, so the channels are standardized
    assert abs(float(df_train[cfg.sensor_columns].to_numpy().mean())) < 1.0


def test_xjtu_feature_mode_keys_only_when_engaged(tmp_path):
    """The recorded indicator-mode key must be byte-identical; every raw variant must
    key APART from it and from each other."""
    base = Config(dataset="XJTU-SY")
    assert base.window_cache_key() == "windows_XJTU-SY_97e96700cc2670b4"
    for absent in ("xjtu_feature_mode", "xjtu_raw_channels", "xjtu_raw_reduce"):
        assert absent not in base._window_key_fields()
    keys = {base.window_cache_key()}
    for over in ({"xjtu_feature_mode": "raw"},
                 {"xjtu_feature_mode": "raw+indicators"},
                 {"xjtu_feature_mode": "raw", "xjtu_raw_channels": 8},
                 {"xjtu_feature_mode": "raw", "xjtu_raw_reduce": "segment_rms"}):
        cfg = Config(dataset="XJTU-SY", sensor_columns=None, **over)
        assert "xjtu_feature_mode" in cfg._window_key_fields()
        keys.add(cfg.window_cache_key())
    assert len(keys) == 5, "each raw variant must get its own cache"


def test_xjtu_mode_fields_never_leak_into_other_families():
    """xjtu-only fields must not re-key a C-MAPSS or N-CMAPSS cache."""
    for ds, expected in (("FD001", "windows_FD001_1da313c871251cec"),
                         ("DS02", "windows_DS02_ba4dfa4567c86cba")):
        cfg = Config(dataset=ds, xjtu_feature_mode="raw", xjtu_raw_channels=99,
                     xjtu_raw_reduce="segment_rms")
        assert cfg.window_cache_key() == expected


def test_config_rejects_bad_xjtu_mode_values():
    with pytest.raises(ValueError, match="xjtu_feature_mode"):
        Config(dataset="XJTU-SY", xjtu_feature_mode="waveform")
    with pytest.raises(ValueError, match="xjtu_raw_reduce"):
        Config(dataset="XJTU-SY", xjtu_raw_reduce="fft")
    with pytest.raises(ValueError, match="xjtu_raw_channels"):
        Config(dataset="XJTU-SY", xjtu_raw_channels=0)


# ---------------------------------------------------------------------------
# §53 -- N-CMAPSS aggregation granularity
# ---------------------------------------------------------------------------
def _nc_cfg(tmp_path: Path, **over) -> Config:
    base = dict(dataset="DS02", data_dir=str(tmp_path / "N-CMAPSS"),
                cache_dir=str(tmp_path / "cache"), results_dir=str(tmp_path / "results"),
                window_size=4, max_rul=40, ncmapss_test_truncation=0.6)
    base.update(over)
    return Config(**base)


def test_stat_sets_resolve_the_expected_channel_counts():
    assert len(ncmapss_feature_columns("mean_std")) == 18 * 2 + 1
    assert len(ncmapss_feature_columns("mean_std_minmax_slope")) == 18 * 5 + 1
    # interleaved per variable, cycle_len_s last
    cols = ncmapss_feature_columns("mean_std_minmax_slope")
    assert cols[:5] == ["alt_mean", "alt_std", "alt_min", "alt_max", "alt_slope"]
    assert cols[-1] == "cycle_len_s"


@pytest.mark.parametrize("stats", sorted(NCMAPSS_AGG_STAT_SETS))
def test_ncmapss_loads_at_each_stat_set(tmp_path, stats):
    write_synthetic_ncmapss(tmp_path / "N-CMAPSS", dataset="DS02", n_dev_units=3,
                            n_test_units=2, min_cycles=8, max_cycles=10,
                            min_rows=12, max_rows=18, seed=5)
    cfg = _nc_cfg(tmp_path, ncmapss_agg_stats=stats)
    df_train, df_test, rul = NC.load_ncmapss(cfg)
    assert list(cfg.sensor_columns) == ncmapss_feature_columns(stats)
    assert set(cfg.sensor_columns) <= set(df_train.columns)
    assert np.isfinite(df_train[cfg.sensor_columns].to_numpy()).all()
    assert (rul > 0).all()


def test_stride_subsamples_rows_but_keeps_cycle_len_s_at_the_full_count(tmp_path):
    """The RQ-G intervention must coarsen the SENSOR sampling only: flight duration is
    observable regardless of poll rate, so ``cycle_len_s`` must NOT change with stride
    (otherwise the probe confounds two collection choices -- §53)."""
    write_synthetic_ncmapss(tmp_path / "N-CMAPSS", dataset="DS02", n_dev_units=3,
                            n_test_units=2, min_cycles=8, max_cycles=10,
                            min_rows=20, max_rows=20, seed=6)
    full = NC.load_ncmapss(_nc_cfg(tmp_path))[0]
    strided = NC.load_ncmapss(_nc_cfg(tmp_path, ncmapss_agg_stride=4))[0]
    assert len(full) == len(strided)                       # same cycles, fewer samples
    assert np.array_equal(full["cycle_len_s"].to_numpy(),
                          strided["cycle_len_s"].to_numpy())
    assert (full["cycle_len_s"] == 20).all()
    # the statistics DID change (fewer rows entered them)
    assert not np.allclose(full["T24_std"].to_numpy(), strided["T24_std"].to_numpy())


def test_stride_keeps_every_flight_nonempty(tmp_path):
    """Row 0 of each flight is always retained, so a stride larger than a flight's
    length still yields exactly one row per flight -- never an empty group."""
    write_synthetic_ncmapss(tmp_path / "N-CMAPSS", dataset="DS02", n_dev_units=2,
                            n_test_units=2, min_cycles=8, max_cycles=9,
                            min_rows=6, max_rows=8, seed=7)
    cfg = _nc_cfg(tmp_path, ncmapss_agg_stride=999)
    df_train, _df_test, _rul = NC.load_ncmapss(cfg)
    assert len(df_train) > 0
    # one retained row per flight => std is the 1-row convention (0.0)
    assert np.allclose(df_train["T24_std"].to_numpy(), 0.0)


def test_group_slopes_recovers_a_known_linear_trend():
    """Closed-form check of the vectorized cov(t,x)/var(t) slope."""
    df = pd.DataFrame({
        "__unit": [1] * 5 + [2] * 5,
        "__cycle": [1] * 5 + [1] * 5,
        "__t": list(range(5)) * 2,
        "a": [0.0, 2.0, 4.0, 6.0, 8.0] + [10.0, 7.0, 4.0, 1.0, -2.0],
    })
    slopes = NC._group_slopes(df, ["__unit", "__cycle"], "__t", ["a"])
    assert np.allclose(slopes["a"].to_numpy(), [2.0, -3.0])


def test_group_slopes_zero_variance_group_is_zero_not_nan():
    df = pd.DataFrame({"__unit": [1], "__cycle": [1], "__t": [0.0], "a": [5.0]})
    slopes = NC._group_slopes(df, ["__unit", "__cycle"], "__t", ["a"])
    assert float(slopes["a"].iloc[0]) == 0.0


def test_aggregate_cache_filename_carries_the_knobs(tmp_path):
    """Each knob combination gets its OWN coexisting aggregate, and the default keeps
    the historical filename so pre-§53 caches stay valid."""
    default = NC._agg_cache_path(_nc_cfg(tmp_path), "DS02")
    assert default.name == "ncmapss_agg_DS02_v1.npz"
    assert NC._agg_cache_path(_nc_cfg(tmp_path, ncmapss_agg_stride=10),
                              "DS02").name == "ncmapss_agg_DS02_v1_s10.npz"
    rich = NC._agg_cache_path(
        _nc_cfg(tmp_path, ncmapss_agg_stride=2,
                ncmapss_agg_stats="mean_std_minmax_slope"), "DS02")
    assert rich.name == "ncmapss_agg_DS02_v1_s2_mean_std_minmax_slope.npz"


def test_aggregate_cache_roundtrips_per_knob(tmp_path):
    """A second load hits the cache (no re-parse) and returns the same frame; a
    different stat set builds a SEPARATE cache rather than reusing the first."""
    write_synthetic_ncmapss(tmp_path / "N-CMAPSS", dataset="DS02", n_dev_units=2,
                            n_test_units=2, min_cycles=8, max_cycles=9,
                            min_rows=12, max_rows=14, seed=8)
    cfg = _nc_cfg(tmp_path)
    first = NC.load_ncmapss(cfg)[0]
    assert NC._agg_cache_path(cfg, "DS02").exists()
    second = NC.load_ncmapss(cfg)[0]                     # cache-hit path
    assert np.allclose(first[cfg.sensor_columns].to_numpy(),
                       second[cfg.sensor_columns].to_numpy())
    rich_cfg = _nc_cfg(tmp_path, ncmapss_agg_stats="mean_std_minmax_slope")
    rich = NC.load_ncmapss(rich_cfg)[0]
    assert NC._agg_cache_path(rich_cfg, "DS02").exists()
    assert len(rich.columns) > len(first.columns)


def test_ncmapss_agg_keys_only_when_non_default():
    base = Config(dataset="DS02")
    assert base.window_cache_key() == "windows_DS02_ba4dfa4567c86cba"
    for absent in ("ncmapss_agg_stride", "ncmapss_agg_stats"):
        assert absent not in base._window_key_fields()
    keys = {base.window_cache_key()}
    for over in ({"ncmapss_agg_stride": 5},
                 {"ncmapss_agg_stats": "mean_std_minmax_slope"},
                 {"ncmapss_agg_stride": 5, "ncmapss_agg_stats": "mean_std_minmax_slope"}):
        keys.add(Config(dataset="DS02", sensor_columns=None, **over).window_cache_key())
    assert len(keys) == 4
    # ... and never leak into another family
    assert Config(dataset="FD001", ncmapss_agg_stride=7,
                  ncmapss_agg_stats="mean_std_minmax_slope"
                  ).window_cache_key() == "windows_FD001_1da313c871251cec"


def test_config_rejects_bad_ncmapss_agg_values():
    with pytest.raises(ValueError, match="ncmapss_agg_stats"):
        Config(dataset="DS02", ncmapss_agg_stats="median")
    with pytest.raises(ValueError, match="ncmapss_agg_stride"):
        Config(dataset="DS02", ncmapss_agg_stride=0)


def test_channel_set_factor_probe_override_reresolves(tmp_path):
    """A ``feature_mode``/``aggregation`` probe level must re-resolve sensor_columns --
    the regression this guard exists for (§52 probe wiring)."""
    from src import probes as P
    cfg = _nc_cfg(tmp_path)
    assert len(cfg.sensor_columns) == 37
    over = P._level_overrides("aggregation",
                              {"ncmapss_agg_stats": "mean_std_minmax_slope"})
    assert over["sensor_columns"] is None
    assert len(cfg.replace(**over).sensor_columns) == 91
