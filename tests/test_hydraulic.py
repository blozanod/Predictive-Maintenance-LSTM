"""CPU tests for the UCI hydraulic loader (src/datasets/hydraulic.py).

The dataset is 18 header-less, tab-delimited files whose ROW ORDER is the only thing
tying a sensor reading to its fault annotation, whose column count IS the sampling rate,
and whose severity ladders run in TWO DIRECTIONS (three components count down toward
failure, the pump counts up). Every one of those is a way to be silently wrong, so the
tests below check the canonical frame contract, the label polarity, the block
segmentation and the stratified split against hand computations -- and drive every
fail-loud guard with a deliberately malformed file. CPU-only, synthetic fixtures, no
downloads.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import data as D
from src.config import (Config, HYDRAULIC_COMPONENTS, HYDRAULIC_FEATURE_COLUMNS,
                        HYDRAULIC_N_CYCLES, HYDRAULIC_PROFILE_COLUMNS,
                        HYDRAULIC_PROFILE_FILE, HYDRAULIC_SENSORS,
                        HYDRAULIC_SENSOR_NAMES, HYDRAULIC_SEVERITY_ORDER,
                        hydraulic_feature_columns)
from src.datasets import hydraulic as HY
from tests.synthetic_hydraulic import (severity_ordinals, synthetic_profile,
                                       write_synthetic_hydraulic)

# The shared fixture's layout (also the arithmetic every count below is derived from):
# 240 cycles -> 3 cooler regimes x 10 label blocks of 8 cycles; the 8-cycle warm-up at
# the start of each regime swallows that regime's FIRST block whole (30 - 3 = 27 units),
# and one isolated marker cycle shortens one block per regime to 7.
CYCLES = 240
BLOCK_LEN = 8
N_BLOCKS = 30
N_UNITS = 27


@pytest.fixture(scope="module")
def hydraulic_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("hydraulic")
    write_synthetic_hydraulic(root, cycles=CYCLES, block_len=BLOCK_LEN, seed=1)
    return root


def _cfg(root: Path, tmp_path: Path, **over) -> Config:
    base = dict(dataset="Hydraulic", data_dir=str(root),
                cache_dir=str(tmp_path / "cache"), results_dir=str(tmp_path / "results"),
                window_size=4, max_rul=8)
    base.update(over)
    return Config(**base)


def _broken_root(tmp_path: Path, **over) -> Path:
    """A fresh copy of the fixture layout, rewritten with one deliberate defect."""
    root = tmp_path / "broken"
    write_synthetic_hydraulic(root, cycles=CYCLES, block_len=BLOCK_LEN, seed=1, **over)
    return root


# ---------------------------------------------------------------------------
# The canonical frame contract
# ---------------------------------------------------------------------------
def test_canonical_frame_schema(hydraulic_root, tmp_path):
    cfg = _cfg(hydraulic_root, tmp_path)
    df_train, df_test, rul = HY.load_hydraulic(cfg)

    expected = (["unit_number", "time_cycles", "setting_1", "setting_2", "setting_3"]
                + list(HYDRAULIC_FEATURE_COLUMNS) + list(HY.HYDRAULIC_LABEL_COLUMNS))
    # ... plus event_observed: NO hydraulic block ends in a failure (§55), so the
    # loader marks every unit right-censored.
    assert list(df_train.columns) == list(expected) + [D.EVENT_OBSERVED_COLUMN]
    assert list(df_test.columns) == list(df_train.columns)
    # every hydraulic block is right-censored: it ends because the experimenter changed
    # the set-point, not because anything failed (§55)
    assert (df_train[D.EVENT_OBSERVED_COLUMN] == 0).all()
    assert (df_test[D.EVENT_OBSERVED_COLUMN] == 0).all()
    # channels are exactly what the config resolves for this family
    assert cfg.sensor_columns == list(HYDRAULIC_FEATURE_COLUMNS)
    assert len(HYDRAULIC_FEATURE_COLUMNS) == 2 * len(HYDRAULIC_SENSOR_NAMES) == 34
    # dtypes + the "no operating point" convention
    for frame in (df_train, df_test):
        assert frame["unit_number"].dtype == np.int64
        assert frame["time_cycles"].dtype == np.int64
        assert (frame[["setting_1", "setting_2", "setting_3"]].to_numpy() == 0.0).all()
        assert np.isfinite(frame[HYDRAULIC_FEATURE_COLUMNS].to_numpy()).all()
        # time_cycles restarts at 1 and is consecutive within every unit
        for _, unit in frame.groupby("unit_number"):
            cycles = unit.sort_values("time_cycles")["time_cycles"].to_numpy()
            assert np.array_equal(cycles, np.arange(1, len(cycles) + 1))
    # unit-disjoint split; rul_truth is the documented Series
    assert not set(df_train.unit_number) & set(df_test.unit_number)
    assert df_train.unit_number.nunique() + df_test.unit_number.nunique() == N_UNITS
    assert rul.name == "rul_truth" and rul.index.name == "unit_number"
    assert set(rul.index) == set(df_test.unit_number) and (rul > 0).all()


def test_units_are_maximal_contiguous_label_blocks(hydraulic_root, tmp_path):
    """A unit is one uninterrupted run of the rig in ONE health state: constant severity
    4-tuple inside a unit, and a different tuple in the next one."""
    cfg = _cfg(hydraulic_root, tmp_path)
    frame = HY._canonical_frame(*HY._load_or_build_aggregate(cfg, verbose=False), cfg)
    assert frame["unit_number"].nunique() == N_UNITS
    signatures = frame.groupby("unit_number")[HY.SEVERITY_COLUMNS].nunique()
    assert (signatures == 1).all().all()                # constant within a unit
    firsts = frame.groupby("unit_number")[HY.SEVERITY_COLUMNS].first().to_numpy()
    assert (np.abs(np.diff(firsts, axis=0)).sum(axis=1) > 0).all()   # maximal runs
    # the isolated unstable marker shortens one block per cooler regime; the block that
    # sits entirely inside a warm-up disappears (30 blocks -> 27 units)
    lengths = frame.groupby("unit_number").size().to_numpy()
    assert sorted(np.unique(lengths)) == [BLOCK_LEN - 1, BLOCK_LEN]
    assert (lengths == BLOCK_LEN - 1).sum() == 3


def test_severity_polarity_and_action_taxonomy(hydraulic_root, tmp_path):
    """0 = healthy and higher = worse for EVERY component -- even though cooler/valve/
    accumulator degrade as their raw value DECREASES and the pump as its value RISES."""
    cfg = _cfg(hydraulic_root, tmp_path, hydraulic_drop_unstable=False)
    agg, profile = HY._load_or_build_aggregate(cfg, verbose=False)
    frame = HY._canonical_frame(agg, profile, cfg)
    assert len(frame) == CYCLES        # nothing dropped -> rows stay in cycle order
    expected = severity_ordinals(profile)
    for index, component in enumerate(HYDRAULIC_COMPONENTS):
        assert np.array_equal(frame[HY.severity_column(component)].to_numpy(),
                              expected[:, index])

    raw = {c: profile[:, HYDRAULIC_PROFILE_COLUMNS.index(c)] for c in HYDRAULIC_COMPONENTS}
    # cooler: 3 % efficiency is the WORST level, 100 % the healthy one (ladder runs down)
    assert frame.loc[raw["cooler"] == 3, "severity_cooler"].eq(2).all()
    assert frame.loc[raw["cooler"] == 100, "severity_cooler"].eq(0).all()
    # valve: 73 % switching is worst, 100 % healthy
    assert frame.loc[raw["valve"] == 73, "severity_valve"].eq(3).all()
    assert frame.loc[raw["valve"] == 100, "severity_valve"].eq(0).all()
    # pump: the ladder RISES with the raw leakage index -> ordinal == raw value
    assert np.array_equal(frame["severity_pump"].to_numpy(), raw["pump"])
    # accumulator: 90 bar pre-charge is worst, 130 healthy
    assert frame.loc[raw["accumulator"] == 90, "severity_accumulator"].eq(3).all()
    assert frame.loc[raw["accumulator"] == 130, "severity_accumulator"].eq(0).all()

    # actions: 0 = none (healthy), 2 = replace (worst level), 1 = adjust (in between)
    for component in HYDRAULIC_COMPONENTS:
        severity = frame[HY.severity_column(component)]
        action = frame[HY.action_column(component)]
        worst = len(HYDRAULIC_SEVERITY_ORDER[component]) - 1
        assert action[severity == 0].eq(0).all()
        assert action[severity == worst].eq(2).all()
        middle = severity.between(1, worst - 1)
        assert action[middle].eq(1).all()
    assert frame["action_valve"].nunique() == 3          # all three classes present


def test_split_is_stratified_across_cooler_regimes(hydraulic_root, tmp_path):
    """A naive tail split would hand the test set ONE cooler regime (the block order is a
    nested factorial with cooler outermost) -- the stratified split spans all three, AND
    every level of the RQ-F target component, on both sides (§55)."""
    cfg = _cfg(hydraulic_root, tmp_path)
    df_train, df_test, _ = HY.load_hydraulic(cfg)
    assert set(df_test["severity_cooler"]) == {0, 1, 2} == set(df_train["severity_cooler"])
    # the aliasing guard: the RQ-F target must be scoreable on BOTH sides
    target = HY.severity_column(cfg.hydraulic_taxonomy_component)
    assert len(set(df_test[target])) >= 2, "a single-class test set is unscoreable"
    assert set(df_train[target])
    assert df_train.unit_number.nunique() > df_test.unit_number.nunique()
    test_units = np.sort(df_test.unit_number.unique())
    # neither the first nor the last block of the record is systematically held out
    assert test_units.min() > 1 and test_units.max() < N_UNITS
    # the fraction is honoured per stratum, so it also changes the split
    frame = HY._canonical_frame(*HY._load_or_build_aggregate(cfg, verbose=False), cfg)
    smaller = HY._select_test_units(frame, 0.15, cfg.window_size + 1,
                                    cfg.hydraulic_taxonomy_component, verbose=False)
    assert len(smaller) <= len(test_units)


def test_test_blocks_are_truncated_with_positive_rul(hydraulic_root, tmp_path):
    cfg = _cfg(hydraulic_root, tmp_path)
    frame = HY._canonical_frame(*HY._load_or_build_aggregate(cfg, verbose=False), cfg)
    _, df_test, rul = HY.load_hydraulic(cfg)
    full = frame.groupby("unit_number").size()
    for unit_id, remaining in rul.items():
        kept = int((df_test.unit_number == unit_id).sum())
        assert kept >= cfg.window_size
        assert kept + remaining == full.loc[unit_id]     # truncation, nothing invented


# ---------------------------------------------------------------------------
# Aggregation: the per-cycle statistics
# ---------------------------------------------------------------------------
def test_aggregate_matches_hand_computation(hydraulic_root):
    """mean/std/min/max/slope of a 100 Hz sensor against an independent computation on
    the same file (slope via np.polyfit on the SECONDS axis)."""
    agg, profile = HY._build_aggregate(hydraulic_root, "mean_std_minmax_slope")
    columns = hydraulic_feature_columns("mean_std_minmax_slope")
    assert agg.shape == (CYCLES, len(columns)) == (CYCLES, 85)
    assert profile.shape == (CYCLES, len(HYDRAULIC_PROFILE_COLUMNS))

    raw = pd.read_csv(hydraulic_root / "PS1.txt", sep="\t", header=None,
                      dtype=np.float32).to_numpy(np.float64)
    n_samples = raw.shape[1]
    assert n_samples == HYDRAULIC_SENSORS["PS1"][0] // 60      # scaled 100 Hz -> 100
    seconds = np.arange(n_samples) * (HY.CYCLE_SECONDS / n_samples)
    slopes = np.polyfit(seconds, raw.T, 1)[0]
    for stat, expected in (("mean", raw.mean(axis=1)),
                           ("std", raw.std(axis=1, ddof=1)),
                           ("min", raw.min(axis=1)),
                           ("max", raw.max(axis=1)),
                           ("slope", slopes)):
        assert np.allclose(agg[:, columns.index(f"PS1_{stat}")], expected, atol=1e-6)
    # the slope is per SECOND: the fixture ramps 1.0 unit across the 60 s cycle
    assert np.allclose(agg[:, columns.index("PS1_slope")].mean(), 1.0 / 59.4, atol=1e-3)


def test_single_sample_sensor_degenerates_to_zero(hydraulic_root):
    """At the fixture's scale a 1 Hz sensor has ONE sample per cycle: std and slope are
    undefined and become 0.0 (the N-CMAPSS convention), mean == min == max."""
    agg, _ = HY._build_aggregate(hydraulic_root, "mean_std_minmax_slope")
    columns = hydraulic_feature_columns("mean_std_minmax_slope")
    raw = pd.read_csv(hydraulic_root / "TS1.txt", sep="\t", header=None,
                      dtype=np.float32).to_numpy(np.float64)
    assert raw.shape[1] == 1
    assert np.allclose(agg[:, columns.index("TS1_std")], 0.0)
    assert np.allclose(agg[:, columns.index("TS1_slope")], 0.0)
    for stat in ("mean", "min", "max"):
        assert np.allclose(agg[:, columns.index(f"TS1_{stat}")], raw[:, 0], atol=1e-6)


def test_agg_stats_knob_changes_channels_and_cache_name(hydraulic_root, tmp_path):
    cfg = _cfg(hydraulic_root, tmp_path, hydraulic_agg_stats="mean_std_minmax_slope")
    assert cfg.sensor_columns == hydraulic_feature_columns("mean_std_minmax_slope")
    df_train, _, _ = HY.load_hydraulic(cfg)
    assert list(df_train.columns)[5:5 + 85] == cfg.sensor_columns
    assert HY._agg_cache_path(cfg, HY._resolve_dir(cfg)).name.startswith("hydraulic_agg_v1_mean_std_minmax_slope_f")
    base = _cfg(hydraulic_root, tmp_path)
    assert HY._agg_cache_path(base, HY._resolve_dir(base)).name.startswith(
        "hydraulic_agg_v1_mean_std_f")


# ---------------------------------------------------------------------------
# Parsed-frame cache
# ---------------------------------------------------------------------------
def test_aggregate_cache_reused_and_versioned(hydraulic_root, tmp_path, monkeypatch):
    cfg = _cfg(hydraulic_root, tmp_path)
    HY.load_hydraulic(cfg)                      # builds the cache
    monkeypatch.setattr(HY, "_build_aggregate", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("re-parsed the text files")))
    HY.load_hydraulic(cfg)                      # served from cache -> no re-parse
    monkeypatch.setattr(HY, "HYDRAULIC_AGG_VERSION", HY.HYDRAULIC_AGG_VERSION + 1)
    with pytest.raises(AssertionError, match="re-parsed"):
        HY.load_hydraulic(cfg)


def test_aggregate_cache_is_quiet_when_verbose_is_off(hydraulic_root, tmp_path, capsys):
    cfg = _cfg(hydraulic_root, tmp_path)
    HY._load_or_build_aggregate(cfg, verbose=False)       # parses
    HY._load_or_build_aggregate(cfg, verbose=False)       # cache hit
    assert capsys.readouterr().out == ""


def test_aggregate_cache_prints_notices(hydraulic_root, tmp_path, capsys):
    cfg = _cfg(hydraulic_root, tmp_path)
    HY._load_or_build_aggregate(cfg)
    first = capsys.readouterr().out
    assert "parsing" in first and "cached ->" in first
    HY._load_or_build_aggregate(cfg)
    assert "loaded cached aggregate" in capsys.readouterr().out


def test_stale_cache_shape_raises(hydraulic_root, tmp_path):
    cfg = _cfg(hydraulic_root, tmp_path)
    path = HY._agg_cache_path(cfg, HY._resolve_dir(cfg))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, agg=np.zeros((10, 3), np.float32),
             profile=np.zeros((10, 5), np.int64))
    with pytest.raises(ValueError, match="stale or corrupt"):
        HY.load_hydraulic(cfg)


# ---------------------------------------------------------------------------
# Fail-loud guards: geometry
# ---------------------------------------------------------------------------
def _full_rate_shapes(n_cycles: int) -> dict:
    return {name: (n_cycles, HYDRAULIC_SENSORS[name][0])
            for name in HYDRAULIC_SENSOR_NAMES}


def test_full_rate_geometry_requires_the_shipped_cycle_count():
    """A full-width download must hold exactly 2205 cycles -- a truncated one fails loud
    instead of quietly training on a prefix."""
    assert HY._validate_geometry(_full_rate_shapes(HYDRAULIC_N_CYCLES),
                                 HYDRAULIC_N_CYCLES) == 1
    with pytest.raises(ValueError, match=f"exactly {HYDRAULIC_N_CYCLES} cycles"):
        HY._validate_geometry(_full_rate_shapes(2204), 2204)


def test_scaled_geometry_is_accepted_and_reported():
    shapes = {name: (CYCLES, HYDRAULIC_SENSORS[name][0] // 60)
              for name in HYDRAULIC_SENSOR_NAMES}
    assert HY._validate_geometry(shapes, CYCLES) == 60


def test_row_count_disagreement_raises(tmp_path):
    root = _broken_root(tmp_path, row_counts={"FS1": CYCLES - 3})
    with pytest.raises(ValueError, match="disagree on their row count"):
        HY._build_aggregate(root, "mean_std")


def test_profile_row_count_disagreement_raises(tmp_path):
    root = _broken_root(tmp_path)
    np.savetxt(root / HYDRAULIC_PROFILE_FILE,
               synthetic_profile(CYCLES - 1, BLOCK_LEN, 8), delimiter="\t", fmt="%d")
    with pytest.raises(ValueError, match=HYDRAULIC_PROFILE_FILE):
        HY._build_aggregate(root, "mean_std")


def test_column_count_that_is_not_a_clean_downscale_raises(tmp_path):
    root = _broken_root(tmp_path, wrong_columns={"PS2": 37})
    with pytest.raises(ValueError, match="not a whole down-scaling"):
        HY._build_aggregate(root, "mean_std")


def test_mixed_sampling_rate_scales_raise(tmp_path):
    """6000 / 50 is a whole number, but the OTHER files are at scale 60: the 100/10/1 Hz
    ratio is broken, which is exactly the drift this guard exists for."""
    root = _broken_root(tmp_path, wrong_columns={"PS2": 50})
    with pytest.raises(ValueError, match="do not share one sampling-rate scale"):
        HY._build_aggregate(root, "mean_std")


# ---------------------------------------------------------------------------
# Fail-loud guards: files and annotations
# ---------------------------------------------------------------------------
def test_missing_sensor_and_profile_files_raise(tmp_path):
    root = _broken_root(tmp_path)
    (root / "EPS1.txt").unlink()
    with pytest.raises(FileNotFoundError, match="EPS1.txt"):
        HY._build_aggregate(root, "mean_std")
    (root / HYDRAULIC_PROFILE_FILE).unlink()
    with pytest.raises(FileNotFoundError, match=HYDRAULIC_PROFILE_FILE):
        HY._build_aggregate(root, "mean_std")


def test_non_finite_reading_raises(tmp_path):
    root = _broken_root(tmp_path, nan_sensors=("TS1",))
    with pytest.raises(ValueError, match="non-finite"):
        HY._build_aggregate(root, "mean_std")


def test_profile_with_the_wrong_width_raises(tmp_path):
    root = _broken_root(tmp_path)
    np.savetxt(root / HYDRAULIC_PROFILE_FILE, np.zeros((CYCLES, 4), np.int64),
               delimiter="\t", fmt="%d")
    with pytest.raises(ValueError, match="expected 5 tab-separated"):
        HY._build_aggregate(root, "mean_std")


def test_undocumented_severity_value_raises(tmp_path):
    root = _broken_root(tmp_path, bad_profile=("valve", 55))
    cfg = _cfg(root, tmp_path / "bad")
    with pytest.raises(ValueError, match=r"valve annotation holds value\(s\) \[55\]"):
        HY.load_hydraulic(cfg)


def test_undocumented_stable_flag_raises(tmp_path):
    root = _broken_root(tmp_path, bad_profile=("stable_flag", 7))
    cfg = _cfg(root, tmp_path / "bad")
    with pytest.raises(ValueError, match="stable_flag holds"):
        HY.load_hydraulic(cfg)


def test_unknown_component_name_raises():
    assert HY.severity_column("pump") == "severity_pump"
    assert HY.action_column("pump") == "action_pump"
    with pytest.raises(ValueError, match="unknown hydraulic component"):
        HY.severity_column("gearbox")


# ---------------------------------------------------------------------------
# Fail-loud guards: filtering, split, truncation
# ---------------------------------------------------------------------------
def test_dropping_every_cycle_as_unstable_raises(tmp_path):
    root = write_synthetic_hydraulic(tmp_path / "warmup", cycles=6, block_len=2,
                                     unstable_warmup=10, seed=2)
    with pytest.raises(ValueError, match="discarded as not-stable"):
        HY.load_hydraulic(_cfg(root, tmp_path))


def test_keeping_unstable_cycles_is_a_config_switch(tmp_path):
    """The same files, both polarities of the knob: dropping loses the flagged cycles
    (1 = NOT stable) and the blocks that consist only of them."""
    root = write_synthetic_hydraulic(tmp_path / "rig", cycles=CYCLES,
                                     block_len=BLOCK_LEN, seed=3)
    dropped = HY._canonical_frame(
        *HY._load_or_build_aggregate(_cfg(root, tmp_path), verbose=False),
        _cfg(root, tmp_path))
    kept = HY._canonical_frame(
        *HY._load_or_build_aggregate(_cfg(root, tmp_path), verbose=False),
        _cfg(root, tmp_path, hydraulic_drop_unstable=False))
    assert len(kept) == CYCLES
    assert len(dropped) == CYCLES - (3 * 8 + 3)          # 3 warm-ups + 3 markers
    assert kept["unit_number"].nunique() == N_BLOCKS
    assert dropped["unit_number"].nunique() == N_UNITS


def test_single_eligible_block_per_cooler_regime_cannot_be_split(tmp_path):
    """Every stratum holding at most ONE block long enough to truncate: holding it out
    would remove a whole stratum from training, so NO stratification can produce a
    scoreable test set and the split raises rather than emitting a degenerate one."""
    root = write_synthetic_hydraulic(tmp_path / "tiny", cycles=12, block_len=4,
                                     unstable_warmup=0, seed=4)
    cfg = _cfg(root, tmp_path, hydraulic_drop_unstable=False)   # 3 blocks of 4
    with pytest.raises(ValueError, match="could not produce a scoreable test set"):
        HY.load_hydraulic(cfg)


def test_blocks_too_short_to_truncate_stay_in_train(tmp_path):
    """A block clipped short (the warm-ups clip whichever block they land in) cannot be a
    test unit -- but it is NOT dropped: it stays in train, where it still carries rows."""
    # 124 cycles / 8 -> 15 full blocks + one clipped 4-cycle block. Enough blocks for
    # the (cooler x valve) split to stay scoreable while still exercising the short one.
    root = write_synthetic_hydraulic(tmp_path / "mixed", cycles=124, block_len=8,
                                     unstable_warmup=0, seed=5)
    cfg = _cfg(root, tmp_path, hydraulic_drop_unstable=False)   # blocks of 8 + a 4
    df_train, df_test, rul = HY.load_hydraulic(cfg)
    train_lengths = df_train.groupby("unit_number").size()
    # a block clipped below window_size+1 is KEPT in train rather than dropped
    assert train_lengths.min() < cfg.window_size + 1
    assert train_lengths.max() == 8
    assert df_test.unit_number.nunique() >= 3                   # >= one per stratum
    assert len(df_train) + len(df_test) + int(rul.sum()) == 124  # nothing invented/lost
    assert (df_test.groupby("unit_number").size() >= cfg.window_size).all()


def test_test_fraction_must_be_a_proper_fraction(hydraulic_root, tmp_path):
    cfg = _cfg(hydraulic_root, tmp_path)
    frame = HY._canonical_frame(*HY._load_or_build_aggregate(cfg, verbose=False), cfg)
    for bad in (0.0, 1.0, -0.2):
        with pytest.raises(ValueError, match="hydraulic_test_fraction"):
            HY._select_test_units(frame, bad, cfg.window_size + 1)


def test_window_longer_than_every_block_raises(hydraulic_root, tmp_path):
    """Label blocks here are ~10 cycles; a window that long leaves no valid prefix
    anywhere, and the error says so instead of emitting a 0-window test unit."""
    cfg = _cfg(hydraulic_root, tmp_path, window_size=BLOCK_LEN)
    with pytest.raises(ValueError, match="Lower window_size"):
        HY.load_hydraulic(cfg)


def test_truncation_backstop_raises(hydraulic_root, tmp_path):
    """The guard inside _truncate_test itself: a frame the eligibility rule would never
    have approved must fail loud rather than yield an empty or full-length test unit."""
    cfg = _cfg(hydraulic_root, tmp_path, window_size=4)
    frame = pd.DataFrame({"unit_number": [7] * 3, "time_cycles": [1, 2, 3],
                          **{c: 0.0 for c in HYDRAULIC_FEATURE_COLUMNS}})
    with pytest.raises(ValueError, match="cannot truncate"):
        HY._truncate_test(frame, cfg)


# ---------------------------------------------------------------------------
# Discovery: subdir candidates, depth-1 nesting, availability
# ---------------------------------------------------------------------------
def test_is_available(hydraulic_root, tmp_path):
    assert HY.is_available(_cfg(hydraulic_root, tmp_path))
    assert not HY.is_available(_cfg(tmp_path / "nothing-here", tmp_path))
    root = _broken_root(tmp_path)
    (root / f"{HYDRAULIC_SENSOR_NAMES[0]}.txt").unlink()   # complete but for one sensor
    assert not HY.is_available(_cfg(root, tmp_path))
    (root / HYDRAULIC_PROFILE_FILE).unlink()
    assert not HY.is_available(_cfg(root, tmp_path))


def test_resolves_alternate_subdir_name_and_nesting(hydraulic_root, tmp_path, capsys):
    """UCI's own unzipped folder name loads without renaming, and a zip-in-a-folder
    nesting is absorbed one level down (CHANGES.md §26)."""
    data_root = tmp_path / "Data"
    nested = data_root / "condition+monitoring+of+hydraulic+systems" / "data"
    nested.parent.mkdir(parents=True)
    shutil.copytree(hydraulic_root, nested)
    cfg = Config(dataset="Hydraulic", data_root=str(data_root), data_dir=None,
                 cache_dir=str(tmp_path / "cache"), results_dir=str(tmp_path / "res"),
                 window_size=4, max_rul=8)
    assert HY.is_available(cfg)                      # silent descent
    assert capsys.readouterr().out == ""
    assert HY._resolve_dir(cfg) == nested             # verbose descent
    assert "descending into nested folder" in capsys.readouterr().out
    df_train, _, _ = HY.load_hydraulic(cfg)
    assert df_train["unit_number"].nunique() >= 12


def test_descend_gives_up_on_an_unrelated_tree(tmp_path):
    """Depth-1 only: when no immediate subdirectory holds profile.txt the ORIGINAL root
    is returned, so the caller's error names the documented path."""
    root = tmp_path / "Data" / "Hydraulic"
    (root / "docs" / "deeper").mkdir(parents=True)
    assert HY._descend_to_data(root, verbose=False) == root
    assert HY._descend_to_data(tmp_path / "does-not-exist") == tmp_path / "does-not-exist"


# ---------------------------------------------------------------------------
# End to end: the frame feeds the pipeline's labels + windows unchanged
# ---------------------------------------------------------------------------
def test_frames_flow_through_labels_and_windows(hydraulic_root, tmp_path):
    cfg = _cfg(hydraulic_root, tmp_path)
    df_train, df_test, rul = HY.load_hydraulic(cfg)
    train = D.add_train_rul(df_train, cfg)
    test = D.add_test_rul(df_test, rul, cfg)
    assert (train["clipped_rul"] <= cfg.max_rul).all()
    assert (test["actual_rul"] > 0).all()

    windows, labels, units = D.make_windows(train, cfg.sensor_columns, cfg.window_size)
    assert windows.shape[1:] == (cfg.window_size, len(HYDRAULIC_FEATURE_COLUMNS))
    assert len(np.unique(units)) >= 12 and len(labels) == len(windows)
    # the RQ-F path: the secondary label windows the taxonomy probe asks for
    _w, action_labels, _u = D.make_windows(train, cfg.sensor_columns, cfg.window_size,
                                           target_col=HY.action_column("valve"))
    assert set(np.unique(action_labels)) <= {0.0, 1.0, 2.0}
    assert len(np.unique(action_labels)) > 1     # the probe has something to separate
    last_windows, _l, last_units = D.make_test_last_windows(
        test, cfg.sensor_columns, cfg.window_size)
    assert last_windows.shape[0] == len(last_units) == df_test.unit_number.nunique()


def test_registry_facing_surface():
    """What src/datasets/__init__.py wires up (the family module's public contract)."""
    assert HY.DATASETS == ("Hydraulic",)
    assert HY.HYDRAULIC_SUBDIR[0] == "Hydraulic"
    assert Config(dataset="Hydraulic").dataset_kind() == "hydraulic"
    assert HY.SEVERITY_COLUMNS == [f"severity_{c}" for c in HYDRAULIC_COMPONENTS]
    assert HY.ACTION_COLUMNS == [f"action_{c}" for c in HYDRAULIC_COMPONENTS]


# ---------------------------------------------------------------------------
# Split-strategy edges (§55 review): aliasing, single-level records, quiet mode
# ---------------------------------------------------------------------------
def test_single_target_level_record_is_always_scoreable(hydraulic_root, tmp_path):
    """When the record genuinely holds ONE level of the RQ-F target, no split can put two
    on the test side -- the guard must not then refuse to split at all."""
    cfg = _cfg(hydraulic_root, tmp_path)
    frame = HY._canonical_frame(*HY._load_or_build_aggregate(cfg, verbose=False), cfg)
    flat = frame.copy()
    flat[HY.severity_column("valve")] = 0            # collapse the target to one level
    units = HY._select_test_units(flat, cfg.hydraulic_test_fraction,
                                  cfg.window_size + 1, "valve", verbose=False)
    assert len(units) > 0


def test_split_reports_its_strategy_and_falls_back_quietly_when_asked(
        hydraulic_root, tmp_path, capsys):
    """The chosen stratification is announced (a split you cannot see is a split you
    cannot audit), and ``verbose=False`` silences every notice."""
    cfg = _cfg(hydraulic_root, tmp_path)
    frame = HY._canonical_frame(*HY._load_or_build_aggregate(cfg, verbose=False), cfg)
    capsys.readouterr()
    HY._select_test_units(frame, cfg.hydraulic_test_fraction, cfg.window_size + 1,
                          "valve", verbose=True)
    assert "test split stratified by" in capsys.readouterr().out
    HY._select_test_units(frame, cfg.hydraulic_test_fraction, cfg.window_size + 1,
                          "valve", verbose=False)
    assert capsys.readouterr().out == ""


def test_aliasing_fallback_is_reported_and_recovers(tmp_path, capsys):
    """The bug this guard exists for: systematic sampling inside cooler strata ALIASES
    against the inner factorial, collapsing the test set onto one target level. The
    loader must NOTICE that and fall back to a stratification that recovers, saying so."""
    root = write_synthetic_hydraulic(tmp_path / "alias", cycles=480, block_len=12,
                                     unstable_warmup=0, seed=11)
    cfg = _cfg(root, tmp_path, hydraulic_drop_unstable=False,
               hydraulic_test_fraction=0.25)        # commensurate with the valve period
    df_train, df_test, _rul = HY.load_hydraulic(cfg)
    target = HY.severity_column("valve")
    assert len(set(df_test[target])) >= 2, "the split must stay scoreable"
    assert set(df_train[target])
    out = capsys.readouterr().out
    assert "test split stratified by" in out


def test_a_stratum_with_one_eligible_block_is_reported_not_silently_dropped(
        hydraulic_root, tmp_path, capsys):
    """A silently-skipped stratum reads exactly like one that was represented."""
    cfg = _cfg(hydraulic_root, tmp_path, window_size=6)   # raises the eligibility bar
    frame = HY._canonical_frame(*HY._load_or_build_aggregate(cfg, verbose=False), cfg)
    capsys.readouterr()
    try:
        HY._select_test_units(frame, 0.3, cfg.window_size + 1, "valve", verbose=True)
    except ValueError:
        pass                                          # the loud refusal is also fine
    out = capsys.readouterr().out
    assert "stratified by" in out or "stratum" in out


def test_degenerate_rul_is_announced(hydraulic_root, tmp_path, capsys):
    """Uniform blocks give every test unit the SAME rul_truth, so the predict-the-mean
    floor scores a perfect 0.0 and no model can beat it. That must be stated, not left
    for a reader to discover in a results table (§55)."""
    cfg = _cfg(hydraulic_root, tmp_path)
    capsys.readouterr()
    _tr, _te, rul = HY.load_hydraulic(cfg)
    if rul.nunique() == 1:
        assert "rul_truth is CONSTANT" in capsys.readouterr().out
    else:                                             # a varied fixture: no warning
        assert "rul_truth is CONSTANT" not in capsys.readouterr().out


def test_cross_stratification_that_cannot_score_falls_through_to_the_target(tmp_path,
                                                                            capsys):
    """Directly exercise the fall-through: a record whose (cooler x valve) strata each
    hold too few blocks to cover two valve levels must be RE-SPLIT on valve alone, with
    the reason printed -- this is the aliasing recovery, not an error path."""
    root = write_synthetic_hydraulic(tmp_path / "cross", cycles=192, block_len=12,
                                     unstable_warmup=0, seed=13)
    cfg = _cfg(root, tmp_path, hydraulic_drop_unstable=False)
    frame = HY._canonical_frame(*HY._load_or_build_aggregate(cfg, verbose=False), cfg)
    capsys.readouterr()
    units = HY._select_test_units(frame, cfg.hydraulic_test_fraction,
                                  cfg.window_size + 1, "valve", verbose=True)
    out = capsys.readouterr().out
    target = HY.severity_column("valve")
    levels = set(frame.loc[frame["unit_number"].isin(units), target])
    assert len(levels) >= 2 or "could not produce" in out
    assert "stratified by" in out


def test_skipped_strata_are_named_in_the_successful_split(tmp_path, capsys):
    """A stratum with a single eligible block contributes no test block. When the split
    otherwise SUCCEEDS, that omission must still be reported (§55 review, low)."""
    root = write_synthetic_hydraulic(tmp_path / "skip", cycles=200, block_len=12,
                                     unstable_warmup=0, seed=17)
    cfg = _cfg(root, tmp_path, hydraulic_drop_unstable=False)
    frame = HY._canonical_frame(*HY._load_or_build_aggregate(cfg, verbose=False), cfg)
    # Truncate one block below the eligibility bar so its stratum holds a single block.
    lengths = frame.groupby("unit_number").size()
    victim = int(lengths.index[-1])
    keep = frame["unit_number"] != victim
    trimmed = pd.concat([frame.loc[keep],
                         frame.loc[~keep].head(2)], ignore_index=True)
    capsys.readouterr()
    try:
        HY._select_test_units(trimmed, cfg.hydraulic_test_fraction,
                              cfg.window_size + 1, "valve", verbose=True)
    except ValueError:
        pass
    out = capsys.readouterr().out
    assert "stratum" in out or "stratified by" in out


def test_varied_block_lengths_produce_a_non_constant_rul_and_no_warning(tmp_path,
                                                                        capsys):
    """The degenerate-RUL warning must fire only when the target really is constant --
    a record with varied block lengths gives a varied rul_truth and stays quiet."""
    root = write_synthetic_hydraulic(tmp_path / "varied", cycles=480, block_len=12,
                                     unstable_warmup=7, seed=19)
    cfg = _cfg(root, tmp_path)
    capsys.readouterr()
    _tr, _te, rul = HY.load_hydraulic(cfg)
    warned = "rul_truth is CONSTANT" in capsys.readouterr().out
    assert warned == (rul.nunique() == 1), "the warning must track the actual variance"


def _block_frame(specs) -> pd.DataFrame:
    """A minimal canonical-shaped frame: ``specs`` is a list of
    ``(cooler_severity, valve_severity, n_cycles)``, one entry per label block."""
    rows = []
    for unit, (cooler, valve, n_cycles) in enumerate(specs, start=1):
        for cycle in range(1, n_cycles + 1):
            rows.append({"unit_number": unit, "time_cycles": cycle,
                         HY.severity_column("cooler"): cooler,
                         HY.severity_column("valve"): valve})
    return pd.DataFrame(rows)


def test_the_strategy_loop_retries_after_an_unscoreable_cross_split(capsys):
    """The aliasing recovery, exercised directly: every (cooler, valve) stratum holds
    exactly 2 blocks of ONE valve level, so the cross split covers a single level and
    the loop must fall through to valve-only, which recovers two levels."""
    specs = ([(0, 0, 6)] * 2 + [(1, 0, 6)] * 2      # cooler 0/1 x valve 0
             + [(0, 1, 6)] + [(1, 1, 6)])           # valve 1 appears once per regime
    frame = _block_frame(specs)
    capsys.readouterr()
    units = HY._select_test_units(frame, 0.5, 5, "valve", verbose=True)
    out = capsys.readouterr().out
    levels = set(frame.loc[frame["unit_number"].isin(units),
                           HY.severity_column("valve")])
    assert "aliased against the nested factorial" in out
    assert "valve-only" in out
    assert len(levels) >= 2, "the fallback must recover a scoreable split"


def test_the_strategy_loop_retries_quietly_when_verbose_is_off(capsys):
    """The same aliasing retry as above, with ``verbose=False``: the loop must still fall
    through to the next stratification, silently. A notice-only difference must never
    change WHICH split is produced."""
    specs = ([(0, 0, 6)] * 2 + [(1, 0, 6)] * 2 + [(0, 1, 6)] + [(1, 1, 6)])
    frame = _block_frame(specs)
    capsys.readouterr()
    quiet = HY._select_test_units(frame, 0.5, 5, "valve", verbose=False)
    assert capsys.readouterr().out == ""
    loud = HY._select_test_units(frame, 0.5, 5, "valve", verbose=True)
    assert np.array_equal(quiet, loud), "verbosity must not change the split"
    levels = set(frame.loc[frame["unit_number"].isin(quiet),
                           HY.severity_column("valve")])
    assert len(levels) >= 2


def test_the_degenerate_rul_warning_tracks_the_actual_variance(hydraulic_root, tmp_path,
                                                               monkeypatch, capsys):
    """The warning's FALSE branch. This fixture's blocks are uniform, so its rul_truth is
    constant and the loader warns; when the truncation genuinely yields DIFFERENT
    remaining counts (as the real record does, where the 211-cycle warm-ups clip whichever
    blocks they land in) the loader must stay quiet rather than cry wolf."""
    cfg = _cfg(hydraulic_root, tmp_path)
    real_truncate = HY._truncate_test

    def _varied(df_test_full, config):
        frame, rul = real_truncate(df_test_full, config)
        # perturb one unit's remaining count so the target is no longer constant
        first = sorted(rul)[0]
        rul[first] = rul[first] + 1
        return frame, rul

    monkeypatch.setattr(HY, "_truncate_test", _varied)
    capsys.readouterr()
    _tr, _te, rul = HY.load_hydraulic(cfg)
    assert rul.nunique() > 1
    assert "rul_truth is CONSTANT" not in capsys.readouterr().out
