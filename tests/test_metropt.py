"""CPU tests for the MetroPT-3 loader (src/datasets/metropt.py).

MetroPT-3 is a REAL industrial record with no label column, an irregular ~10 s cadence,
~17.6% of wall-clock time invisibly absent, and ground truth supplied out-of-band as
four failure reports. Almost every guard in the loader exists because a real file (or a
fork of it) can violate an assumption silently: a renamed/"corrected" header, a
reformatted timestamp, a digital channel that stopped being binary, a sparsely covered
bin that looks like a full one, a censored run masquerading as a failure. Repo invariant
§7 says each of those must raise naming BOTH the expected and the observed value -- these
tests hold that line, alongside the canonical-frame contract and the run segmentation.

CPU-only, synthetic fixtures in the real on-disk format, no downloads.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import Config, METROPT_FAILURE_EVENTS, METROPT_FEATURE_COLUMNS
from src import data as D
from src.datasets import metropt as MP
from tests.synthetic_metropt import (SYNTHETIC_METROPT_EVENTS,
                                     write_synthetic_metropt)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def metropt_root(tmp_path_factory) -> Path:
    """A 3-day, 3-event synthetic record (4 runs: 3 observed + 1 censored tail) with
    one invisible hole, written once for the whole module."""
    root = tmp_path_factory.mktemp("metropt") / "Data"
    write_synthetic_metropt(root / "MetroPT-3")
    return root


@pytest.fixture
def synthetic_events(monkeypatch):
    """Point the loader's out-of-band failure table at the fixture's injected events.

    The real table is a CODE constant (``config.METROPT_FAILURE_EVENTS``), so a short
    synthetic window can only be segmented by swapping it -- the same swap a future
    re-reading of the failure reports would make."""
    monkeypatch.setattr(MP, "METROPT_FAILURE_EVENTS", SYNTHETIC_METROPT_EVENTS)
    return SYNTHETIC_METROPT_EVENTS


def _cfg(root: Path, tmp_path: Path, **over) -> Config:
    base = dict(dataset="MetroPT-3", data_root=str(root), data_dir=None,
                cache_dir=str(tmp_path / "cache"), results_dir=str(tmp_path / "results"),
                window_size=6, max_rul=40, metropt_cycle_minutes=30,
                metropt_test_runs=[3], metropt_test_truncation=0.6)
    base.update(over)
    return Config(**base)


def _tiny_csv(tmp_path: Path, **over) -> Path:
    """A 1-hour, event-free, hole-free record (~360 rows) for the schema guards, which
    only need ``_read_raw`` to see a well-formed file."""
    params = dict(end="2020-02-01 01:00:00", events=(), gaps=())
    params.update(over)
    return write_synthetic_metropt(tmp_path / "MetroPT-3", **params)


def _edit_field(csv_path: Path, row: int, column: str, value: str) -> None:
    """Overwrite one field of one DATA row in place (row 0 = the first data row)."""
    lines = csv_path.read_text().splitlines()
    fields = lines[1 + row].split(",")
    fields[list(MP.METROPT_EXPECTED_HEADER).index(column)] = value
    lines[1 + row] = ",".join(fields)
    csv_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Documented schema constants (the file's own spelling wins)
# ---------------------------------------------------------------------------
def test_metropt_schema_constants_match_the_shipped_file():
    assert MP.METROPT_EXPECTED_HEADER == (
        "Unnamed: 0", "timestamp", "TP2", "TP3", "H1", "DV_pressure", "Reservoirs",
        "Oil_temperature", "Motor_current", "COMP", "DV_eletric", "Towers", "MPG",
        "LPS", "Pressure_switch", "Oil_level", "Caudal_impulses")
    # the shipped misspelling is preserved, and the FILE order (not the UCI prose's)
    assert "DV_electric" not in MP.METROPT_EXPECTED_HEADER
    header = list(MP.METROPT_EXPECTED_HEADER)
    assert header.index("Oil_temperature") < header.index("Motor_current")
    # 7 analog x (mean, std) + 8 digital duties, in that exact order
    assert len(METROPT_FEATURE_COLUMNS) == 22
    assert METROPT_FEATURE_COLUMNS[:2] == ["TP2_mean", "TP2_std"]
    assert METROPT_FEATURE_COLUMNS[-1] == "Caudal_impulses_duty"
    assert MP.DATASETS == ("MetroPT-3",)
    assert MP.METROPT_SUBDIR == ("MetroPT-3", "MetroPT")
    assert MP._event_observed_column() == D.EVENT_OBSERVED_COLUMN


def test_metropt_config_wiring():
    cfg = Config(dataset="MetroPT-3")
    assert cfg.dataset_kind() == "metropt"
    assert cfg.is_censored_dataset()
    assert cfg.sensor_columns == list(METROPT_FEATURE_COLUMNS)
    # one APU, one operating point -> condition norm resolves auto-OFF
    assert not cfg.effective_condition_norm()


# ---------------------------------------------------------------------------
# The canonical frame
# ---------------------------------------------------------------------------
def test_metropt_frame_schema_runs_and_censoring(metropt_root, tmp_path,
                                                 synthetic_events):
    cfg = _cfg(metropt_root, tmp_path)
    df_train, df_test, rul = MP.load_metropt(cfg)
    obs = D.EVENT_OBSERVED_COLUMN
    expected = (["unit_number", "time_cycles", "setting_1", "setting_2", "setting_3"]
                + list(METROPT_FEATURE_COLUMNS)
                + [obs, MP.FAULT_TYPE_COLUMN, MP.FAULT_SEVERITY_COLUMN])
    assert list(df_train.columns) == expected == list(df_test.columns)
    assert df_train["unit_number"].dtype == np.int64
    assert df_train["time_cycles"].dtype == np.int64

    # 3 events -> 4 runs; run 3 held out, runs 1/2/4 train, ids never shared
    assert sorted(df_train["unit_number"].unique()) == [1, 2, 4]
    assert sorted(df_test["unit_number"].unique()) == [3]
    assert not set(df_train.unit_number) & set(df_test.unit_number)
    # a single APU: no operating-point concept
    assert (df_train[["setting_1", "setting_2", "setting_3"]] == 0.0).all().all()
    # 1-based consecutive cycles per run (gaps collapse; see the module DECISION)
    for _, run in df_train.groupby("unit_number"):
        cyc = run.sort_values("time_cycles").time_cycles.to_numpy()
        assert cyc[0] == 1 and np.array_equal(cyc, np.arange(1, len(cyc) + 1))
    # censoring: runs 1-3 end at a documented event, run 4 is the censored tail
    by_run = df_train.groupby("unit_number")[obs].max()
    assert by_run.loc[1] == 1 and by_run.loc[2] == 1 and by_run.loc[4] == 0
    assert (df_test[obs] == 1).all()
    # RQ-F secondary labels come from the event table; a censored run has no fault
    assert set(df_train.loc[df_train.unit_number == 2, MP.FAULT_TYPE_COLUMN]) == {"Air leak"}
    assert set(df_train.loc[df_train.unit_number == 2, MP.FAULT_SEVERITY_COLUMN]) \
        == {"Medium stress"}
    assert set(df_train.loc[df_train.unit_number == 4, MP.FAULT_TYPE_COLUMN]) == {"none"}
    assert set(df_test[MP.FAULT_TYPE_COLUMN]) == {"Oil leak"}   # run 3 -> event 3

    # truncation protocol: a shortened prefix, RUL > 0, still long enough to window
    assert rul.name == "rul_truth" and rul.index.name == "unit_number"
    assert list(rul.index) == [3] and int(rul.loc[3]) > 0
    assert int(df_test["time_cycles"].max()) >= cfg.window_size
    assert np.isfinite(df_train[METROPT_FEATURE_COLUMNS].to_numpy()).all()


def test_metropt_channels_carry_a_degradation_trend(metropt_root, tmp_path,
                                                    synthetic_events):
    """The point of the run segmentation: within a run the analog level and the digital
    duty drift toward the intervention, so there is learnable RUL signal."""
    cfg = _cfg(metropt_root, tmp_path)
    df_train, _, _ = MP.load_metropt(cfg)
    run = df_train[df_train.unit_number == 1].sort_values("time_cycles")
    assert run["TP2_mean"].iloc[-3:].mean() > run["TP2_mean"].iloc[:3].mean()
    assert run["Oil_temperature_mean"].iloc[-3:].mean() > \
        run["Oil_temperature_mean"].iloc[:3].mean()
    assert run["COMP_duty"].iloc[-3:].mean() > run["COMP_duty"].iloc[:3].mean()
    # duty is a FRACTION of the bin, so it can never leave [0, 1]
    duty_cols = [c for c in METROPT_FEATURE_COLUMNS if c.endswith("_duty")]
    assert df_train[duty_cols].to_numpy().min() >= 0.0
    assert df_train[duty_cols].to_numpy().max() <= 1.0


def test_metropt_aggregate_values_match_a_hand_computation(metropt_root, tmp_path,
                                                           synthetic_events):
    """Run 1 / cycle 1 is the [00:00, 00:30) bin of the raw file: check mean (analog),
    sample std (ddof=1, the N-CMAPSS convention) and duty (digital mean) exactly."""
    cfg = _cfg(metropt_root, tmp_path)
    df_train, _, _ = MP.load_metropt(cfg)
    raw = pd.read_csv(metropt_root / "MetroPT-3" / MP.METROPT_CSV_NAMES[0])
    ts = pd.to_datetime(raw["timestamp"], format=MP.METROPT_TIMESTAMP_FORMAT)
    first_bin = raw[(ts >= pd.Timestamp("2020-02-01 00:00:00"))
                    & (ts < pd.Timestamp("2020-02-01 00:30:00"))]
    row = df_train[(df_train.unit_number == 1) & (df_train.time_cycles == 1)].iloc[0]
    assert np.isclose(row.TP2_mean, first_bin["TP2"].mean(), atol=1e-5)
    assert np.isclose(row.TP2_std, first_bin["TP2"].std(ddof=1), atol=1e-5)
    assert np.isclose(row.COMP_duty, first_bin["COMP"].mean(), atol=1e-6)
    assert np.isclose(row.Oil_temperature_mean, first_bin["Oil_temperature"].mean(),
                      atol=1e-4)


def test_metropt_sparse_bins_are_dropped_not_aggregated(metropt_root, tmp_path,
                                                        synthetic_events):
    """The invisible-gap defence: the hole leaves a partially covered bin, which the
    threshold drops. Lowering it must ADD cycles back.

    The COVERAGE rule is switched off here (``metropt_min_bin_coverage=0``) so the test
    isolates the ABSOLUTE floor; the scale-invariant coverage rule has its own test."""
    strict = _cfg(metropt_root, tmp_path, metropt_min_samples_per_cycle=10,
                  metropt_min_bin_coverage=0.0)
    loose = _cfg(metropt_root, tmp_path, metropt_min_samples_per_cycle=1,
                 metropt_min_bin_coverage=0.0)
    n_strict = sum(len(f) for f in MP.load_metropt(strict)[:2])
    n_loose = sum(len(f) for f in MP.load_metropt(loose)[:2])
    assert n_loose > n_strict
    # the two aggregates coexist -- every knob that shapes them is in the FILENAME
    names = {p.name for p in Path(strict.cache_dir).glob("metropt_agg_*.npz")}
    assert len(names) == 2
    assert any("_n10_" in n for n in names) and any("_n1_" in n for n in names)


def test_metropt_coverage_rule_is_scale_invariant(metropt_root, tmp_path,
                                                  synthetic_events):
    """The gap defence must be expressed as a FRACTION of a bin, not an absolute count:
    ``metropt_cycle_minutes`` is this dataset's RQ-G sweep lever, so an absolute floor
    would make the data-quality filter ~140x stricter at 10-minute bins than at
    1440-minute ones and confound the granularity comparison with a coverage gradient."""
    from src.config import METROPT_NOMINAL_CADENCE_S
    # the resolved threshold scales with the bin width at a fixed coverage fraction
    for minutes in (10, 30, 60):
        cfg = _cfg(metropt_root, tmp_path, metropt_cycle_minutes=minutes,
                   metropt_min_samples_per_cycle=1, metropt_min_bin_coverage=0.5)
        expected = minutes * 60.0 / METROPT_NOMINAL_CADENCE_S
        # a bin at exactly the coverage floor survives; one just below does not
        assert int(np.ceil(0.5 * expected)) >= 1
    tight = _cfg(metropt_root, tmp_path, metropt_min_samples_per_cycle=1,
                 metropt_min_bin_coverage=0.95)
    slack = _cfg(metropt_root, tmp_path, metropt_min_samples_per_cycle=1,
                 metropt_min_bin_coverage=0.0)
    n_tight = sum(len(f) for f in MP.load_metropt(tight)[:2])
    n_slack = sum(len(f) for f in MP.load_metropt(slack)[:2])
    assert n_tight < n_slack, "a stricter coverage rule must drop more bins"
    # and the coverage fraction re-keys the aggregate cache
    assert MP._agg_cache_path(tight, SYNTHETIC_METROPT_EVENTS,
                              MP._find_csv(MP._resolve_dir(tight))) != \
        MP._agg_cache_path(slack, SYNTHETIC_METROPT_EVENTS,
                           MP._find_csv(MP._resolve_dir(slack)))


def test_metropt_bin_of_one_row_gets_zero_std(tmp_path):
    """A 1-row bin has no sample std (NaN) -> 0.0, mirroring N-CMAPSS §27. Checked on
    ``_aggregate`` directly so the exact bin membership is unambiguous."""
    stamps = ["2020-02-01 00:00:00", "2020-02-01 00:10:00", "2020-02-01 00:20:00",
              "2020-02-01 01:00:00"]                      # 3 rows in bin 0, 1 in bin 1
    ts = pd.Series(pd.to_datetime(stamps))
    signals = pd.DataFrame({c: [1.0, 3.0, 5.0, 7.0] for c in MP._ANALOG})
    for c in MP._DIGITAL:
        signals[c] = [0.0, 1.0, 1.0, 0.0]
    mat = MP._aggregate(signals, ts, (), cycle_minutes=60, min_samples=1)
    df = MP._frame_from_matrix(mat, ())
    assert list(df["time_cycles"]) == [1, 2]
    assert np.isclose(df.TP2_mean.iloc[0], 3.0) and np.isclose(df.TP2_std.iloc[0], 2.0)
    assert np.isclose(df.COMP_duty.iloc[0], 2 / 3)
    assert np.isclose(df.TP2_mean.iloc[1], 7.0) and df.TP2_std.iloc[1] == 0.0
    # no events at all -> one run, right-censored, no fault labels
    assert (df["unit_number"] == 1).all()
    assert (df[D.EVENT_OBSERVED_COLUMN] == 0).all()
    assert set(df[MP.FAULT_TYPE_COLUMN]) == {MP.NO_FAULT_LABEL}


def test_metropt_run_assignment_semantics():
    """run k = the period ENDING at event k's start; the window itself belongs to no
    run; the tail after the last window is run n+1."""
    events = SYNTHETIC_METROPT_EVENTS[:2]
    starts, ends = MP._validate_events(events)
    ts = pd.Series(pd.to_datetime([
        "2020-02-01 10:00:00",   # before event 1            -> run 1
        "2020-02-01 18:00:00",   # exactly at event 1 start  -> inside
        "2020-02-01 19:00:00",   # inside event 1            -> inside
        "2020-02-01 20:00:00",   # exactly at event 1 end    -> inside
        "2020-02-02 10:00:00",   # between the events        -> run 2
        "2020-02-03 10:00:00",   # after event 2             -> run 3 (censored tail)
    ]))
    run, inside = MP._assign_runs(ts, starts, ends)
    assert list(inside) == [False, True, True, True, False, False]
    assert list(run[~inside]) == [1, 2, 3]


# ---------------------------------------------------------------------------
# Downstream contract (labels + windows), without touching the shared registry
# ---------------------------------------------------------------------------
def test_metropt_frames_feed_the_label_and_window_path(metropt_root, tmp_path,
                                                       synthetic_events):
    """The canonical frames must flow through src/data.py unchanged: RUL labels, the
    censoring-aware alarm target, and windowing."""
    cfg = _cfg(metropt_root, tmp_path, alarm_horizon=10)
    df_train, df_test, rul = MP.load_metropt(cfg)
    tr = D.add_train_rul(df_train, cfg)
    te = D.add_test_rul(df_test, rul, cfg)
    assert (te["actual_rul"] >= int(rul.loc[3])).all()

    labelled = D.add_alarm_label(tr, cfg)
    censored = labelled[labelled.unit_number == 4]
    observed = labelled[labelled.unit_number == 1]
    # censored rows inside the horizon are UNKNOWABLE (NaN), never guessed 0
    assert censored[D.ALARM_LABEL_COLUMN].isna().sum() == cfg.alarm_horizon + 1
    assert observed[D.ALARM_LABEL_COLUMN].notna().all()
    assert observed[D.ALARM_LABEL_COLUMN].sum() == cfg.alarm_horizon + 1
    kept = D.drop_unlabeled_rows(labelled, D.ALARM_LABEL_COLUMN)
    assert len(kept) == len(labelled) - (cfg.alarm_horizon + 1)

    w, y, u = D.make_windows(kept, cfg.sensor_columns, cfg.window_size,
                             target_col=D.ALARM_LABEL_COLUMN)
    assert w.shape[1:] == (cfg.window_size, len(METROPT_FEATURE_COLUMNS))
    assert set(np.unique(u)) == {1, 2, 4} and set(np.unique(y)) <= {0.0, 1.0}


# ---------------------------------------------------------------------------
# Parsed-frame cache
# ---------------------------------------------------------------------------
def test_metropt_aggregate_cache_is_reused_and_versioned(metropt_root, tmp_path,
                                                         synthetic_events, monkeypatch):
    cfg = _cfg(metropt_root, tmp_path)
    MP.load_metropt(cfg)                     # builds the aggregate cache
    monkeypatch.setattr(MP, "_read_raw",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("re-parsed")))
    MP.load_metropt(cfg)                     # served from cache -> no re-parse
    # a different bin width is a different aggregate (knob is in the filename)
    with pytest.raises(AssertionError, match="re-parsed"):
        MP.load_metropt(cfg.replace(metropt_cycle_minutes=15))
    # bumping the aggregate VERSION invalidates the cache the same way
    monkeypatch.setattr(MP, "METROPT_AGG_VERSION", MP.METROPT_AGG_VERSION + 1)
    with pytest.raises(AssertionError, match="re-parsed"):
        MP.load_metropt(cfg)


def test_metropt_aggregate_cache_tracks_the_failure_table(metropt_root, tmp_path,
                                                          synthetic_events):
    """The event table is a code constant, so it cannot ride the config cache key --
    it rides the cache FILENAME instead, and re-reading the failure reports must not
    silently reuse an aggregate segmented by the old ones."""
    cfg = _cfg(metropt_root, tmp_path)
    MP._load_or_build_aggregate(cfg)
    csv_path = MP._find_csv(MP._resolve_dir(cfg))
    moved = tuple(dict(e, start="2020-02-01 19:00:00") if e["event"] == 1 else dict(e)
                  for e in SYNTHETIC_METROPT_EVENTS)
    assert MP._events_digest(moved) != MP._events_digest(SYNTHETIC_METROPT_EVENTS)
    assert MP._agg_cache_path(cfg, moved, csv_path) != MP._agg_cache_path(
        cfg, SYNTHETIC_METROPT_EVENTS, csv_path)


def test_metropt_aggregate_is_quiet_when_verbose_is_off(metropt_root, tmp_path,
                                                        synthetic_events, capsys):
    cfg = _cfg(metropt_root, tmp_path)
    MP._load_or_build_aggregate(cfg, verbose=False)      # build, silently
    MP._load_or_build_aggregate(cfg, verbose=False)      # cache hit, silently
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
def test_metropt_availability_and_missing_file(tmp_path, metropt_root):
    absent = _cfg(tmp_path / "nowhere", tmp_path)
    assert not MP.is_available(absent)
    with pytest.raises(FileNotFoundError, match="no MetroPT-3 CSV"):
        MP.load_metropt(absent)
    # an existing folder with no accepted file name is still "not available"
    (tmp_path / "Data" / "MetroPT-3").mkdir(parents=True)
    (tmp_path / "Data" / "MetroPT-3" / "readme.txt").write_text("hi\n")
    assert not MP.is_available(_cfg(tmp_path / "Data", tmp_path))
    assert MP.is_available(_cfg(metropt_root, tmp_path))


def test_metropt_accepts_the_renamed_forks(tmp_path):
    """Forks rename the shipped file; each accepted spelling resolves on its own."""
    for name in MP.METROPT_CSV_NAMES[1:]:
        root = tmp_path / name.replace(".", "_") / "Data"
        _tiny_csv(root, filename=name)
        cfg = _cfg(root, tmp_path)
        assert MP.is_available(cfg)
        assert MP._find_csv(MP._resolve_dir(cfg)).name == name


def test_metropt_ambiguous_copies_raise(tmp_path):
    root = tmp_path / "Data"
    _tiny_csv(root)
    _tiny_csv(root, filename="MetroPT3.csv")
    with pytest.raises(ValueError, match="ambiguous MetroPT-3 files"):
        MP.load_metropt(_cfg(root, tmp_path))


# ---------------------------------------------------------------------------
# Fail-loud schema guards (§7: name the expected AND the observed)
# ---------------------------------------------------------------------------
def test_metropt_header_drift_raises(tmp_path):
    """The classic "helpful fix": someone corrects the shipped misspelling."""
    csv_path = _tiny_csv(tmp_path)
    lines = csv_path.read_text().splitlines()
    lines[0] = lines[0].replace("DV_eletric", "DV_electric")
    csv_path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="header does not match") as exc:
        MP._read_raw(csv_path)
    assert "DV_eletric" in str(exc.value) and "DV_electric" in str(exc.value)


def test_metropt_reformatted_timestamp_raises(tmp_path):
    csv_path = _tiny_csv(tmp_path)
    _edit_field(csv_path, 2, "timestamp", "01/02/2020 00:00:20")
    with pytest.raises(ValueError, match="do not match the expected format") as exc:
        MP._read_raw(csv_path)
    assert "01/02/2020 00:00:20" in str(exc.value)


def test_metropt_non_numeric_signal_raises(tmp_path):
    csv_path = _tiny_csv(tmp_path)
    _edit_field(csv_path, 1, "TP2", "oops")
    with pytest.raises(ValueError, match="did not parse as numbers") as exc:
        MP._read_raw(csv_path)
    assert "TP2" in str(exc.value)


def test_metropt_missing_signal_value_raises(tmp_path):
    """Absent time is absent ROWS in this file; a NaN CELL means something else broke."""
    csv_path = _tiny_csv(tmp_path)
    _edit_field(csv_path, 3, "Reservoirs", "")
    with pytest.raises(ValueError, match="hold NaNs") as exc:
        MP._read_raw(csv_path)
    assert "Reservoirs" in str(exc.value)


def test_metropt_non_binary_digital_channel_raises(tmp_path):
    csv_path = _tiny_csv(tmp_path)
    _edit_field(csv_path, 4, "COMP", "2.0")
    with pytest.raises(ValueError, match="DIGITAL channels must be valued") as exc:
        MP._read_raw(csv_path)
    assert "COMP" in str(exc.value) and "2.0" in str(exc.value)


def test_metropt_good_file_passes_every_guard(tmp_path):
    signals, ts = MP._read_raw(_tiny_csv(tmp_path))
    assert list(signals.columns) == list(MP.METROPT_SIGNAL_COLUMNS)
    assert MP.METROPT_ROW_INDEX_COLUMN not in signals.columns   # never a feature
    assert ts.dtype == np.dtype("datetime64[ns]") and len(ts) == len(signals)


# ---------------------------------------------------------------------------
# Failure-table guards (the out-of-band ground truth must be well formed)
# ---------------------------------------------------------------------------
def test_metropt_real_failure_table_is_well_formed():
    starts, ends = MP._validate_events(METROPT_FAILURE_EVENTS)
    assert len(starts) == len(ends) == 4
    assert (starts <= ends).all()


def test_metropt_backwards_failure_window_raises():
    bad = (dict(SYNTHETIC_METROPT_EVENTS[0]),
           dict(SYNTHETIC_METROPT_EVENTS[1], start="2020-02-02 20:00:00",
                end="2020-02-02 18:00:00"))
    with pytest.raises(ValueError, match="start > end"):
        MP._validate_events(bad)


def test_metropt_overlapping_failure_windows_raise():
    bad = (dict(SYNTHETIC_METROPT_EVENTS[0], end="2020-02-02 19:00:00"),
           dict(SYNTHETIC_METROPT_EVENTS[1]))
    with pytest.raises(ValueError, match="chronological and disjoint"):
        MP._validate_events(bad)


# ---------------------------------------------------------------------------
# Binning / split guards
# ---------------------------------------------------------------------------
def test_metropt_no_bin_dense_enough_raises(tmp_path):
    root = tmp_path / "Data"
    _tiny_csv(root)
    cfg = _cfg(root, tmp_path, metropt_min_samples_per_cycle=10_000)
    with pytest.raises(ValueError, match="metropt_min_samples_per_cycle=10000") as exc:
        MP.load_metropt(cfg)
    assert "largest" in str(exc.value)


def test_metropt_unknown_test_run_raises(metropt_root, tmp_path, synthetic_events):
    cfg = _cfg(metropt_root, tmp_path, metropt_test_runs=[9])
    with pytest.raises(ValueError, match=r"metropt_test_runs \[9\] do not exist") as exc:
        MP.load_metropt(cfg)
    assert "[1, 2, 3, 4]" in str(exc.value)


def test_metropt_censored_test_run_raises(metropt_root, tmp_path, synthetic_events):
    """Run 4 here is the censored tail: it has no true RUL, so it can never be scored
    under the predict-at-last-observed-cycle protocol."""
    cfg = _cfg(metropt_root, tmp_path, metropt_test_runs=[4])
    with pytest.raises(ValueError, match="RIGHT-CENSORED") as exc:
        MP.load_metropt(cfg)
    assert "[1, 2, 3]" in str(exc.value)          # the observed runs it suggests instead


def test_metropt_empty_split_raises(tmp_path, monkeypatch):
    """Two ways to empty a side of the split: hold out nothing, or hold out everything.
    Holding out everything is only reachable when NO run is censored -- here the record
    stops inside the last failure window, so there is no tail run at all."""
    monkeypatch.setattr(MP, "METROPT_FAILURE_EVENTS", SYNTHETIC_METROPT_EVENTS)
    root = tmp_path / "Data"
    write_synthetic_metropt(root / "MetroPT-3", end="2020-02-03 12:30:00")
    cfg = _cfg(root, tmp_path)
    df_train, _, _ = MP.load_metropt(cfg)
    assert (df_train[D.EVENT_OBSERVED_COLUMN] == 1).all()      # no censored tail run
    with pytest.raises(ValueError, match="empty train or test set"):
        MP.load_metropt(cfg.replace(metropt_test_runs=[]))
    with pytest.raises(ValueError, match="empty train or test set"):
        MP.load_metropt(cfg.replace(metropt_test_runs=[1, 2, 3]))


def test_metropt_test_run_too_short_to_truncate_raises(metropt_root, tmp_path,
                                                       synthetic_events):
    cfg = _cfg(metropt_root, tmp_path, window_size=60)
    with pytest.raises(ValueError, match="cannot truncate"):
        MP.load_metropt(cfg)


# ---------------------------------------------------------------------------
# The REAL four-event table over the REAL date range
# ---------------------------------------------------------------------------
def test_metropt_real_event_table_yields_five_runs(tmp_path):
    """No monkeypatching: the shipped failure table over the shipped date range must
    cut the record into 5 runs (4 observed + the censored tail), with the default
    ``metropt_test_runs=[4]`` -- the last run that ends in an observed event."""
    root = tmp_path / "Data"
    write_synthetic_metropt(root / "MetroPT-3", start="2020-02-01 00:00:00",
                            end="2020-09-01 03:59:50", events=METROPT_FAILURE_EVENTS,
                            step_seconds=600)
    default_runs = Config(dataset="MetroPT-3").metropt_test_runs
    assert default_runs == [4]                   # the config default, not a test knob
    # The fixture's 10-minute cadence puts ~18 rows in a 180-minute bin, not the ~1080
    # the shipped 10 s stream would; the coverage rule is therefore stated against that
    # cadence rather than the shipped one (this test is about run SEGMENTATION).
    cfg = _cfg(root, tmp_path, metropt_cycle_minutes=180, window_size=10,
               metropt_test_runs=default_runs, metropt_min_bin_coverage=0.0)
    df_train, df_test, rul = MP.load_metropt(cfg)
    assert sorted(df_train["unit_number"].unique()) == [1, 2, 3, 5]
    assert sorted(df_test["unit_number"].unique()) == [4]
    obs = df_train.groupby("unit_number")[D.EVENT_OBSERVED_COLUMN].max()
    assert list(obs) == [1, 1, 1, 0]             # only the tail run is censored
    assert int(rul.loc[4]) > 0
    assert set(df_train[MP.FAULT_TYPE_COLUMN]) == {"Air leak", "none"}


# ---------------------------------------------------------------------------
# Regression: censoring must come from the RECORD, not the run index (§54 review)
# ---------------------------------------------------------------------------
def test_censoring_is_derived_from_the_record_not_the_run_index(tmp_path, synthetic_events):
    """A run the record STOPS SHORT OF is right-censored, whatever its index.

    Deriving ``event_observed`` from ``run_id <= len(events)`` alone silently converts
    the genuinely censored tail into an "observed failure" with a FABRICATED
    ``rul_truth`` -- the exact bug §54 exists to prevent -- on any truncated mirror, and
    the moment a new event is appended to the table (which this module's own schema
    errors instruct the user to do). The guard is: did the record actually reach the
    event this run is supposed to end at?
    """
    root = tmp_path / "Data"
    # Three documented events, but the record stops BEFORE the third one occurs.
    events = tuple(dict(e) for e in SYNTHETIC_METROPT_EVENTS)      # 3 events
    write_synthetic_metropt(root / "MetroPT-3", start="2020-02-01 00:00:00",
                            end="2020-02-03 06:00:00", events=events,
                            gaps=(), step_seconds=10, seed=3)
    cfg = _cfg(root, tmp_path, metropt_cycle_minutes=30, window_size=4,
               metropt_test_runs=[1], metropt_min_bin_coverage=0.0)
    df_train, df_test, _rul = MP.load_metropt(cfg)
    both = pd.concat([df_train, df_test])
    observed = both.groupby("unit_number")[D.EVENT_OBSERVED_COLUMN].max()
    # run 3 ends at an event the record never reaches -> censored, not observed
    assert observed.loc[3] == 0, "a run the record stops short of must be CENSORED"
    assert observed.loc[1] == 1 and observed.loc[2] == 1
    # ... and it therefore cannot be selected as a test run
    with pytest.raises(ValueError, match="CENSORED"):
        MP.load_metropt(_cfg(root, tmp_path, metropt_cycle_minutes=30, window_size=4,
                             metropt_test_runs=[3], metropt_min_bin_coverage=0.0))


def test_unreached_events_are_announced(tmp_path, capsys, synthetic_events):
    """Silently reclassifying a run is the thing being guarded against, so the loader
    says which events lie beyond the end of the record."""
    root = tmp_path / "Data"
    write_synthetic_metropt(root / "MetroPT-3", start="2020-02-01 00:00:00",
                            end="2020-02-03 06:00:00",
                            events=SYNTHETIC_METROPT_EVENTS, gaps=(),
                            step_seconds=10, seed=4)
    cfg = _cfg(root, tmp_path, metropt_cycle_minutes=30, window_size=4,
               metropt_test_runs=[1], metropt_min_bin_coverage=0.0)
    MP.load_metropt(cfg)
    out = capsys.readouterr().out
    assert "RIGHT-CENSORED" in out and "[3]" in out


def test_aggregate_cache_distinguishes_two_different_source_files(tmp_path, synthetic_events):
    """Two different CSVs under the same knobs must NOT collide on one aggregate: the
    cache would otherwise serve the first file's readings for the second (§54 review)."""
    root_a, root_b = tmp_path / "A", tmp_path / "B"
    for root, seed in ((root_a, 0), (root_b, 99)):
        write_synthetic_metropt(root / "MetroPT-3", events=SYNTHETIC_METROPT_EVENTS,
                                seed=seed)
    shared_cache = tmp_path / "cache"
    cfg_a = _cfg(root_a, tmp_path, cache_dir=str(shared_cache),
                 metropt_min_bin_coverage=0.0)
    cfg_b = _cfg(root_b, tmp_path, cache_dir=str(shared_cache),
                 metropt_min_bin_coverage=0.0)
    assert (MP._agg_cache_path(cfg_a, SYNTHETIC_METROPT_EVENTS,
                               MP._find_csv(MP._resolve_dir(cfg_a)))
            != MP._agg_cache_path(cfg_b, SYNTHETIC_METROPT_EVENTS,
                                  MP._find_csv(MP._resolve_dir(cfg_b))))
    a = MP.load_metropt(cfg_a)[0]
    b = MP.load_metropt(cfg_b)[0]
    col = METROPT_FEATURE_COLUMNS[0]
    assert not np.allclose(a[col].to_numpy()[:5], b[col].to_numpy()[:5]), \
        "the second load must re-parse its own file, not serve the first's cache"
