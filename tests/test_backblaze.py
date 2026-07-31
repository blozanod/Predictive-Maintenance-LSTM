"""CPU tests for the Backblaze Drive Stats loader (src/datasets/backblaze.py).

Backblaze is the fleet-scale, REAL, CENSORED dataset of the study, and almost every
guard in the loader exists because the shipped corpus violates a convenient assumption:
the column count drifts across quarters (and new SMART columns are INSERTED, not
appended), most SMART cells are empty, ``capacity_bytes`` carries a -1 "distrust this
row" sentinel, the archives ship with ``__MACOSX`` junk that matches the day-file glob,
and -- the point of the milestone -- a drive that stops appearing is RIGHT-CENSORED,
not failed. Repo invariant §7 says each violation must raise naming BOTH the expected
and the observed value; these tests hold that line, alongside the canonical-frame
contract, the censoring flag, the seeded survivor subsampling and the stratified split.

CPU-only, synthetic fixtures in the real on-disk format, no downloads.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import Config, BACKBLAZE_DEFAULT_SMART, BACKBLAZE_META_COLUMNS
from src import data as D
from src.datasets import backblaze as BB
from tests.synthetic_backblaze import (SYNTHETIC_MODELS, SYNTHETIC_SMART_ATTRS,
                                       write_synthetic_backblaze)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def backblaze_root(tmp_path_factory) -> Path:
    """A 40-day, 2-model archive (4 failed + 10 censored drives, one vanishing, one
    gapped, one too short, one capacity_bytes=-1 row), written once for the module."""
    root = tmp_path_factory.mktemp("backblaze") / "Data"
    write_synthetic_backblaze(root / "Backblaze")
    return root


def _cfg(root: Path, tmp_path: Path, **over) -> Config:
    base = dict(dataset="Backblaze", data_root=str(root), data_dir=None,
                cache_dir=str(tmp_path / "cache"), results_dir=str(tmp_path / "results"),
                window_size=4, max_rul=40, backblaze_min_days=10,
                backblaze_max_survivors_per_model=3, backblaze_test_fraction=0.3)
    base.update(over)
    return Config(**base)


def _tiny(tmp_path: Path, name: str = "Backblaze", **over) -> Path:
    """A 6-day archive under ``tmp_path/<name>`` -- enough for the schema guards, which
    only need one well-formed file. Returns the DATA ROOT (the parent of ``name``)."""
    params = dict(n_days=6, n_survivors=3, junk=False)
    params.update(over)
    write_synthetic_backblaze(tmp_path / name, **params)
    return tmp_path


def _day_files(root: Path) -> list:
    """The real day files of an archive (never the __MACOSX shadow copies)."""
    return sorted(p for p in (root / "Backblaze").rglob("2*.csv")
                  if "__MACOSX" not in p.parts)


def _rows_of(path: Path) -> list:
    """The file's CSV fields, header row first (CRLF collapses on read)."""
    return [line.split(",") for line in path.read_text(encoding="utf-8").splitlines()]


def _write_rows(path: Path, rows: list) -> None:
    path.write_text("\r\n".join(",".join(r) for r in rows) + "\r\n",
                    encoding="utf-8", newline="")


def _data_row(rows: list, serial: str) -> int:
    """Index (into ``rows``) of the first data row belonging to ``serial``."""
    column = rows[0].index("serial_number")
    return next(i for i, row in enumerate(rows) if i and row[column] == serial)


def _edit(path: Path, serial: str, column: str, value: str) -> None:
    """Overwrite one field of ``serial``'s row in one day file."""
    rows = _rows_of(path)
    rows[_data_row(rows, serial)][rows[0].index(column)] = value
    _write_rows(path, rows)


# ---------------------------------------------------------------------------
# Documented schema constants
# ---------------------------------------------------------------------------
def test_backblaze_schema_constants_match_the_published_files():
    # The five columns that have been present, first and in this order in every release.
    assert tuple(BACKBLAZE_META_COLUMNS) == ("date", "serial_number", "model",
                                             "capacity_bytes", "failure")
    # 5 (pre-2023), 8 (Q2 2023 +vault/pod/legacy), 11 (Q3 2023 +datacenter trio).
    assert BB.BACKBLAZE_META_WIDTHS == (5, 8, 11)
    assert BB.FAILURE_VALUE_SET == (0, 1)
    assert BB.DATASETS == ("Backblaze",)
    assert BB.BACKBLAZE_SUBDIR == ("Backblaze", "backblaze")
    # The day files are found RECURSIVELY: the archives nest inconsistently.
    assert BB.DAILY_CSV_GLOB.startswith("**/")
    # ...and the fixture writes every attribute the default channel set asks for, as
    # normalized/raw PAIRS, so the tests never pass against a file the loader rejects.
    attrs = {int(name.split("_")[1]) for name in BACKBLAZE_DEFAULT_SMART}
    assert attrs <= set(SYNTHETIC_SMART_ATTRS)


def test_is_available_sees_day_files_at_any_depth(backblaze_root, tmp_path):
    assert BB.is_available(_cfg(backblaze_root, tmp_path))
    empty = tmp_path / "empty"
    (empty / "Backblaze").mkdir(parents=True)
    assert not BB.is_available(_cfg(empty, tmp_path))
    # A root that does not exist at all is simply unavailable, never an exception.
    assert not BB.is_available(_cfg(tmp_path / "nope", tmp_path))


def test_macosx_and_dotfiles_never_reach_the_parser(backblaze_root, tmp_path):
    """__MACOSX carries a shadow copy of the FIRST day file, whose name matches the day
    glob exactly -- unfiltered it would be read as a second copy of that day."""
    root = backblaze_root / "Backblaze"
    assert (root / "__MACOSX" / "data_Q1_2024" / "2024-01-01.csv").is_file()
    paths = BB._daily_paths(root)
    assert paths and not any("__MACOSX" in p.parts for p in paths)
    assert len(paths) == 40


# ---------------------------------------------------------------------------
# The canonical frame contract
# ---------------------------------------------------------------------------
def test_load_returns_the_canonical_frames(backblaze_root, tmp_path):
    config = _cfg(backblaze_root, tmp_path)
    df_train, df_test, rul_truth = BB.load_backblaze(config)

    expected = (["unit_number", "time_cycles", "setting_1", "setting_2", "setting_3"]
                + list(BACKBLAZE_DEFAULT_SMART) + [D.EVENT_OBSERVED_COLUMN])
    assert list(df_train.columns) == expected
    assert list(df_test.columns) == expected
    # The channel block is exactly what the config resolves for this family.
    assert list(BACKBLAZE_DEFAULT_SMART) == config.default_sensor_columns()

    for frame in (df_train, df_test):
        assert frame["unit_number"].dtype == np.int64
        assert frame["time_cycles"].dtype == np.int64
        assert frame[D.EVENT_OBSERVED_COLUMN].dtype == np.int64
        # 1-based, dense, consecutive cycles within every unit.
        for _, unit in frame.groupby("unit_number"):
            cycles = unit["time_cycles"].to_numpy()
            assert cycles.tolist() == list(range(1, len(unit) + 1))
        # setting_2/3 carry no meaning for a drive; setting_1 is the model index.
        assert (frame["setting_2"] == 0.0).all() and (frame["setting_3"] == 0.0).all()
        assert set(frame["setting_1"].unique()) <= {0.0, 1.0}
        assert frame[list(BACKBLAZE_DEFAULT_SMART)].notna().all().all()

    assert rul_truth.name == "rul_truth"
    assert rul_truth.index.name == "unit_number"
    assert set(rul_truth.index) == set(df_test["unit_number"].unique())
    assert (rul_truth > 0).all()
    # Units never straddle the split.
    assert not set(df_train["unit_number"]) & set(df_test["unit_number"])


def test_setting_1_is_the_model_index_and_matches_the_manifest(backblaze_root, tmp_path):
    config = _cfg(backblaze_root, tmp_path)
    df_train, df_test, _ = BB.load_backblaze(config)
    manifest = BB.drive_manifest(config)
    models = sorted(set(config.backblaze_models))
    frame = pd.concat([df_train, df_test], ignore_index=True)
    seen = frame.groupby("unit_number")["setting_1"].first()
    for _, row in manifest.iterrows():
        assert seen[row["unit_number"]] == models.index(row["model"])


def test_drive_manifest_records_the_kept_fleet(backblaze_root, tmp_path):
    manifest = BB.drive_manifest(_cfg(backblaze_root, tmp_path))
    assert list(manifest.columns) == ["unit_number", "serial_number", "model", "n_days",
                                      D.EVENT_OBSERVED_COLUMN]
    # every failed drive is kept (2 per model), survivors are capped at 3 per model
    per_model = manifest.groupby("model")[D.EVENT_OBSERVED_COLUMN].agg(["sum", "count"])
    assert per_model["sum"].tolist() == [2, 2]
    assert per_model["count"].tolist() == [5, 5]
    assert manifest["serial_number"].is_unique
    # unit ids are the sorted-(serial_number, model) rank + 1, so the manifest ascends
    # in exactly that order and the ids are stable under the scope knobs.
    assert manifest["unit_number"].is_monotonic_increasing
    pairs = list(zip(manifest["serial_number"], manifest["model"]))
    assert pairs == sorted(pairs)


# ---------------------------------------------------------------------------
# Censoring: the whole point of this milestone
# ---------------------------------------------------------------------------
def test_failed_drives_are_observed_and_vanished_drives_are_censored(backblaze_root,
                                                                    tmp_path):
    config = _cfg(backblaze_root, tmp_path, backblaze_max_survivors_per_model=None)
    manifest = BB.drive_manifest(config)
    fates = dict(zip(manifest["serial_number"], manifest[D.EVENT_OBSERVED_COLUMN]))
    # ...F0/F1 end in a terminal failure=1 row
    assert fates["S0F0"] == 1 and fates["S1F1"] == 1
    # ...V0 simply stops appearing halfway through: right-censored, NOT a failure
    assert fates["S0V0"] == 0 and manifest.set_index("serial_number").loc["S0V0",
                                                                         "n_days"] == 20
    # every drive is one or the other, and the fleet is mostly healthy
    assert set(fates.values()) == {0, 1}
    assert sum(fates.values()) < len(fates) / 2


def test_event_observed_is_constant_within_a_drive(backblaze_root, tmp_path):
    df_train, df_test, _ = BB.load_backblaze(_cfg(backblaze_root, tmp_path))
    for frame in (df_train, df_test):
        counts = frame.groupby("unit_number")[D.EVENT_OBSERVED_COLUMN].nunique()
        assert (counts == 1).all()


def test_censored_rows_inside_the_alarm_horizon_are_dropped_not_guessed(backblaze_root,
                                                                       tmp_path):
    """The §54 machinery consuming this loader's flag: a survivor's last rows are
    UNKNOWABLE (the horizon runs past the end of observation) and must not become 0s."""
    config = _cfg(backblaze_root, tmp_path, alarm_horizon=5)
    df_train, df_test, rul_truth = BB.load_backblaze(config)
    labelled = D.add_alarm_label(D.add_train_rul(df_train, config), config)
    censored = labelled[labelled[D.EVENT_OBSERVED_COLUMN] == 0]
    unknowable = censored[censored["actual_rul"] <= 5]
    assert len(unknowable) and labelled.loc[unknowable.index,
                                            D.ALARM_LABEL_COLUMN].isna().all()
    # a censored drive observed for the WHOLE horizon is a genuine negative
    assert (labelled.loc[(labelled[D.EVENT_OBSERVED_COLUMN] == 0)
                         & (labelled["actual_rul"] > 5), D.ALARM_LABEL_COLUMN] == 0).all()
    # a failed drive's last rows are genuine positives
    observed = labelled[(labelled[D.EVENT_OBSERVED_COLUMN] == 1)
                        & (labelled["actual_rul"] <= 5)]
    assert len(observed) and (observed[D.ALARM_LABEL_COLUMN] == 1).all()
    kept = D.drop_unlabeled_rows(labelled, D.ALARM_LABEL_COLUMN)
    assert len(kept) == len(labelled) - len(unknowable)
    # and the test frame labels through the provided rul_truth without a KeyError
    assert D.add_test_rul(df_test, rul_truth, config)["actual_rul"].notna().all()


# ---------------------------------------------------------------------------
# Schema drift: selection is BY NAME, never by position
# ---------------------------------------------------------------------------
def test_metadata_prefix_width_does_not_change_the_frames(tmp_path):
    """5- / 8- / 11-column metadata prefixes are the three shapes the record contains;
    a positional reader would take a different attribute from each."""
    frames = {}
    for width in (5, 8, 11):
        root = _tiny(tmp_path / f"w{width}", meta_width=width, n_days=12)
        path = _day_files(root)[0]
        header, _ = BB._read_header(path)
        assert BB._check_header(path, header, list(BACKBLAZE_DEFAULT_SMART)) == width
        df_train, df_test, _ = BB.load_backblaze(
            _cfg(root, tmp_path / f"c{width}", backblaze_min_days=4))
        frames[width] = pd.concat([df_train, df_test], ignore_index=True)
    pd.testing.assert_frame_equal(frames[5], frames[8])
    pd.testing.assert_frame_equal(frames[5], frames[11])


def test_inserted_smart_columns_do_not_shift_the_channels(tmp_path):
    """New attributes are INSERTED in ascending order in the real files. The same drives
    written with a wider attribute set must yield byte-identical channel values."""
    narrow = _tiny(tmp_path / "narrow", n_days=12)
    wide = _tiny(tmp_path / "wide", n_days=12, meta_width=11,
                 smart_attrs=(1, 2, 5, 9, 10, 187, 188, 190, 193, 194, 197, 198, 240))
    frames = []
    for index, root in enumerate((narrow, wide)):
        config = _cfg(root, tmp_path / f"cache{index}", backblaze_min_days=4)
        df_train, df_test, _ = BB.load_backblaze(config)
        frames.append(pd.concat([df_train, df_test], ignore_index=True)
                      .sort_values(["unit_number", "time_cycles"]).reset_index(drop=True))
    pd.testing.assert_frame_equal(frames[0], frames[1])


def test_channels_follow_the_requested_order_not_the_file_order(tmp_path):
    """Both readers hand back FILE order; the loader re-indexes by name, so asking for
    the channels in a non-ascending order must permute the VALUES too."""
    root = _tiny(tmp_path, n_days=12)
    forward = _cfg(root, tmp_path / "a", backblaze_min_days=4,
                   backblaze_smart_columns=["smart_9_raw", "smart_5_raw"],
                   sensor_columns=None)
    reverse = _cfg(root, tmp_path / "b", backblaze_min_days=4,
                   backblaze_smart_columns=["smart_5_raw", "smart_9_raw"],
                   sensor_columns=None)
    a, _, _ = BB.load_backblaze(forward)
    b, _, _ = BB.load_backblaze(reverse)
    assert list(a.columns)[5:7] == ["smart_9_raw", "smart_5_raw"]
    assert list(b.columns)[5:7] == ["smart_5_raw", "smart_9_raw"]
    # power-on hours (9) are large and rise daily; reallocated sectors (5) are 0 here
    np.testing.assert_array_equal(a["smart_9_raw"].to_numpy(), b["smart_9_raw"].to_numpy())
    assert (a["smart_9_raw"] > a["smart_5_raw"]).all()


def test_model_conditional_smart_columns_become_zero_not_dropped(backblaze_root,
                                                                 tmp_path):
    """The HGST model reports neither 187 nor 188; those cells are EMPTY STRINGS in the
    file. They become 0.0 (documented DECISION) rather than NaN or a missing column."""
    config = _cfg(backblaze_root, tmp_path)
    df_train, df_test, _ = BB.load_backblaze(config)
    frame = pd.concat([df_train, df_test], ignore_index=True)
    hgst = frame[frame["setting_1"] == sorted(SYNTHETIC_MODELS).index(
        "HGST HMS5C4040ALE640")]
    seagate = frame[frame["setting_1"] != sorted(SYNTHETIC_MODELS).index(
        "HGST HMS5C4040ALE640")]
    assert len(hgst) and (hgst["smart_187_raw"] == 0.0).all()
    assert (seagate["smart_187_raw"] > 0).any()      # the ramp before a real failure


def test_huge_raw_counters_survive_the_cache_exactly(tmp_path):
    """smart_1_raw is a vendor-encoded ~1.2e11 counter: float32 storage would round it,
    so the parsed-frame cache keeps float64."""
    root = _tiny(tmp_path, n_days=12)
    config = _cfg(root, tmp_path / "cache", backblaze_min_days=4,
                  backblaze_smart_columns=["smart_1_raw"], sensor_columns=None)
    df_train, _, _ = BB.load_backblaze(config)          # builds the cache
    fresh, _, _ = BB.load_backblaze(config)             # reads it back
    values = fresh["smart_1_raw"].to_numpy()
    assert values.max() > 1e11
    assert (values == values.astype(np.int64)).all()    # no rounding drift
    pd.testing.assert_frame_equal(df_train, fresh)


def test_utf8_bom_in_the_header_is_defended(tmp_path):
    """Some redistributions carry a UTF-8 BOM, and it lands in the FIRST column's name --
    ``date``, without which nothing is selectable. Current pandas and pyarrow both strip
    it, so the loader's defence (keep the canonical name AND the file's own spelling) is
    belt-and-braces; the property it guarantees is that the canonical names are clean and
    a full load works either way."""
    root = _tiny(tmp_path, n_days=12, bom=True)
    path = _day_files(root)[0]
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")     # the file really has one
    header, raw = BB._read_header(path)
    assert header[0] == "date" and raw["date"].lstrip("\ufeff") == "date"
    df_train, _, _ = BB.load_backblaze(_cfg(root, tmp_path / "cache",
                                            backblaze_min_days=4))
    assert len(df_train)


# ---------------------------------------------------------------------------
# Fail-loud: header / schema guards
# ---------------------------------------------------------------------------
def test_renamed_metadata_prefix_raises(tmp_path):
    root = _tiny(tmp_path, n_days=2)
    path = _day_files(root)[0]
    rows = _rows_of(path)
    rows[0][0] = "Date"                         # a fork "tidying" the header
    _write_rows(path, rows)
    with pytest.raises(ValueError, match="metadata prefix is wrong"):
        BB.load_backblaze(_cfg(root, tmp_path / "cache"))


def test_unknown_metadata_width_raises(tmp_path):
    root = _tiny(tmp_path, n_days=2)
    path = _day_files(root)[0]
    rows = _rows_of(path)
    for index, row in enumerate(rows):          # a 6th metadata column appears
        row.insert(5, "region" if index == 0 else "eu-1")
    _write_rows(path, rows)
    with pytest.raises(ValueError, match=r"metadata prefix is 6 column\(s\) wide"):
        BB.load_backblaze(_cfg(root, tmp_path / "cache"))


def test_missing_requested_smart_column_raises(tmp_path):
    root = _tiny(tmp_path, n_days=2, omit_columns=("smart_187_raw",))
    with pytest.raises(ValueError, match="smart_187_raw"):
        BB.load_backblaze(_cfg(root, tmp_path / "cache"))


def test_file_without_any_smart_column_raises(tmp_path):
    """The metadata-width probe falls back to the whole header when no SMART column
    exists; the requested-columns check then names every one of them."""
    root = _tiny(tmp_path, n_days=2, smart_attrs=())
    header, _ = BB._read_header(_day_files(root)[0])
    assert BB._metadata_width(header) == len(header) == 5
    with pytest.raises(ValueError, match="0 SMART column"):
        BB.load_backblaze(_cfg(root, tmp_path / "cache"))


def test_non_numeric_smart_cell_raises(tmp_path):
    # NOT one of pandas' own NA spellings ("n/a", "NULL", ...): those are legitimately
    # read as empty. This is text where a counter should be.
    root = _tiny(tmp_path, n_days=2)
    _edit(_day_files(root)[0], "S0F0", "smart_5_raw", "bad")
    with pytest.raises(ValueError, match="neither a number nor an empty cell"):
        BB.load_backblaze(_cfg(root, tmp_path / "cache", backblaze_min_days=2))


def test_infinite_smart_cell_raises(tmp_path):
    root = _tiny(tmp_path, n_days=2)
    _edit(_day_files(root)[0], "S0F0", "smart_9_raw", "inf")
    with pytest.raises(ValueError, match="non-finite"):
        BB.load_backblaze(_cfg(root, tmp_path / "cache", backblaze_min_days=2))


def test_wrong_date_inside_a_daily_file_raises(tmp_path):
    root = _tiny(tmp_path, n_days=2)
    _edit(_day_files(root)[0], "S0F0", "date", "2023-12-31")
    with pytest.raises(ValueError, match="must carry that day's date"):
        BB.load_backblaze(_cfg(root, tmp_path / "cache"))


def test_failure_outside_zero_one_raises(tmp_path):
    root = _tiny(tmp_path, n_days=2)
    _edit(_day_files(root)[0], "S0F0", "failure", "2")
    with pytest.raises(ValueError, match=r"'failure' must be \[0, 1\]"):
        BB.load_backblaze(_cfg(root, tmp_path / "cache"))


def test_impossible_day_file_name_raises(tmp_path):
    root = _tiny(tmp_path, n_days=2)
    day_dir = _day_files(root)[0].parent
    (day_dir / "2024-13-45.csv").write_text("date\n")
    with pytest.raises(ValueError, match="not a real date"):
        BB.load_backblaze(_cfg(root, tmp_path / "cache"))


def test_non_date_file_matching_the_glob_shape_raises(tmp_path):
    """``????-??-??.csv`` is a SHAPE, not a date: a stray ``abcd-ef-gh.csv`` matches it
    and must be named, not skipped (skipping would hide a whole day of a real archive
    whose name was mangled by a transfer)."""
    root = _tiny(tmp_path, n_days=2)
    (_day_files(root)[0].parent / "abcd-ef-gh.csv").write_text("date\n")
    with pytest.raises(ValueError, match="by SHAPE"):
        BB.load_backblaze(_cfg(root, tmp_path / "cache"))


def test_two_files_covering_the_same_day_raise(tmp_path):
    root = _tiny(tmp_path, n_days=2)
    first = _day_files(root)[0]
    duplicate = first.parent.parent / "data_Q1_2024_again"
    duplicate.mkdir()
    (duplicate / first.name).write_bytes(first.read_bytes())
    with pytest.raises(ValueError, match="two files cover 2024-01-01"):
        BB.load_backblaze(_cfg(root, tmp_path / "cache"))


def test_missing_archive_raises(tmp_path):
    (tmp_path / "Backblaze").mkdir()
    with pytest.raises(FileNotFoundError, match="no Backblaze day files"):
        BB.load_backblaze(_cfg(tmp_path, tmp_path / "cache"))


# ---------------------------------------------------------------------------
# Fail-loud: the failure-flag semantics
# ---------------------------------------------------------------------------
def test_rows_after_a_failure_are_truncated_and_announced(tmp_path, capsys):
    """Real releases keep reporting a handful of drives for a few days AFTER their
    failure=1 row (e.g. ZHZ3N9S2 in the 2024 corpus, §60). The failure day still ends
    the life being modelled: the kept segment is truncated at the failure row -- keeping
    the OBSERVED failure -- and every such drive is announced, never trimmed silently."""
    root = _tiny(tmp_path, n_days=6, n_survivors=4,        # V3 = a full-length survivor
                 bad_capacity=None)                        # (no hole in V3's rows here)
    _edit(_day_files(root)[3], "S0V3", "failure", "1")     # ..."fails" on day 4...
    # ...but its rows for days 5-6 remain: the zombie-tail shape that aborted the run.
    cfg = _cfg(root, tmp_path, backblaze_min_days=2)
    payload = BB._load_or_build_aggregate(*BB._check_scope(cfg), cfg, verbose=False)
    records = BB._drive_records(payload, sorted(set(cfg.backblaze_models)), cfg,
                                verbose=True)
    out = capsys.readouterr().out
    assert "1 drive(s) kept reporting AFTER their failure=1 day" in out
    assert "S0V3" in out and "+2 row(s)" in out
    record = next(r for r in records if r["serial"] == "S0V3")
    assert record["observed"] == 1          # the failure is kept -- only the tail goes
    assert len(record["rows"]) == 4         # days 1-4; the 2 zombie rows are trimmed


def test_second_failure_row_is_part_of_the_zombie_tail(tmp_path, capsys):
    """A tail that itself carries another failure=1 row is the same lagging-report
    artifact: the life ends at the FIRST failure=1 row, and everything after it --
    including the later failure row -- goes with the trimmed tail instead of aborting
    the parse. Here the first failure lands on day 1, so the 1-day life then falls
    under backblaze_min_days and the drive is dropped as too short (still announced)."""
    root = _tiny(tmp_path, n_days=12)                      # long enough to split after the drop
    _edit(_day_files(root)[0], "S0F0", "failure", "1")     # it already fails on day 12
    cfg = _cfg(root, tmp_path, backblaze_min_days=2)
    BB.load_backblaze(cfg)                                 # end-to-end: must not raise
    out = capsys.readouterr().out
    assert "kept reporting AFTER their failure=1 day" in out and "+11 row(s)" in out
    payload = BB._load_or_build_aggregate(*BB._check_scope(cfg), cfg, verbose=False)
    records = BB._drive_records(payload, sorted(set(cfg.backblaze_models)), cfg,
                                verbose=False)
    assert not any(r["serial"] == "S0F0" for r in records)  # 1-day life -> too short


def test_check_drive_contract_still_rejects_multiple_failure_rows():
    """_drive_records' truncation guarantees at most one failure=1 row reaches
    _check_drive, but the contract itself stays fail-loud for any other caller."""
    days = np.arange(4, dtype=np.int64)
    with pytest.raises(ValueError, match="carries 2 failure=1 rows"):
        BB._check_drive("S1", "M", days, np.array([0, 1, 0, 1], np.int64))


def test_a_drive_twice_on_one_day_raises(tmp_path):
    root = _tiny(tmp_path, n_days=6)
    path = _day_files(root)[0]
    rows = _rows_of(path)
    rows.append(list(rows[_data_row(rows, "S0F0")]))       # the same drive, again
    _write_rows(path, rows)
    with pytest.raises(ValueError, match="appears more than once"):
        BB.load_backblaze(_cfg(root, tmp_path / "cache", backblaze_min_days=2))


# ---------------------------------------------------------------------------
# Scope: models, dates, capacity, min_days, survivor cap
# ---------------------------------------------------------------------------
def test_capacity_sentinel_row_is_dropped_whole(backblaze_root, tmp_path):
    """The -1 row is dropped, which punches a one-day hole in that drive's run; a hole
    that small COLLAPSES (<= BACKBLAZE_MAX_GAP_DAYS), so the drive keeps 39 of 40 days."""
    manifest = BB.drive_manifest(
        _cfg(backblaze_root, tmp_path,
             backblaze_max_survivors_per_model=None)).set_index("serial_number")
    assert manifest.loc["S0V3", "n_days"] == 39
    assert manifest.loc["S1V3", "n_days"] == 40         # the same drive without the -1


def test_long_gap_keeps_only_the_final_segment(backblaze_root, tmp_path):
    """S0V1 is absent for 8 days: the pre-gap segment is a different life and is not
    glued on. 40 days - 8 absent - 5 pre-gap = 27."""
    manifest = BB.drive_manifest(
        _cfg(backblaze_root, tmp_path,
             backblaze_max_survivors_per_model=None)).set_index("serial_number")
    assert manifest.loc["S0V1", "n_days"] == 27
    assert BB._last_segment_start(np.array([0, 1, 2, 20, 21])) == 3
    assert BB._last_segment_start(np.array([0, 1, 2, 3])) == 0
    # a gap no larger than the allowance collapses instead of cutting
    assert BB._last_segment_start(np.array([0, 1, 4, 5])) == 0


def test_short_drives_are_dropped_by_min_days(backblaze_root, tmp_path):
    manifest = BB.drive_manifest(_cfg(backblaze_root, tmp_path,
                                      backblaze_max_survivors_per_model=None))
    assert "S0V2" not in set(manifest["serial_number"])      # only 3 observed days
    assert (manifest["n_days"] >= 10).all()


def test_every_drive_too_short_raises(backblaze_root, tmp_path):
    with pytest.raises(ValueError, match="no Backblaze drive reached"):
        BB.load_backblaze(_cfg(backblaze_root, tmp_path, backblaze_min_days=500))


def test_date_bounds_are_inclusive_and_cheap(backblaze_root, tmp_path):
    config = _cfg(backblaze_root, tmp_path, backblaze_start_date="2024-01-05",
                  backblaze_end_date="2024-02-09", backblaze_min_days=5,
                  backblaze_max_survivors_per_model=None)
    df_train, df_test, _ = BB.load_backblaze(config)
    frame = pd.concat([df_train, df_test], ignore_index=True)
    # 36 days in the inclusive window, and the longest-lived drives cover all of them
    assert frame.groupby("unit_number").size().max() == 36
    # the bounds are part of the cache identity, never of the loader's behaviour elsewhere
    other = _cfg(backblaze_root, tmp_path, backblaze_start_date="2024-01-06")
    assert BB._agg_cache_path(sorted(SYNTHETIC_MODELS), list(BACKBLAZE_DEFAULT_SMART),
                              config) != BB._agg_cache_path(
        sorted(SYNTHETIC_MODELS), list(BACKBLAZE_DEFAULT_SMART), other)


def test_an_end_bound_before_a_failure_censors_that_drive(backblaze_root, tmp_path):
    """Administrative censoring, exactly as in the field: cut the record short and a
    drive that fails later is a SURVIVOR as far as this run can know."""
    manifest = BB.drive_manifest(_cfg(backblaze_root, tmp_path,
                                      backblaze_end_date="2024-01-24",
                                      backblaze_min_days=5,
                                      backblaze_max_survivors_per_model=None))
    assert len(manifest) and (manifest[D.EVENT_OBSERVED_COLUMN] == 0).all()


def test_date_bounds_outside_the_record_raise(backblaze_root, tmp_path):
    with pytest.raises(ValueError, match="no Backblaze day file falls inside"):
        BB.load_backblaze(_cfg(backblaze_root, tmp_path,
                               backblaze_start_date="2030-01-01"))


def test_model_scope_restricts_the_fleet(backblaze_root, tmp_path):
    config = _cfg(backblaze_root, tmp_path, backblaze_models=["ST12000NM0008"])
    manifest = BB.drive_manifest(config)
    assert set(manifest["model"]) == {"ST12000NM0008"}
    # and it is the ONLY model index left, so setting_1 is 0 everywhere
    df_train, _, _ = BB.load_backblaze(config)
    assert (df_train["setting_1"] == 0.0).all()


def test_a_model_that_never_appears_raises(backblaze_root, tmp_path):
    with pytest.raises(ValueError, match="never appear in the"):
        BB.load_backblaze(_cfg(backblaze_root, tmp_path,
                               backblaze_models=["ST12000NM0008", "ST12000NM0O08"]))


def test_days_with_no_in_scope_drive_are_skipped(tmp_path):
    """A quarter in which none of the scoped models was deployed is skipped, not fatal
    -- the fleet turns over, and the scoped models still appear in the other files."""
    root = tmp_path / "root"
    write_synthetic_backblaze(root / "Backblaze", n_days=12, n_survivors=3, junk=False)
    write_synthetic_backblaze(root / "Backblaze", start="2024-03-01", n_days=4,
                              n_survivors=3, junk=False, subdir="data_Q2_2024",
                              models=("TOSHIBA MG07ACA14TA",))
    config = _cfg(root, tmp_path / "cache", backblaze_min_days=4)
    manifest = BB.drive_manifest(config)
    assert set(manifest["model"]) == set(SYNTHETIC_MODELS)
    assert manifest["n_days"].max() == 12


def test_a_serial_reused_by_another_model_is_a_different_drive(tmp_path):
    """Serials are not globally unique forever, so a drive is the PAIR (serial, model).
    Keyed on the serial alone these two would fuse into one drive whose failure=1 row
    sits in the middle of the run -- which is exactly what the failure guard rejects."""
    root = tmp_path / "root"
    write_synthetic_backblaze(root / "Backblaze", n_days=12, n_survivors=3, junk=False,
                              models=("MODEL_A",), subdir="q1")
    write_synthetic_backblaze(root / "Backblaze", start="2024-02-01", n_days=12,
                              n_survivors=3, junk=False, models=("MODEL_B",), subdir="q2")
    manifest = BB.drive_manifest(_cfg(root, tmp_path / "cache", backblaze_min_days=4,
                                      backblaze_models=["MODEL_A", "MODEL_B"],
                                      backblaze_max_survivors_per_model=None))
    reused = manifest[manifest["serial_number"] == "S0F0"]
    assert len(reused) == 2 and set(reused["model"]) == {"MODEL_A", "MODEL_B"}
    assert reused["unit_number"].nunique() == 2
    assert (reused["n_days"] == 12).all()          # neither run absorbed the other


def test_an_archive_of_only_unreliable_rows_raises(tmp_path):
    root = _tiny(tmp_path, n_days=6, all_bad_capacity=True)
    with pytest.raises(ValueError, match="no drive-day survived scoping"):
        BB.load_backblaze(_cfg(root, tmp_path / "cache", backblaze_min_days=2))


def test_survivor_subsampling_is_seeded_and_keeps_every_failure(backblaze_root, tmp_path):
    kept = {}
    for seed in (42, 7):
        config = _cfg(backblaze_root, tmp_path / f"s{seed}", seed=seed,
                      backblaze_max_survivors_per_model=2)
        manifest = BB.drive_manifest(config)
        assert manifest.groupby("model")[D.EVENT_OBSERVED_COLUMN].sum().tolist() == [2, 2]
        assert (manifest.groupby("model")[D.EVENT_OBSERVED_COLUMN].count() == 4).all()
        kept[seed] = set(manifest["serial_number"])
        # ...and the same seed is reproducible
        assert set(BB.drive_manifest(config)["serial_number"]) == kept[seed]
    assert kept[42] != kept[7]
    # the per-model stream is derived from (seed, model), so it is stable under a
    # widening of backblaze_models
    assert BB._model_seed(42, "A") != BB._model_seed(42, "B")
    assert BB._model_seed(42, "A") == BB._model_seed(42, "A")


def test_no_survivor_cap_keeps_every_drive(backblaze_root, tmp_path):
    uncapped = BB.drive_manifest(_cfg(backblaze_root, tmp_path,
                                      backblaze_max_survivors_per_model=None))
    assert len(uncapped) == 12                    # 4 failed + 8 long-enough survivors
    # a cap at or above the survivor count draws nothing at all
    generous = BB.drive_manifest(_cfg(backblaze_root, tmp_path / "b",
                                      backblaze_max_survivors_per_model=50))
    assert set(generous["serial_number"]) == set(uncapped["serial_number"])


# ---------------------------------------------------------------------------
# The split
# ---------------------------------------------------------------------------
def test_split_is_stratified_and_holds_real_failures(backblaze_root, tmp_path):
    config = _cfg(backblaze_root, tmp_path)
    df_train, df_test, _ = BB.load_backblaze(config)
    test_fates = df_test.groupby("unit_number")[D.EVENT_OBSERVED_COLUMN].max()
    train_fates = df_train.groupby("unit_number")[D.EVENT_OBSERVED_COLUMN].max()
    assert test_fates.sum() >= 1                  # the point: scoreable at all
    assert (test_fates == 0).any()                # ...and censored negatives too
    assert train_fates.sum() >= 1                 # failures on both sides
    # both models are represented on both sides
    models = pd.concat([df_train, df_test]).groupby("unit_number")["setting_1"].first()
    assert set(models[df_test["unit_number"].unique()]) == {0.0, 1.0}
    assert set(models[df_train["unit_number"].unique()]) == {0.0, 1.0}


def test_split_is_deterministic(backblaze_root, tmp_path):
    a = BB.load_backblaze(_cfg(backblaze_root, tmp_path / "a"))
    b = BB.load_backblaze(_cfg(backblaze_root, tmp_path / "b"))
    for left, right in zip(a[:2], b[:2]):
        pd.testing.assert_frame_equal(left, right)
    pd.testing.assert_series_equal(a[2], b[2])


def test_test_drives_are_truncated_and_rul_is_the_remainder(backblaze_root, tmp_path):
    config = _cfg(backblaze_root, tmp_path)
    _, df_test, rul_truth = BB.load_backblaze(config)
    manifest = BB.drive_manifest(config).set_index("unit_number")
    for unit, kept in df_test.groupby("unit_number").size().items():
        full = manifest.loc[unit, "n_days"]
        assert kept == max(config.window_size, min(int(full * 0.6), full - 1))
        assert rul_truth[unit] == full - kept


def test_an_unreachable_alarm_horizon_is_reported_not_hidden(backblaze_root, tmp_path,
                                                              capsys):
    """The alarm arm scores each test drive at its LAST kept day, where a truncated
    failed drive still has 40% of its life left. A horizon shorter than that produces no
    positive at all -- a scoring-configuration problem the loader names out loud rather
    than leaving as a silent nan."""
    _, _, rul = BB.load_backblaze(_cfg(backblaze_root, tmp_path, alarm_horizon=5))
    assert "NOTICE: no test drive is within alarm_horizon=5" in capsys.readouterr().out
    # a horizon that a failed test drive DOES reach says nothing at all
    BB.load_backblaze(_cfg(backblaze_root, tmp_path, alarm_horizon=int(max(rul)) + 20))
    assert "NOTICE" not in capsys.readouterr().out
    # ...and neither does the pure-RUL configuration (no horizon at all)
    BB.load_backblaze(_cfg(backblaze_root, tmp_path))
    assert "NOTICE" not in capsys.readouterr().out


def test_a_fleet_with_one_failure_per_model_is_rejected(tmp_path):
    """A stratum with a single eligible drive contributes none -- and if that leaves the
    test set without a positive, the split is unscoreable and must say so."""
    root = _tiny(tmp_path, n_days=20, n_failed=1)
    with pytest.raises(ValueError, match="NOT ONE observed failure"):
        BB.load_backblaze(_cfg(root, tmp_path / "cache", backblaze_min_days=5))


def test_a_window_no_drive_can_fill_is_rejected(backblaze_root, tmp_path):
    with pytest.raises(ValueError, match="produced an empty test set"):
        BB.load_backblaze(_cfg(backblaze_root, tmp_path, window_size=500))


def test_truncating_an_ineligible_drive_raises(backblaze_root, tmp_path):
    """Defensive backstop: _select_test_units never hands over a drive this short."""
    config = _cfg(backblaze_root, tmp_path, window_size=5)
    frame = pd.DataFrame({"unit_number": [1, 1], "time_cycles": [1, 2]})
    with pytest.raises(ValueError, match="cannot truncate 2 observed day"):
        BB._truncate_test(frame, config)


# ---------------------------------------------------------------------------
# Scope-knob validation (before any file is opened)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("override, message", [
    (dict(backblaze_models=[]), "backblaze_models is empty"),
    (dict(backblaze_smart_columns=[], sensor_columns=["smart_5_raw"]),
     "backblaze_smart_columns is empty"),
    (dict(sensor_columns=["h_rms"]), "does not emit"),
    (dict(backblaze_min_days=0), "backblaze_min_days must be"),
    (dict(backblaze_max_survivors_per_model=0), "backblaze_max_survivors_per_model"),
    (dict(backblaze_test_fraction=1.0), "backblaze_test_fraction must be in"),
])
def test_scope_knobs_are_validated(backblaze_root, tmp_path, override, message):
    with pytest.raises(ValueError, match=message):
        BB.load_backblaze(_cfg(backblaze_root, tmp_path, **override))


# ---------------------------------------------------------------------------
# The parsed-frame cache
# ---------------------------------------------------------------------------
def test_cache_is_written_once_and_reused(backblaze_root, tmp_path, capsys):
    config = _cfg(backblaze_root, tmp_path)
    first = BB.load_backblaze(config)
    cached = sorted(Path(config.cache_dir).glob("backblaze_agg_v*.npz"))
    assert len(cached) == 1
    capsys.readouterr()
    second = BB.load_backblaze(config)
    assert "loaded cached table" in capsys.readouterr().out
    pd.testing.assert_frame_equal(first[0], second[0])


def test_cache_name_is_location_independent_but_scope_and_corpus_dependent(
        backblaze_root, tmp_path):
    """Two copies of the SAME corpus share a cache (its identity is day NAMES, never
    paths -- §23), but a different SCOPE or a different CORPUS must not: the cache holds
    a parse of specific day files, and silently serving a stale one after another quarter
    is unzipped is the failure mode this digest exists to prevent."""
    config = _cfg(backblaze_root, tmp_path)
    models, smart = sorted(SYNTHETIC_MODELS), list(BACKBLAZE_DEFAULT_SMART)
    inventory = BB._corpus_inventory(config)
    assert inventory, "the fixture corpus must have in-scope days"
    # same corpus inventory, different location -> same cache name
    elsewhere = _cfg(tmp_path / "another" / "copy", tmp_path)
    assert BB._agg_cache_path(models, smart, config, inventory).name == \
        BB._agg_cache_path(models, smart, elsewhere, inventory).name
    # scope changes re-key
    assert BB._agg_cache_path(models, smart, config, inventory) != \
        BB._agg_cache_path(models[:1], smart, config, inventory)
    assert BB._agg_cache_path(models, smart, config, inventory) != \
        BB._agg_cache_path(models, smart[:2], config, inventory)
    # ... and so does a GROWN corpus (another quarter unzipped into the same folder)
    grown = inventory + ["2099-01-01"]
    assert BB._agg_cache_path(models, smart, config, inventory) != \
        BB._agg_cache_path(models, smart, config, grown)


def test_a_cache_holding_other_channels_raises(backblaze_root, tmp_path):
    config = _cfg(backblaze_root, tmp_path)
    models, smart = sorted(SYNTHETIC_MODELS), list(BACKBLAZE_DEFAULT_SMART)
    path = BB._agg_cache_path(models, smart, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, meta=np.zeros((1, 3), np.int64),
             presence=np.zeros((1, 2), np.int64), smart=np.zeros((1, 1)),
             serials=np.array(["x"], dtype=np.str_),
             models=np.array(["y"], dtype=np.str_),
             columns=np.array(["smart_1_raw"], dtype=np.str_))
    with pytest.raises(ValueError, match="holds channels"):
        BB.load_backblaze(config)


# ---------------------------------------------------------------------------
# The optional pyarrow reader
# ---------------------------------------------------------------------------
class _StubArrowTable:
    def __init__(self, frame):
        self._frame = frame

    def to_pandas(self):
        return self._frame


class _StubArrowCsv:
    """Stand-in for ``pyarrow.csv``: the three names the loader touches, with the same
    call shape. Columns come back as STRINGS (pyarrow's typing of a column it cannot
    infer), which also exercises the object-dtype path of ``_coerce_numeric``."""

    def __init__(self):
        self.calls = []

    @staticmethod
    def ReadOptions(**kwargs):
        return dict(kwargs)

    @staticmethod
    def ConvertOptions(**kwargs):
        return dict(kwargs)

    def read_csv(self, path, read_options, convert_options):
        self.calls.append((path, read_options, convert_options))
        frame = pd.read_csv(path, usecols=convert_options["include_columns"],
                            encoding="utf-8", dtype=str)
        return _StubArrowTable(frame)


def test_pyarrow_is_optional_and_the_seam_reports_its_absence(monkeypatch):
    """pyarrow is an accelerator for a multi-GB parse, never a requirement: with it
    absent the seam reports None and the pandas reader takes over. Forced, so the test
    means the same thing in an environment that has pyarrow and one that does not."""
    monkeypatch.setattr(BB.importlib.util, "find_spec", lambda name: None)
    assert BB._csv_engine() is None


def test_both_csv_readers_produce_the_same_frames(tmp_path, monkeypatch):
    """The two readers are interchangeable by construction -- same by-name selection,
    same validation -- so an environment with pyarrow and one without must not disagree
    about a single reading. Both are FORCED here rather than left to the environment."""
    root = _tiny(tmp_path, n_days=12)
    monkeypatch.setattr(BB, "_csv_engine", lambda: None)
    with_pandas, _, _ = BB.load_backblaze(_cfg(root, tmp_path / "pandas",
                                               backblaze_min_days=4))
    engine = _StubArrowCsv()
    monkeypatch.setattr(BB, "_csv_engine", lambda: engine)
    with_arrow, _, _ = BB.load_backblaze(_cfg(root, tmp_path / "arrow",
                                              backblaze_min_days=4))
    assert engine.calls and engine.calls[0][1] == {"use_threads": True}
    assert engine.calls[0][2]["strings_can_be_null"] is True
    pd.testing.assert_frame_equal(with_pandas, with_arrow)


def test_the_arrow_shaped_reader_also_fails_loud_on_junk(tmp_path, monkeypatch):
    """A reader that hands back STRING columns (what pyarrow does with a column it
    cannot type) must reach the same verdict as the numeric one."""
    root = _tiny(tmp_path, n_days=2)
    _edit(_day_files(root)[0], "S0F0", "smart_5_raw", "bad")
    monkeypatch.setattr(BB, "_csv_engine", lambda: _StubArrowCsv())
    with pytest.raises(ValueError, match="neither a number nor an empty cell"):
        BB.load_backblaze(_cfg(root, tmp_path / "cache", backblaze_min_days=2))


# ---------------------------------------------------------------------------
# §56 review regressions: presence-based segmentation, per-segment semantics,
# duplicate channels, atomic cache
# ---------------------------------------------------------------------------
def test_a_self_inflicted_capacity_hole_does_not_discard_the_drive_history(tmp_path):
    """A run of ``capacity_bytes = -1`` rows is a hole the LOADER punches (Backblaze's
    own guidance is that such a row is unreliable). Segmenting on the surviving days
    would make that hole indistinguishable from the drive leaving the fleet and silently
    throw away everything before it -- so segmentation runs on PRESENCE instead."""
    serial = "S0V3"
    root = tmp_path / "hole"
    # a 4-day unusable stretch, longer than BACKBLAZE_MAX_GAP_DAYS
    holes = [(serial, offset) for offset in (6, 7, 8, 9)]
    write_synthetic_backblaze(root / "Backblaze", n_days=30, n_survivors=4, junk=False,
                              bad_capacity=holes)
    cfg = _cfg(tmp_path / "hole", tmp_path, backblaze_min_days=8,
               backblaze_max_survivors_per_model=None)
    df_train, df_test, _rul = BB.load_backblaze(cfg)
    both = pd.concat([df_train, df_test])
    kept = both[both["unit_number"].isin(
        both["unit_number"].unique())]
    assert len(kept) > 0
    # the drive's pre-hole days survive: its run is longer than the post-hole tail alone
    payload = BB._load_or_build_aggregate(*BB._check_scope(cfg), cfg, verbose=False)
    serials = [str(x) for x in payload["serials"]]
    assert serial in serials, "the holed drive must still be in the parsed table"
    code = serials.index(serial)
    meta = np.asarray(payload["meta"], np.int64)
    usable_days = np.sort(meta[meta[:, 0] == code, 1])
    records = BB._drive_records(payload, sorted(set(cfg.backblaze_models)), cfg,
                                verbose=False)
    record = next((r for r in records if r["serial"] == serial), None)
    assert record is not None, "the holed drive must survive the per-drive rules"
    # every usable day is kept -- the hole cost only the unreadable rows themselves
    assert len(record["rows"]) == len(usable_days)


def test_failure_semantics_are_checked_on_the_segment_that_is_kept(tmp_path):
    """A reused serial's EARLIER life may legitimately contain its own terminal failure.
    Validating the semantics over the whole history would hard-abort the very
    serial-reuse case the gap rule exists to support, so the check runs on the kept
    segment only."""
    days = np.array([1, 2, 3, 40, 41, 42], np.int64)      # a >3-day gap: two lives
    failure = np.array([0, 0, 1, 0, 0, 0], np.int64)      # the FIRST life failed
    offset = BB._last_segment_start(days)
    assert offset == 3
    # the kept segment is well-formed ...
    BB._check_drive("S1", "M", days[offset:], failure[offset:])
    # ... while the full history is not (a non-terminal failure=1)
    with pytest.raises(ValueError):
        BB._check_drive("S1", "M", days, failure)


def test_duplicate_or_metadata_smart_columns_fail_loud(backblaze_root, tmp_path):
    """A repeated channel would otherwise die inside the reader with a pandas-internal
    'Length mismatch' naming neither the knob nor the duplicate. It is NOT de-duplicated
    silently: the emitted channel ORDER is the config's contract."""
    repeated = list(BACKBLAZE_DEFAULT_SMART) + [BACKBLAZE_DEFAULT_SMART[0]]
    cfg = _cfg(backblaze_root, tmp_path, backblaze_smart_columns=repeated,
               sensor_columns=None)
    with pytest.raises(ValueError, match="repeats channel"):
        BB._check_scope(cfg)
    meta = _cfg(backblaze_root, tmp_path,
                backblaze_smart_columns=["smart_5_raw", "failure"], sensor_columns=None)
    with pytest.raises(ValueError, match="metadata column"):
        BB._check_scope(meta)


def test_an_interrupted_cache_write_is_reported_not_re_raised_as_badzipfile(
        backblaze_root, tmp_path):
    """A truncated .npz at the FINAL cache name is what a killed parse used to leave.
    Writing atomically prevents it; the loader still names the file and the remedy if one
    is found (e.g. written by an older build)."""
    cfg = _cfg(backblaze_root, tmp_path)
    models, smart = BB._check_scope(cfg)
    path = BB._agg_cache_path(models, smart, cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a zip file at all")
    with pytest.raises(ValueError, match="unreadable"):
        BB._load_or_build_aggregate(models, smart, cfg, verbose=False)
    # ... and a real parse leaves no .tmp.npz behind
    path.unlink()
    BB._load_or_build_aggregate(models, smart, cfg, verbose=False)
    assert not list(Path(cfg.cache_dir).glob("*.tmp.npz"))


def test_corpus_inventory_of_a_missing_directory_is_empty(tmp_path):
    """The cache digest is computed before anything is opened, so it must survive a
    data_root that is not there yet (the campaign probes availability that way)."""
    cfg = _cfg(tmp_path / "nothing_here", tmp_path)
    assert BB._corpus_inventory(cfg) == []
    # ... and the digest is still well-defined, so a cache path can be named
    models, smart = BB._check_scope(cfg)
    assert BB._agg_cache_path(models, smart, cfg).name.startswith("backblaze_agg_v")


def test_drive_records_tolerates_an_empty_presence_table(backblaze_root, tmp_path):
    """A payload written by an older build (or one where every scoped row was usable and
    none was dropped) carries an EMPTY presence array; segmentation must then fall back
    to the usable days rather than crash on an empty lexsort."""
    cfg = _cfg(backblaze_root, tmp_path)
    models, smart = BB._check_scope(cfg)
    payload = dict(BB._load_or_build_aggregate(models, smart, cfg, verbose=False))
    payload["presence"] = np.zeros((0, 2), np.int64)
    records = BB._drive_records(payload, models, cfg, verbose=False)
    assert records, "an empty presence table must still yield drive records"
    assert all(len(r["rows"]) > 0 for r in records)
