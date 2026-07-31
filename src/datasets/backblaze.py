"""Backblaze Drive Stats loader -> canonical C-MAPSS-shaped frames (a CENSORED fleet).

Dataset (Backblaze "Hard Drive Test Data", https://www.backblaze.com/cloud-storage/
resources/hard-drive-test-data): daily SMART snapshots of every drive in Backblaze's
data centers, published as quarterly ZIPs (``data_Q1_2024.zip``; pre-2016 releases are
ANNUAL, ``data_2013.zip``). Each archive holds **one CSV per day**, named
``YYYY-MM-DD.csv``. Unzip them anywhere under ``config.data_root/Backblaze`` -- the
layout inside the archives is neither flat nor consistent (pre-2016 days nest under a
year folder, and ``__MACOSX/`` + ``.DS_Store`` junk rides along), so the day files are
found by a RECURSIVE glob of ``**/????-??-??.csv`` and junk paths are skipped.

Three properties of the raw files shape everything below:

  * **The schema drifts across quarters.** The column count grows over the record
    (2013/2014: 85 columns; Q3 2023+: 197 = 11 metadata + 186 SMART) and new SMART
    columns are **INSERTED in ascending attribute order, not appended**. Every column
    here is therefore selected **BY NAME** from each file's own header; positional
    indexing would silently read a different attribute per quarter. The metadata prefix
    is 5 columns (``date, serial_number, model, capacity_bytes, failure`` -- always
    present, always first), 8 (+ ``vault_id, pod_id, is_legacy_format``, Q2 2023) or 11
    (+ ``datacenter, cluster_id, pod_slot_num``, Q3 2023); an unrecognized width raises.
  * **``failure`` is a terminal marker, not a state.** ``failure == 1`` marks the LAST
    day a drive was operational; at most one such row exists per drive and it is always
    its final row. A drive that simply STOPS APPEARING with its last row ``failure == 0``
    is **RIGHT-CENSORED** (retired, migrated, or the record just ended) -- NOT a failure.
    Conflating the two is the classic Drive Stats analysis bug and telling them apart is
    the whole point of this milestone (RESEARCH_PLAN §4; the machinery is CHANGES.md §54).
    Real releases bend the "always its final row" half: a handful of drives keep
    reporting for a few days AFTER their ``failure == 1`` row (lagging reports -- e.g.
    ``ZHZ3N9S2`` in 2024, 3 rows after its 2024-09-20 failure). The failure day still
    ends the life being modelled, so ``_drive_records`` TRUNCATES the kept segment at
    the first ``failure == 1`` row and ANNOUNCES every such drive (§60) -- one zombie
    tail must not abort a fleet-scale parse, and a corpus where the flag meant a
    persistent STATE would show up loudly as "most failed drives truncated".
  * **SMART availability is model-conditional and mostly empty.** A given model
    populates ~17-22 of the 93 attributes; every other cell is an EMPTY STRING (-> NaN).
    ``smart_187``/``smart_188`` are absent on several models and ``smart_193`` on some,
    which is why the fleet is SCOPED to ``config.backblaze_models`` before anything else:
    within one model the channel set is comparable across drives. SSDs and HDDs share the
    files with no type flag -- the model string is the only discriminator.

Adaptation to the pipeline's canonical (cycle-level) frame -- so everything downstream of
``data.load_prepared`` runs unchanged (same device as XJTU §22 / N-CMAPSS §27 /
MetroPT §54):
  * one "cycle" = one **drive-day**; ``time_cycles`` = 1-based index over the drive's
    OBSERVED days, dense and consecutive;
  * one "unit"  = one **drive**, identified by ``(serial_number, model)`` -- serials are
    not globally unique forever, so the pair is the key. The pairs are sorted and
    enumerated, and ``unit_number`` is that (stable) index + 1;
  * "sensors"   = exactly ``config.backblaze_smart_columns`` (RQ-C: *which* attributes
    you record is the config choice), taken by name, empty -> NaN -> ``0.0``;
  * ``setting_1`` = the drive's model index within ``sorted(config.backblaze_models)``,
    ``setting_2/3`` = 0.0. Model is the only operating-point-like axis a drive has, and
    raw SMART counters are on wildly different scales across vendors, so
    ``condition_norm=True`` normalizes per model. It stays auto-OFF by default
    (Backblaze is in neither ``MULTI_CONDITION_DATASETS`` nor ``XJTU_DATASETS``);
  * ``event_observed`` (``data.EVENT_OBSERVED_COLUMN``) = 1 for a drive whose run ends in
    a ``failure == 1`` row, 0 for a censored survivor.

Scope, cleaning and split protocol (all DECISIONs, uncited -- there is no community
standard split for Drive Stats; each is a Config field unless marked as a constant):
  * ``backblaze_models`` restricts the fleet; ``backblaze_start_date`` /
    ``backblaze_end_date`` bound the days INCLUSIVELY (applied to the file NAME, so
    out-of-range days are never even opened);
  * a row with ``capacity_bytes < 0`` (the ``-1`` sentinel) is dropped whole -- Backblaze's
    own guidance is that such a row is unreliable, not just its capacity;
  * ``backblaze_min_days`` drops drives with too little history to window;
  * ``backblaze_max_survivors_per_model`` subsamples the CENSORED survivors per model,
    seeded from ``config.seed``. Every FAILED drive is always kept: at ~4.2e-5 failures
    per drive-day (~1 in 23,500) an unsubsampled fleet is almost entirely survivors;
  * the test split holds out ``backblaze_test_fraction`` of DRIVES, **stratified by
    (model, event_observed)**, so the test set can never come out all-survivors -- a
    plain random split of a 1-in-23,500 fleet trivially contains zero failures and is
    then unscoreable. A test set with no observed failure raises;
  * every test drive is truncated at ``BACKBLAZE_TEST_TRUNCATION`` of its observed run so
    the predict-at-the-last-observed-cycle protocol applies. ``rul_truth`` is the number
    of observed days that were cut off. **For a FAILED test drive that is its true
    remaining life; for a CENSORED one it is only a LOWER BOUND** (time to the end of
    observation), which is exactly what ``data.add_train_rul`` documents and what
    ``data.add_alarm_label`` consumes -- rows whose alarm label would be unknowable are
    dropped there, never guessed. Run this dataset with ``config.alarm_horizon`` set: its
    RUL numbers are a plumbing quantity, the alarm/lead-time metric is the result.

``max_rul`` is in observed DRIVE-DAYS here. The community-convention 125 reads as
"degradation is not considered observable more than 125 drive-days before failure",
which is defensible for HDDs but is a per-experiment choice, not a constant inherited
from turbofan cycles -- set it deliberately, as for XJTU (§22).

DECISION (uncited) -- **gaps**. Backblaze's collection misses days. A gap of at most
``BACKBLAZE_MAX_GAP_DAYS`` COLLAPSES: ``time_cycles`` counts observed days, so RUL is
measured in observed drive-days, not wall-clock days (the same convention MetroPT uses
for its dropped bins, §54). A LONGER gap means the drive left the fleet and came back
(re-deployed, or a serial reused), so only the FINAL contiguous segment is kept -- the
segment that actually ends in the failure or the censoring event being modelled. The
earlier segment is a different life and is not silently glued onto it.

**Comparability warning:** published Backblaze failure-prediction numbers use wildly
varying protocols -- different model scopes, different SMART subsets, different
prediction horizons, different (usually undocumented) treatment of censored drives, and
frequently a random drive-DAY split that leaks a drive across train and test. Nothing
here is comparable to any of them, and these numbers must never share a table with a
published Drive Stats result (RESEARCH_PLAN role: same-protocol cross-model comparison
for the censoring chapter, RQ-B/RQ-C/RQ-E, scored with the alarm/lead-time metric --
never the NASA score).
"""

from __future__ import annotations

import hashlib
import os
import zipfile
import importlib
import importlib.util
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..config import (Config, INDEX_COLUMNS, SETTING_COLUMNS, BACKBLAZE_DATASETS,
                      BACKBLAZE_META_COLUMNS)
from .base import resolve_data_dir

# Accepted subdirectory name(s) of ``config.data_root`` holding the unzipped day files
# (at any depth). ``Backblaze`` is the documented layout; the lowercase spelling is the
# common rename (§26 style).
BACKBLAZE_SUBDIR = ("Backblaze", "backblaze")

# Dataset names this family serves (the campaign sweeps these).
DATASETS = tuple(BACKBLAZE_DATASETS)

# Bump whenever the PARSING/SCOPING LOGIC changes, so stale parsed-frame caches
# (cache/backblaze_agg_v<N>_<scope>.npz) are invalidated -- the role CACHE_SCHEMA_VERSION
# plays for embeddings. The scope knobs that change the cache's CONTENT (models, SMART
# columns, date bounds) ride the cache FILENAME instead, so each scope is a separate,
# coexisting aggregate.
BACKBLAZE_AGG_VERSION = 1

# One CSV per day, at ANY depth under the root (the archives nest inconsistently).
DAILY_CSV_GLOB = "**/????-??-??.csv"
_DAILY_CSV_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.csv$")

# Directory names inside the shipped archives that never hold data. ``__MACOSX`` holds
# AppleDouble shadows of the real files -- some of which match the day-file glob.
JUNK_DIR_NAMES = ("__MACOSX",)

# A UTF-8 BOM lands in the FIRST column's name; some redistributions carry one.
_BOM = "\ufeff"

# What a SMART column looks like: the metadata prefix is everything before the first one.
_SMART_COLUMN_RE = re.compile(r"^smart_\d+_(normalized|raw)$")

# Metadata columns beyond the always-present five, by the quarter that introduced them.
# Named only in error messages -- the loader selects by name and never reads them.
BACKBLAZE_OPTIONAL_META_COLUMNS = ("vault_id", "pod_id", "is_legacy_format",   # Q2 2023
                                   "datacenter", "cluster_id", "pod_slot_num")  # Q3 2023
# The three metadata prefix WIDTHS the record contains. A fourth width means a metadata
# column was added or removed: that is schema drift a human must look at, not something
# to absorb silently (repo invariant §7).
BACKBLAZE_META_WIDTHS = (5, 8, 11)

# ``failure`` is a flag, not a count: any other value means the column changed meaning.
FAILURE_VALUE_SET = (0, 1)

# DECISION (uncited): the largest collection gap (in days) that COLLAPSES rather than
# splitting a drive's history -- see the module docstring. 3 days absorbs the ordinary
# one- or two-day collection outage (and the hole left by dropping a capacity_bytes=-1
# row) while still cutting a drive that genuinely left the fleet.
BACKBLAZE_MAX_GAP_DAYS = 3

# DECISION (uncited): test drives are truncated at 0.6 of their observed run, mirroring
# ``xjtu_test_truncation`` / ``ncmapss_test_truncation`` (§22, §27) so the pipeline's
# predict-at-the-last-observed-cycle protocol applies here too. A module constant rather
# than a Config field because this dataset is scored on the ALARM metric, not on RUL
# (module docstring); promote it to a keyed Config field before sweeping it.
BACKBLAZE_TEST_TRUNCATION = 0.6

# DECISION (uncited): an unpopulated SMART cell (empty string -> NaN) becomes 0.0. The
# TSFM path cannot ingest NaN (LightGBM could, but then the two arms would see different
# data, which is the one thing the fairness design forbids), and 0 is the neutral value
# for the RAW counters this loader selects -- they all count events SINCE MANUFACTURE, so
# "not reported" and "nothing has happened yet" are the same number. Whole channels go to
# 0.0 for models that do not populate an attribute, which is the honest representation of
# "this fleet cannot record that" (RQ-C).
BACKBLAZE_SMART_FILL = 0.0

def _event_observed_column() -> str:
    """``data.EVENT_OBSERVED_COLUMN``, imported lazily: ``src.data`` imports the dataset
    registry at module load, so a module-level import here would be circular. Resolved
    per call (never cached) so the constant has exactly one home."""
    from ..data import EVENT_OBSERVED_COLUMN
    return EVENT_OBSERVED_COLUMN


# ---------------------------------------------------------------------------
# Scope validation (fail loud BEFORE a multi-GB parse, not after)
# ---------------------------------------------------------------------------
def _check_scope(config: Config) -> tuple[list, list]:
    """Validate the Backblaze scope knobs; return ``(models, smart_columns)``.

    Everything here is cheap and runs before any file is opened, because the alternative
    is discovering a typo'd knob at the end of an hour-long parse. ``backblaze_models`` is
    de-duplicated and SORTED: that order defines the model index emitted as ``setting_1``
    and is the same order ``config._window_key_fields`` hashes."""
    models = sorted(set(config.backblaze_models))
    smart = list(config.backblaze_smart_columns)
    if not models:
        raise ValueError(
            "backblaze_models is empty; the fleet must be scoped to at least one drive "
            "model (SMART availability and failure physics are model-specific, and the "
            "unscoped corpus is tens of GB). See config.BACKBLAZE_DEFAULT_MODELS.")
    if not smart:
        raise ValueError(
            "backblaze_smart_columns is empty; it IS the channel set this dataset feeds "
            "the models (config.default_sensor_columns()). See "
            "config.BACKBLAZE_DEFAULT_SMART.")
    # NOT de-duplicated silently: the emitted channel ORDER is the config's contract, so
    # a repeat is a config error, not something to quietly repair. Left unchecked it dies
    # deep in the reader with a pandas "Length mismatch" that names neither knob nor cause.
    duplicates = sorted({c for c in smart if smart.count(c) > 1})
    if duplicates:
        raise ValueError(
            f"backblaze_smart_columns repeats channel(s) {duplicates}; each channel may "
            f"appear once (its position is the emitted channel order).\n"
            f"  requested: {smart}")
    collisions = sorted(set(smart) & set(BACKBLAZE_META_COLUMNS))
    if collisions:
        raise ValueError(
            f"backblaze_smart_columns names metadata column(s) {collisions}; those are "
            f"identity/label fields, not channels ({list(BACKBLAZE_META_COLUMNS)}).")
    unknown = [c for c in config.sensor_columns if c not in set(smart)]
    if unknown:
        raise ValueError(
            f"config.sensor_columns names channel(s) the Backblaze loader does not emit: "
            f"{unknown}.\n  emitted  : {smart} (backblaze_smart_columns)\n"
            f"  requested: {list(config.sensor_columns)}\n"
            "Set sensor_columns=None to take the dataset default, or add the columns to "
            "backblaze_smart_columns so they are actually read from the files.")
    if config.backblaze_min_days < 1:
        raise ValueError(
            f"backblaze_min_days must be >= 1 observed day, got "
            f"{config.backblaze_min_days}.")
    cap = config.backblaze_max_survivors_per_model
    if cap is not None and cap < 1:
        raise ValueError(
            f"backblaze_max_survivors_per_model must be >= 1 (or None to keep every "
            f"censored survivor), got {cap}.")
    if not 0.0 < config.backblaze_test_fraction < 1.0:
        raise ValueError(
            f"backblaze_test_fraction must be in (0, 1) -- it is the fraction of DRIVES "
            f"held out -- got {config.backblaze_test_fraction}.")
    return models, smart


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
def _resolve_dir(config: Config) -> Path:
    return resolve_data_dir(config, BACKBLAZE_SUBDIR)


def _daily_paths(root: Path) -> list:
    """Every ``YYYY-MM-DD.csv`` under ``root`` at any depth, junk paths excluded, sorted
    by path. Never raises -- ``is_available`` calls it on a folder the user may not have
    created yet."""
    if not root.is_dir():
        return []
    return [path for path in sorted(root.glob(DAILY_CSV_GLOB))
            if not any(part in JUNK_DIR_NAMES for part in path.parts)]


def _file_day(path: Path) -> np.datetime64:
    """The day a file covers, from its NAME (the cheap scope filter: an out-of-range day
    is never opened). The glob guarantees the ``????-??-??`` shape but not that it is a
    real date, so an impossible one raises rather than being skipped."""
    match = _DAILY_CSV_RE.match(path.name)
    if match is None:
        raise ValueError(
            f"{path}: matches the day-file glob {DAILY_CSV_GLOB!r} by SHAPE but its name "
            f"is not YYYY-MM-DD.csv; a Backblaze day file is named for the day it holds.")
    try:
        return np.datetime64(match.group(1), "D")
    except ValueError as exc:
        raise ValueError(
            f"{path}: file name looks like a Backblaze day file but "
            f"{match.group(1)!r} is not a real date ({exc}); expected YYYY-MM-DD.csv."
        ) from None


def _daily_files(root: Path) -> list:
    """``[(day, path), ...]`` in chronological order, validated: two files covering the
    SAME day (an archive unzipped twice, or a quarter that overlaps a year archive) would
    double every drive-day, so it raises instead."""
    files = [(_file_day(path), path) for path in _daily_paths(root)]
    files.sort(key=lambda item: (item[0], str(item[1])))
    for (day, path), (next_day, next_path) in zip(files, files[1:]):
        if day == next_day:
            raise ValueError(
                f"two files cover {str(day)} under {root}:\n  {path}\n  {next_path}\n"
                "Each day must appear exactly once (overlapping archives would double "
                "every drive-day); delete the duplicate copy.")
    return files


def is_available(config: Config) -> bool:
    """Cheap on-disk check: is at least one ``YYYY-MM-DD.csv`` present under the root?
    (The campaign skips unavailable datasets with a notice, CHANGES.md §24.)"""
    return bool(_daily_paths(_resolve_dir(config)))


def _in_scope(files: list, config: Config) -> list:
    """The subset of ``(day, path)`` inside the INCLUSIVE ``backblaze_start_date`` /
    ``backblaze_end_date`` bounds (either may be None = unbounded)."""
    start = config.backblaze_start_date
    end = config.backblaze_end_date
    lo = np.datetime64(start, "D") if start is not None else None
    hi = np.datetime64(end, "D") if end is not None else None
    return [(day, path) for day, path in files
            if (lo is None or day >= lo) and (hi is None or day <= hi)]


# ---------------------------------------------------------------------------
# Reading one daily CSV (BY NAME, always)
# ---------------------------------------------------------------------------
def _read_header(path: Path) -> tuple[list, dict]:
    """``(canonical names, canonical -> as-spelled-in-the-file)`` for one day file.

    Read with ``encoding='utf-8'`` deliberately, NOT ``utf-8-sig``, and keep BOTH
    spellings: whatever a reader does with a leading BOM, the canonical (BOM-stripped)
    name is what every check and every emitted column uses, while the file's own spelling
    is what the reader is handed. Current pandas and pyarrow both strip the BOM
    themselves, so this is a belt-and-braces defence -- but the failure it prevents (a
    by-name selection missing the FIRST column, i.e. ``date``) is silent and total."""
    raw = list(pd.read_csv(path, nrows=0, encoding="utf-8").columns)
    clean = [name.lstrip(_BOM) for name in raw]
    return clean, dict(zip(clean, raw))


def _metadata_width(header: list) -> int:
    """Number of leading METADATA columns = the index of the first SMART column (or the
    whole header when a file carries no SMART column at all)."""
    smart_at = [i for i, name in enumerate(header) if _SMART_COLUMN_RE.match(name)]
    return smart_at[0] if smart_at else len(header)


def _check_header(path: Path, header: list, smart_columns: list) -> int:
    """Validate one file's header against the documented schema; return its metadata
    width. Three checks, each naming BOTH the expected and the observed value (§7):

    1. the five always-present metadata columns are the file's FIRST five, in order;
    2. the metadata prefix is one of the documented widths (5 / 8 / 11) -- a new width
       means a metadata column was added or removed, which a human must look at;
    3. every REQUESTED SMART column exists BY NAME. It may be absent because the quarter
       predates the attribute or because the model does not report it; either way,
       silently substituting zeros would fabricate readings.
    """
    base = list(BACKBLAZE_META_COLUMNS)
    if header[:len(base)] != base:
        raise ValueError(
            f"{path.name}: the Backblaze metadata prefix is wrong.\n"
            f"  expected first {len(base)} columns: {base}\n"
            f"  observed first {len(base)} columns: {header[:len(base)]}\n"
            "These five have been present, first, and in this order in every release; "
            "a file without them is not Drive Stats.")
    width = _metadata_width(header)
    if width not in BACKBLAZE_META_WIDTHS:
        raise ValueError(
            f"{path.name}: metadata prefix is {width} column(s) wide, expected one of "
            f"{list(BACKBLAZE_META_WIDTHS)} "
            f"({list(BACKBLAZE_META_COLUMNS)} plus, in later quarters, "
            f"{list(BACKBLAZE_OPTIONAL_META_COLUMNS)}).\n"
            f"  observed prefix: {header[:width]}\n"
            "A new metadata column changed the schema -- update "
            "BACKBLAZE_META_WIDTHS/BACKBLAZE_OPTIONAL_META_COLUMNS (and bump "
            "BACKBLAZE_AGG_VERSION); do NOT let it pass silently.")
    present = set(header)
    missing = [column for column in smart_columns if column not in present]
    if missing:
        available = [name for name in header if _SMART_COLUMN_RE.match(name)]
        raise ValueError(
            f"{path.name}: requested SMART column(s) {missing} are not in this file's "
            f"header.\n  requested : {list(smart_columns)} (backblaze_smart_columns)\n"
            f"  observed  : {len(available)} SMART column(s), "
            f"e.g. {available[:8]}\n"
            "The header is the authority (new attributes are INSERTED in ascending "
            "order, never appended), so this file simply does not carry them: narrow "
            "backblaze_smart_columns, or narrow the date range to quarters that do.")
    return width


def _csv_engine():
    """``pyarrow.csv`` when pyarrow is installed, else ``None`` (pandas fallback).

    The real corpus is ~10^2 files x ~10^5 rows per quarter and pyarrow's multithreaded
    CSV reader is several times faster than pandas' on exactly this shape, so it is used
    when present -- but it is an optional accelerator, never a hard requirement, and both
    readers go through the same by-name selection and the same validation afterwards."""
    if importlib.util.find_spec("pyarrow") is None:
        return None
    return importlib.import_module("pyarrow.csv")  # pragma: no cover - lazy heavy import


def _read_columns(path: Path, raw_columns: list, names: list) -> pd.DataFrame:
    """Read exactly ``raw_columns`` (spelled as the FILE spells them) from one day file
    and return them as ``names``, in the requested ORDER.

    Both readers return columns in FILE order regardless of the order asked for, so the
    explicit re-index is what actually makes the selection by-name; without it a quarter
    that inserted an attribute would hand back a permuted channel block."""
    engine = _csv_engine()
    if engine is None:
        frame = pd.read_csv(path, usecols=raw_columns, encoding="utf-8")
    else:
        table = engine.read_csv(
            str(path),
            read_options=engine.ReadOptions(use_threads=True),
            convert_options=engine.ConvertOptions(include_columns=list(raw_columns),
                                                  strings_can_be_null=True))
        frame = table.to_pandas()
    frame = frame.loc[:, list(raw_columns)]
    frame.columns = list(names)
    return frame


def _coerce_numeric(frame: pd.DataFrame, columns: list, path: Path) -> None:
    """Coerce ``columns`` to float64 IN PLACE, failing loud on anything that is not a
    number or an empty cell.

    Empty is expected and legal -- most SMART columns are empty for most drives -- and
    becomes NaN. Text that is not a number is NOT: it means the column changed meaning
    (or the file is truncated mid-row), so it raises naming the offenders. Infinities are
    rejected too: a raw SMART counter is a finite integer, and an inf would silently
    poison every normalization statistic downstream. Reader-independent, so the pyarrow
    and pandas paths can never disagree about what a cell meant."""
    for column in columns:
        values = frame[column]
        if not pd.api.types.is_numeric_dtype(values):
            coerced = pd.to_numeric(values, errors="coerce")
            blank = values.isna() | (values.astype(str).str.strip() == "")
            bad = values[coerced.isna() & ~blank]
            if len(bad):
                raise ValueError(
                    f"{path.name}: column {column!r} holds {len(bad)} value(s) that are "
                    f"neither a number nor an empty cell (expected a decimal integer, or "
                    f"'' where the drive does not report the attribute); first "
                    f"offenders: {list(bad.head(3))}.")
            values = coerced
        values = values.astype(np.float64)
        if not bool(np.isfinite(values.to_numpy()[values.notna().to_numpy()]).all()):
            raise ValueError(
                f"{path.name}: column {column!r} holds non-finite value(s); SMART raw "
                f"counters and capacities are finite integers, so an infinity means the "
                f"column is not what it claims to be.")
        frame[column] = values


def _read_day(path: Path, day: np.datetime64, models: list,
              smart_columns: list) -> tuple[pd.DataFrame, int, set, pd.DataFrame]:
    """One day file -> ``(usable rows for the scoped models, rows dropped for bad
    capacity, every model seen in the file, the scoped rows that were DROPPED)``.

    The fourth return value exists so gap segmentation can run on the days a drive was
    PRESENT rather than on the days that survived cleaning (see the note below).

    Order matters. The header check and the date check are about the FILE, so they run on
    everything; the model filter comes next; and the numeric coercion, the capacity
    sentinel and the ``failure`` value set are then checked on the SCOPED rows only. That
    is both cheaper (a day file holds ~10^5 rows of which the scoped models are a few
    percent) and better targeted: a stray value on some unrelated model is not this run's
    problem, and refusing to load because of it would make the fleet scope meaningless.
    """
    header, raw_by_name = _read_header(path)
    _check_header(path, header, smart_columns)
    wanted = list(BACKBLAZE_META_COLUMNS) + list(smart_columns)
    frame = _read_columns(path, [raw_by_name[name] for name in wanted], wanted)
    for column in ("date", "serial_number", "model"):
        # Identity columns are strings, always: a numeric-looking serial that pandas or
        # pyarrow inferred as an int would break the (serial, model) key.
        frame[column] = frame[column].astype(str)

    expected_date = str(day)
    observed_dates = sorted(set(frame["date"].unique()) - {expected_date})
    if observed_dates:
        raise ValueError(
            f"{path.name}: every row of a daily file must carry that day's date.\n"
            f"  expected: {expected_date!r} (from the file name)\n"
            f"  observed: {observed_dates[:5]} (and {len(observed_dates)} distinct "
            f"other value(s))")

    seen_models = set(frame["model"].unique())
    frame = frame.loc[frame["model"].isin(models)].copy()
    _coerce_numeric(frame, ["capacity_bytes", "failure"] + list(smart_columns), path)
    # capacity_bytes == -1 is Backblaze's own "this row is unreliable" sentinel (and a
    # NaN capacity is no better), so the ROW goes, not just the field.
    usable = (frame["capacity_bytes"].to_numpy() >= 0)
    n_bad_capacity = int((~usable).sum())
    # The (serial, model) pairs PRESENT in this file, usable or not. Gap segmentation
    # runs on PRESENCE, not on what survived cleaning: otherwise >= BACKBLAZE_MAX_GAP_DAYS
    # consecutive capacity_bytes<0 rows look exactly like the drive leaving the fleet, and
    # the loader silently discards the drive's entire pre-hole history over a hole it
    # punched itself.
    present = frame.loc[~usable, ["serial_number", "model"]].copy()
    frame = frame.loc[usable]

    failure_values = sorted(set(pd.unique(frame["failure"].to_numpy()))
                            - set(float(v) for v in FAILURE_VALUE_SET))
    if failure_values:
        raise ValueError(
            f"{path.name}: 'failure' must be {list(FAILURE_VALUE_SET)} (1 = the drive's "
            f"LAST operational day), observed extra value(s) {failure_values}.")
    return (frame.drop(columns=["date", "capacity_bytes"]).reset_index(drop=True),
            n_bad_capacity, seen_models, present.reset_index(drop=True))


# ---------------------------------------------------------------------------
# Parsed-frame cache (the multi-GB parse happens ONCE per scope)
# ---------------------------------------------------------------------------
def _corpus_inventory(config: Config) -> list:
    """The in-scope day list, by FILE NAME only -- the corpus's identity for cache
    purposes. Names, not paths, so the cache stays location-independent (§23) while a
    corpus that GREW (another quarter unzipped) or was swapped re-keys instead of
    silently serving the earlier parse. Cheap: it globs names, it opens nothing."""
    root = _resolve_dir(config)
    if not root.is_dir():
        return []
    return [str(day) for day, _path in _in_scope(_daily_files(root), config)]


def _scope_digest(models: list, smart_columns: list, config: Config,
                  inventory: Optional[list] = None) -> str:
    """Short content hash of everything that shapes the cached table: the model scope,
    the channel set, the date bounds AND the in-scope day inventory. Nothing about the
    files' LOCATION is in it, so pointing at another copy of the SAME corpus reuses the
    cache (location-independent, §23) while a different corpus does not."""
    days = _corpus_inventory(config) if inventory is None else list(inventory)
    blob = json.dumps({"models": list(models), "smart": list(smart_columns),
                       "start": config.backblaze_start_date,
                       "end": config.backblaze_end_date,
                       "n_days": len(days),
                       "first_day": days[0] if days else None,
                       "last_day": days[-1] if days else None}, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def _agg_cache_path(models: list, smart_columns: list, config: Config,
                    inventory: Optional[list] = None) -> Path:
    return (Path(config.cache_dir)
            / f"backblaze_agg_v{BACKBLAZE_AGG_VERSION}"
              f"_{_scope_digest(models, smart_columns, config, inventory)}.npz")


def _build_drive_table(models: list, smart_columns: list, config: Config,
                       verbose: bool = True) -> dict:
    """Parse every in-scope day file into ONE table of scoped drive-days.

    Returns the cache payload: ``meta`` (int64 ``[drive index, day index, failure]``),
    ``smart`` (float64, one column per requested channel), and the ``serials``/``models``
    of each drive index. Drives are indexed by SORTING the ``(serial_number, model)``
    pairs and enumerating them, which makes ``unit_number`` a deterministic function of
    the drive identities and nothing else -- in particular it does not shift when
    ``backblaze_min_days`` or the survivor cap changes.

    ``smart`` is float64, NOT the float32 the other families cache: vendor-encoded raw
    counters (smart_1, smart_7) run to ~1e11, far beyond float32's exact-integer range,
    so float32 would silently round real readings.
    """
    root = _resolve_dir(config)
    files = _daily_files(root)
    if not files:
        raise FileNotFoundError(
            f"no Backblaze day files ({DAILY_CSV_GLOB}) under {root}; download the "
            f"quarterly 'Hard Drive Test Data' archives and unzip them there -- any "
            f"nesting is fine (RESEARCH_PLAN §3).")
    scoped = _in_scope(files, config)
    if not scoped:
        raise ValueError(
            f"no Backblaze day file falls inside backblaze_start_date="
            f"{config.backblaze_start_date!r} .. backblaze_end_date="
            f"{config.backblaze_end_date!r}; the {len(files)} file(s) on disk cover "
            f"{str(files[0][0])} .. {str(files[-1][0])}.")
    if verbose:
        print(f"[backblaze] parsing {len(scoped)} daily file(s) "
              f"({str(scoped[0][0])} .. {str(scoped[-1][0])}) for models {models}...")

    frames, days, n_bad_capacity, seen_models = [], [], 0, set()
    presence_frames, presence_days = [], []
    for day, path in scoped:
        frame, bad, models_in_file, dropped = _read_day(path, day, models, smart_columns)
        n_bad_capacity += bad
        seen_models |= models_in_file
        day_int = day.astype("datetime64[D]").astype(np.int64)
        # PRESENCE = usable rows + rows dropped for a bad capacity sentinel. This is what
        # gap segmentation reads, so a self-inflicted hole never looks like an absence.
        for part in (frame[["serial_number", "model"]], dropped):
            if len(part):
                presence_frames.append(part)
                presence_days.append(np.full(len(part), day_int))
        if not len(frame):
            continue                      # a day on which no in-scope drive was running
        frames.append(frame)
        days.append(np.full(len(frame), day_int))
    absent = [model for model in models if model not in seen_models]
    if absent:
        raise ValueError(
            f"backblaze_models {absent} never appear in the {len(scoped)} day file(s) "
            f"read; the model string must match the files EXACTLY. Observed "
            f"{len(seen_models)} model(s), e.g. {sorted(seen_models)[:8]}.")
    if not frames:
        raise ValueError(
            f"no drive-day survived scoping: models {models} over "
            f"{len(scoped)} day file(s) ({str(scoped[0][0])} .. {str(scoped[-1][0])}), "
            f"{n_bad_capacity} row(s) dropped for capacity_bytes < 0.")

    table = pd.concat(frames, ignore_index=True)
    day_index = np.concatenate(days)
    # A drive is the PAIR (serial_number, model) -- serials are not globally unique
    # forever. factorize(sort=True) over the pair index numbers the drives in sorted-pair
    # order, which is what makes unit_number depend on the identities and nothing else.
    codes, identities = pd.factorize(
        pd.MultiIndex.from_arrays([table["serial_number"], table["model"]]), sort=True)
    if verbose:
        print(f"[backblaze] parsed {len(table)} drive-day(s) over {len(identities)} "
              f"drive(s); dropped {n_bad_capacity} row(s) with capacity_bytes < 0")
    # Presence rows, mapped onto the SAME drive codes (a drive present only on days it
    # was unusable never enters `identities`, so it has no usable rows to segment either
    # -- those rows are simply dropped, which is correct).
    presence = pd.concat(presence_frames, ignore_index=True)
    presence_day_index = np.concatenate(presence_days)
    lookup = {pair: index for index, pair in enumerate(identities)}
    presence_codes = np.array(
        [lookup.get((serial, model), -1) for serial, model
         in zip(presence["serial_number"], presence["model"])], dtype=np.int64)
    known = presence_codes >= 0
    return {
        "meta": np.column_stack([codes.astype(np.int64), day_index,
                                 table["failure"].to_numpy(np.int64)]),
        "presence": np.column_stack([presence_codes[known],
                                     presence_day_index[known]]).astype(np.int64),
        "smart": table[list(smart_columns)].to_numpy(np.float64),
        "serials": np.array([serial for serial, _ in identities], dtype=np.str_),
        "models": np.array([model for _, model in identities], dtype=np.str_),
        "columns": np.array(list(smart_columns), dtype=np.str_),
    }


def _load_or_build_aggregate(models: list, smart_columns: list, config: Config,
                             verbose: bool = True) -> dict:
    """Return the parsed drive-day table for this scope, through a versioned cache.

    Parsing the real archives is minutes to hours; the scoped table is orders of magnitude
    smaller. The cache holds the table BEFORE the per-drive rules (gap segmentation,
    ``backblaze_min_days``, survivor subsampling) and before the split, so those knobs
    re-apply from config without re-parsing -- the same contract the hydraulic cache has.
    It does NOT hold the whole corpus: the model scope and the date bounds are in the
    cache filename, so widening either is a re-parse by construction."""
    cache_path = _agg_cache_path(models, smart_columns, config)
    if not cache_path.exists():
        payload = _build_drive_table(models, smart_columns, config, verbose=verbose)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic: an interrupted multi-hour parse must not leave a truncated .npz at the
        # FINAL name, which every later run would then fail to open with a bare
        # BadZipFile naming neither the file nor the remedy. (The name must still end in
        # .npz -- np.savez appends the extension when it is absent.)
        tmp = cache_path.with_name(cache_path.name + ".tmp.npz")
        np.savez(tmp, **payload)
        os.replace(tmp, cache_path)
        if verbose:
            print(f"[backblaze] cached -> {cache_path.name}")
        return payload
    try:
        with np.load(cache_path, allow_pickle=False) as npz:
            payload = {key: npz[key] for key in ("meta", "presence", "smart", "serials",
                                                 "models", "columns")}
    except (zipfile.BadZipFile, EOFError, ValueError, KeyError) as exc:
        raise ValueError(
            f"the Backblaze aggregate cache {cache_path} is unreadable ({type(exc).__name__}: "
            f"{exc}); it was most likely written by an interrupted parse. Delete it and "
            f"rerun -- the parse is idempotent.") from exc
    cached_columns = [str(name) for name in payload["columns"]]
    if cached_columns != list(smart_columns):
        raise ValueError(
            f"cached Backblaze table {cache_path.name} holds channels {cached_columns}, "
            f"but this run asks for {list(smart_columns)}; the cache is stale or was "
            f"hand-edited (its name hashes the scope) -- delete it, and bump "
            f"BACKBLAZE_AGG_VERSION if the parsing logic changed.")
    if verbose:
        print(f"[backblaze] loaded cached table {cache_path.name} "
              f"({len(payload['meta'])} drive-days, {len(payload['serials'])} drives)")
    return payload


# ---------------------------------------------------------------------------
# Per-drive protocol: validate, segment on gaps, censor, subsample
# ---------------------------------------------------------------------------
def _check_drive(serial: str, model: str, days: np.ndarray,
                 failure: np.ndarray) -> None:
    """Assert one drive's KEPT life obeys the documented ``failure`` semantics.

    A drive may appear at most ONCE per day, may carry at most ONE ``failure == 1`` row,
    and that row must be its LAST -- ``failure = 1`` means "the last day this drive was
    operational". ``_drive_records`` truncates the announced zombie tail (rows after the
    first ``failure == 1``, a real-corpus artifact -- §60) BEFORE calling this, so these
    checks are the invariant contract on the life actually modelled; a violation here
    means the caller's protocol broke, and raising beats silently modelling it."""
    duplicates = days[:-1][np.diff(days) == 0]
    if duplicates.size:
        raise ValueError(
            f"drive {serial!r} ({model}) appears more than once on "
            f"{[str(np.datetime64(int(d), 'D')) for d in np.unique(duplicates)[:5]]}; "
            f"a daily snapshot holds one row per drive per day.")
    failed_at = np.flatnonzero(failure == 1)
    if failed_at.size > 1:
        raise ValueError(
            f"drive {serial!r} ({model}) carries {failed_at.size} failure=1 rows on "
            f"{[str(np.datetime64(int(days[i]), 'D')) for i in failed_at[:5]]}; a drive "
            f"fails at most once (failure=1 marks its LAST operational day).")
    if failed_at.size == 1 and failed_at[0] != len(days) - 1:
        raise ValueError(
            f"drive {serial!r} ({model}) has {len(days) - 1 - failed_at[0]} row(s) AFTER "
            f"its failure=1 row ({str(np.datetime64(int(days[failed_at[0]]), 'D'))}, "
            f"last observed {str(np.datetime64(int(days[-1]), 'D'))}); failure=1 marks "
            f"the LAST operational day, so nothing may follow it.")


def _last_segment_start(days: np.ndarray) -> int:
    """Index of the first row of the drive's FINAL contiguous-enough segment: everything
    after the last gap longer than ``BACKBLAZE_MAX_GAP_DAYS`` (0 when there is none).
    See the module docstring's gap DECISION."""
    breaks = np.flatnonzero(np.diff(days) > BACKBLAZE_MAX_GAP_DAYS)
    return int(breaks[-1]) + 1 if breaks.size else 0


def _model_seed(seed: int, model: str) -> int:
    """Per-model survivor-subsampling seed, derived from ``config.seed`` and the model
    STRING. Deriving it per model (rather than drawing from one shared stream) keeps each
    model's kept survivors stable when another model is added to or removed from
    ``backblaze_models`` -- otherwise widening the scope would silently resample a fleet
    whose data did not change."""
    return int.from_bytes(hashlib.sha256(f"{seed}:{model}".encode()).digest()[:8], "big")


def _subsample_survivors(records: list, config: Config, verbose: bool = True) -> list:
    """Cap the CENSORED survivors per model at ``backblaze_max_survivors_per_model``,
    keeping every FAILED drive. Seeded (``_model_seed``) and returned in unit order, so
    the kept fleet is a pure function of (config.seed, model, the drives on disk)."""
    cap = config.backblaze_max_survivors_per_model
    if cap is None:
        return records
    by_model: dict = {}
    for record in records:
        by_model.setdefault(record["model"], []).append(record)
    kept = []
    for model, group in by_model.items():
        failed = [r for r in group if r["observed"]]
        survivors = [r for r in group if not r["observed"]]
        if len(survivors) > cap:
            rng = np.random.default_rng(_model_seed(config.seed, model))
            chosen = np.sort(rng.choice(len(survivors), size=cap, replace=False))
            survivors = [survivors[i] for i in chosen]
            if verbose:
                print(f"[backblaze] {model}: kept {cap} of "
                      f"{len(group) - len(failed)} censored survivors "
                      f"(seed {config.seed}) + all {len(failed)} failed drive(s)")
        kept.extend(failed + survivors)
    kept.sort(key=lambda record: record["unit"])
    return kept


def _drive_records(payload: dict, models: list, config: Config,
                   verbose: bool = True) -> list:
    """One record per drive that SURVIVES the per-drive rules, in unit order.

    Each record carries the drive's identity, its ``unit`` id (sorted-pair index + 1), the
    model index that becomes ``setting_1``, the row indices of its final segment (into the
    day-sorted payload), and whether its run ends in an OBSERVED failure."""
    meta = np.asarray(payload["meta"], np.int64)
    serials = [str(s) for s in payload["serials"]]
    drive_models = [str(m) for m in payload["models"]]
    model_index = {model: index for index, model in enumerate(models)}

    order = np.lexsort((meta[:, 1], meta[:, 0]))        # by (drive, day)
    meta = meta[order]
    codes, starts, counts = np.unique(meta[:, 0], return_index=True, return_counts=True)

    # Days each drive was PRESENT in the files (usable or not). Gap segmentation reads
    # these, so a run of capacity_bytes<0 rows -- a hole the loader punched itself -- can
    # never be mistaken for the drive leaving the fleet and silently discard its history.
    presence = np.asarray(payload["presence"], np.int64)
    presence_days: dict = {}
    if presence.size:
        p_order = np.lexsort((presence[:, 1], presence[:, 0]))
        presence = presence[p_order]
        p_codes, p_starts, p_counts = np.unique(presence[:, 0], return_index=True,
                                                return_counts=True)
        presence_days = {int(c): presence[st:st + ct, 1]
                         for c, st, ct in zip(p_codes, p_starts, p_counts)}

    records, n_short, n_gap_rows, zombies = [], 0, 0, []
    for code, start, count in zip(codes, starts, counts):
        days, failure = meta[start:start + count, 1], meta[start:start + count, 2]
        serial, model = serials[code], drive_models[code]
        # Segment on presence; then keep the usable rows that fall in the final segment.
        seen = presence_days.get(int(code), days)
        segment_start_day = seen[_last_segment_start(seen)]
        offset = int(np.searchsorted(days, segment_start_day, side="left"))
        n_gap_rows += offset
        # Real releases occasionally keep reporting a drive for a few days AFTER its
        # failure=1 row (lagging reports -- observed in the real 2024 corpus, e.g.
        # ZHZ3N9S2 with 3 rows after its 2024-09-20 failure). The failure day still ends
        # the life being modelled, so the kept segment is TRUNCATED at the FIRST
        # failure=1 row -- collected and ANNOUNCED below, never silent -- rather than
        # aborting a fleet-scale parse on one zombie tail (§60). DECISION (uncited):
        # only rows after the failure are trimmed, so no observation of the life itself
        # is ever discarded; a tail carrying further failure=1 rows is the same lagging
        # artifact and goes with it.
        failed_at = np.flatnonzero(failure[offset:] == 1)
        end = offset + int(failed_at[0]) + 1 if failed_at.size else count
        if end < count:
            zombies.append((serial, model,
                            str(np.datetime64(int(days[end - 1]), "D")),
                            str(np.datetime64(int(days[-1]), "D")), count - end))
        # The failure semantics belong to the LIFE being modelled: validating them over a
        # reused serial's whole history would hard-abort the very serial-reuse case the
        # gap rule exists to support (a failure inside a discarded earlier life is
        # dropped by that same rule).
        _check_drive(serial, model, days[offset:end], failure[offset:end])
        if end - offset < config.backblaze_min_days:
            n_short += 1
            continue
        records.append({
            "unit": int(code) + 1, "serial": serial, "model": model,
            "model_index": model_index[model],
            "rows": order[start + offset:start + end],
            # The kept life's final flag IS the drive's fate: 1 = observed failure, 0 =
            # the drive stopped being reported while still alive (right-censored).
            "observed": int(failure[end - 1]),
        })
    if zombies and verbose:
        shown = "; ".join(
            f"{serial} ({model}): failed {fail_day}, reported through {last_day} "
            f"(+{n_after} row(s))"
            for serial, model, fail_day, last_day, n_after in zombies[:3])
        print(f"[backblaze] {len(zombies)} drive(s) kept reporting AFTER their "
              f"failure=1 day -- each truncated at its failure day (failure=1 marks the "
              f"LAST operational day; later rows are lagging reports): {shown}")
    if not records:
        raise ValueError(
            f"no Backblaze drive reached backblaze_min_days={config.backblaze_min_days} "
            f"observed days: {len(codes)} drive(s) in scope, longest run "
            f"{int(counts.max())} day(s). Lower backblaze_min_days or widen the date "
            f"range.")
    if verbose:
        print(f"[backblaze] {len(records)} of {len(codes)} drives kept "
              f"({n_short} under {config.backblaze_min_days} observed days, "
              f"{n_gap_rows} row(s) before a gap > {BACKBLAZE_MAX_GAP_DAYS} days dropped)")
    return _subsample_survivors(records, config, verbose=verbose)


# ---------------------------------------------------------------------------
# The canonical frame
# ---------------------------------------------------------------------------
def _canonical_frame(payload: dict, records: list, smart_columns: list) -> pd.DataFrame:
    """Kept drives -> the canonical frame: ``unit_number``, ``time_cycles``, the three
    settings, the requested SMART channels, then ``event_observed``.

    ``time_cycles`` is 1..n over the drive's OBSERVED days (dense and consecutive by
    construction, see the gap DECISION), and the unpopulated SMART cells become
    ``BACKBLAZE_SMART_FILL``."""
    rows = np.concatenate([record["rows"] for record in records])
    lengths = [len(record["rows"]) for record in records]
    values = np.asarray(payload["smart"], np.float64)[rows]
    frame = pd.DataFrame(np.where(np.isnan(values), BACKBLAZE_SMART_FILL, values),
                         columns=list(smart_columns))
    frame.insert(0, INDEX_COLUMNS[0],
                 np.repeat([record["unit"] for record in records], lengths))
    frame.insert(1, INDEX_COLUMNS[1],
                 np.concatenate([np.arange(1, n + 1, dtype=np.int64) for n in lengths]))
    # setting_1 = the model index (the only operating-point-like axis a drive has);
    # setting_2/3 have no meaning here and stay 0.0 (see the module docstring).
    frame.insert(2, SETTING_COLUMNS[0],
                 np.repeat([float(r["model_index"]) for r in records], lengths))
    frame.insert(3, SETTING_COLUMNS[1], 0.0)
    frame.insert(4, SETTING_COLUMNS[2], 0.0)
    frame[_event_observed_column()] = np.repeat(
        [record["observed"] for record in records], lengths).astype(np.int64)
    return frame


def _manifest(records: list) -> pd.DataFrame:
    """The RECORD of which drives the scope + subsampling kept: one row per unit with its
    ``(serial_number, model)``, observed day count and censoring flag. The mapping is
    deterministic, but it is not recoverable from the emitted frames (they carry no
    serial), so the loader hands it back explicitly."""
    return pd.DataFrame({
        "unit_number": [record["unit"] for record in records],
        "serial_number": [record["serial"] for record in records],
        "model": [record["model"] for record in records],
        "n_days": [len(record["rows"]) for record in records],
        _event_observed_column(): [record["observed"] for record in records],
    })


# ---------------------------------------------------------------------------
# Split: deterministic, drive-level, STRATIFIED so test holds real failures
# ---------------------------------------------------------------------------
def _select_test_units(frame: pd.DataFrame, config: Config) -> np.ndarray:
    """Held-out unit ids: ``backblaze_test_fraction`` of DRIVES, stratified by
    ``(model, event_observed)``.

    DECISION (uncited). Failures are ~1 in 23,500 drive-days, so a plain random drive
    split routinely yields a test set with ZERO failed drives -- which is not a bad split
    but an UNSCOREABLE one (no positives: recall undefined, lead time undefined). The
    split therefore:

      * strata = (model index, censoring flag), so both models AND both fates are
        represented on each side, and no model's failures can end up entirely in train;
      * only drives with at least ``window_size + 1`` observed days are ELIGIBLE for test
        (a test drive must survive truncation into >= 1 window with >= 1 day cut off).
        Shorter drives are NOT dropped -- they stay in TRAIN, where a unit shorter than
        the window simply yields no windows;
      * within a stratum, drives are taken by SYSTEMATIC sampling at the requested rate
        (midpoints of ``k`` equal spans), which spreads the held-out drives across the
        serial ordering instead of preferring one end of it;
      * a stratum with a single eligible drive contributes none (it stays in train), and
        every stratum keeps at least one train drive (``k <= n - 1``).

    No RNG: the split is a pure function of the kept fleet, the fraction and the window
    size, all of which are in the cache key."""
    by_unit = frame.groupby(INDEX_COLUMNS[0], sort=True)
    sizes = by_unit.size()
    units, lengths = sizes.index.to_numpy(), sizes.to_numpy()
    observed = by_unit[_event_observed_column()].max().to_numpy()
    model_index = by_unit[SETTING_COLUMNS[0]].first().to_numpy()
    eligible = lengths >= config.window_size + 1

    chosen = []
    for stratum in np.unique(np.column_stack([model_index, observed]), axis=0):
        in_stratum = eligible & (model_index == stratum[0]) & (observed == stratum[1])
        members = units[in_stratum]
        if members.size < 2:
            continue                       # cannot split one drive; it stays in train
        n_test = min(max(int(round(members.size * config.backblaze_test_fraction)), 1),
                     members.size - 1)
        spans = (np.arange(n_test) + 0.5) * members.size / n_test   # span midpoints
        chosen.append(members[np.floor(spans).astype(np.int64)])
    if not chosen:
        raise ValueError(
            f"the Backblaze split produced an empty test set: of {len(units)} kept "
            f"drive(s) (runs {int(lengths.min())}..{int(lengths.max())} days) only "
            f"{int(eligible.sum())} reach the {config.window_size + 1} days a truncated "
            f"test drive needs, and no (model, censoring) stratum holds two of them. "
            f"Lower window_size, or widen the date range so drives are observed longer.")
    test_units = np.sort(np.concatenate(chosen))
    n_failures = int(observed[np.isin(units, test_units)].sum())
    if not n_failures:
        raise ValueError(
            f"the Backblaze test split holds {len(test_units)} drive(s) but NOT ONE "
            f"observed failure ({int(observed.sum())} of {len(units)} kept drives failed, "
            f"of which {int((observed * eligible).sum())} are long enough to hold out), "
            f"so it cannot be scored: precision/recall and lead time are undefined "
            f"without a positive. Widen the date range or the model scope until each "
            f"model contributes at least two failed drives.")
    return test_units


def _truncate_test(df_test_full: pd.DataFrame, config: Config) -> tuple:
    """Truncate each test drive at ``BACKBLAZE_TEST_TRUNCATION`` of its observed run;
    return the truncated frame and ``{unit: remaining observed days}``.

    Same guards as XJTU/N-CMAPSS/hydraulic: keep at least ``window_size`` days (so the
    drive yields a window) and cut at least one (so the provided RUL is > 0).
    ``_select_test_units`` only ever hands over drives long enough for both, so the raise
    is a defensive backstop for a caller that truncates a frame the eligibility rule never
    approved.

    DECISION (uncited): EVERY test drive is truncated, not only the failed ones. For a
    survivor the cut-off remainder is not its remaining life but the time to the END OF
    OBSERVATION -- exactly the lower bound ``data.add_train_rul`` documents -- and
    truncating it uniformly keeps ONE protocol across the test set: at a horizon shorter
    than that remainder the survivor is a clean alarm NEGATIVE whose rows are all
    knowable, so it contributes its whole window set instead of having its tail dropped
    by ``data.add_alarm_label``."""
    frames, rul = [], {}
    for unit_id, unit_df in df_test_full.groupby(INDEX_COLUMNS[0], sort=True):
        unit_df = unit_df.sort_values(INDEX_COLUMNS[1])
        n_days = len(unit_df)
        keep = int(np.floor(n_days * BACKBLAZE_TEST_TRUNCATION))
        keep = max(config.window_size, min(keep, n_days - 1))
        if keep < 1 or keep >= n_days:
            raise ValueError(
                f"Backblaze test drive {unit_id}: cannot truncate {n_days} observed "
                f"day(s) to a valid prefix (window_size={config.window_size}); the drive "
                f"is too short to hold out.")
        frames.append(unit_df.iloc[:keep])
        rul[int(unit_id)] = n_days - keep
    return pd.concat(frames, ignore_index=True), rul


def _report_alarm_reachability(df_test: pd.DataFrame, rul: dict, config: Config) -> None:
    """Say out loud whether the test split can produce an alarm POSITIVE.

    The alarm arm is scored at each test drive's LAST kept day (``sweep.run_alarm_sweep``
    over ``data.make_test_last_windows``), so a test drive is a positive only if its
    remaining life there is within ``alarm_horizon``. Truncation leaves a failed drive
    ``1 - BACKBLAZE_TEST_TRUNCATION`` of its run to live, so a short horizon on a
    long-observed fleet can leave the test set with no positive at all -- AUROC and lead
    time are then ``nan``, which ``evaluate.alarm_metrics`` reports rather than raising.

    A NOTICE, not an exception: nothing about the DATA is wrong (the split is stratified
    and does hold failed drives), it is the horizon/truncation pairing that cannot be
    scored, and both are knobs the caller sets per experiment."""
    if config.alarm_horizon is None:
        return
    observed = df_test.groupby(INDEX_COLUMNS[0])[_event_observed_column()].max()
    within = [unit for unit, remaining in rul.items()
              if observed[unit] == 1 and remaining <= config.alarm_horizon]
    if within:
        return
    print(f"[backblaze] NOTICE: no test drive is within alarm_horizon="
          f"{config.alarm_horizon} cycles of failure at its last kept day (remaining "
          f"life there: {sorted(rul[u] for u in observed.index if observed[u] == 1)} "
          f"days for the failed test drives), so the alarm metrics will be nan. Raise "
          f"alarm_horizon, or widen the date range so short-lived failed drives enter "
          f"the fleet.")


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------
def _kept_records(config: Config, verbose: bool = True) -> tuple:
    """``(payload, records, smart_columns)`` for the scoped, cleaned, subsampled fleet --
    the shared front half of ``load_backblaze`` and ``drive_manifest``."""
    models, smart_columns = _check_scope(config)
    payload = _load_or_build_aggregate(models, smart_columns, config, verbose=verbose)
    return payload, _drive_records(payload, models, config, verbose=verbose), smart_columns


def drive_manifest(config: Config, verbose: bool = False) -> pd.DataFrame:
    """Which drives this config keeps, as ``unit_number, serial_number, model, n_days,
    event_observed``. The subsampled fleet is seeded and reproducible, but the emitted
    frames carry no serial numbers, so this is how a run records WHICH drives it trained
    on (RESEARCH_PLAN §9: every sampled unit set is written to the run directory)."""
    _, records, _ = _kept_records(config, verbose=verbose)
    return _manifest(records)


def load_backblaze(config: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load the Backblaze Drive Stats fleet and return the canonical ``(df_train,
    df_test, rul_truth)`` triple.

    Both frames carry ``unit_number``, ``time_cycles``, the three settings,
    ``config.backblaze_smart_columns`` and ``event_observed``. ``rul_truth`` is indexed by
    unit_number = observed days remaining at each TEST drive's last kept day -- the true
    remaining life for a FAILED drive, a lower bound for a CENSORED one (module
    docstring); run with ``config.alarm_horizon`` set so the censoring-aware target
    (§54) decides what is knowable."""
    payload, records, smart_columns = _kept_records(config)
    frame = _canonical_frame(payload, records, smart_columns)
    test_units = _select_test_units(frame, config)
    is_test = frame[INDEX_COLUMNS[0]].isin(test_units).to_numpy()
    df_train = frame.loc[~is_test].reset_index(drop=True)
    df_test, rul = _truncate_test(frame.loc[is_test].reset_index(drop=True), config)
    observed_col = df_train.groupby(INDEX_COLUMNS[0])[_event_observed_column()].max()
    print(f"[backblaze] {len(records)} drives -> {len(observed_col)} train "
          f"({int(observed_col.sum())} failed) / {len(test_units)} test "
          f"({len(rul)} truncated at {BACKBLAZE_TEST_TRUNCATION:g} of their run); "
          f"{len(df_train)} + {len(df_test)} drive-days")
    _report_alarm_reachability(df_test, rul, config)
    rul_truth = pd.Series(rul, name="rul_truth").sort_index()
    rul_truth.index.name = INDEX_COLUMNS[0]
    return df_train, df_test, rul_truth
