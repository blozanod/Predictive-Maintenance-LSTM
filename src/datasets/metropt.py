"""MetroPT-3 metro-train APU loader -> canonical C-MAPSS-shaped frames.

Dataset (Veloso et al. 2022, Sci Data 9:764; UCI ML Repository #791): one flat CSV of
telemetry from the Air Production Unit (APU) of a Porto Metro train, 2020-02-01 ..
2020-09-01. ONE file, ``MetroPT3(AirCompressor).csv`` (literal parentheses; forks
rename it -- ``METROPT_CSV_NAMES`` lists every accepted spelling), 1516948 x 17, header
verbatim in ``METROPT_EXPECTED_HEADER``:

  * ``Unnamed: 0``  -- the CSV's first header cell is EMPTY, so pandas names it that.
    A meaningless row counter (0, 10, 20, ...); it is NEVER a feature and is excluded
    by reading with ``usecols`` (the header check still proves it was there).
  * ``timestamp``   -- ``'%Y-%m-%d %H:%M:%S'``, NAIVE local time (Europe/Lisbon). Never
    localized here: every derived quantity (bin edges, failure windows) is computed in
    that same naive clock, so a tz conversion could only introduce error.
  * 7 ANALOG signals (``METROPT_ANALOG_COLUMNS``) and 8 DIGITAL ones
    (``METROPT_DIGITAL_COLUMNS``, float-valued 0.0/1.0). NOTE ``DV_eletric`` is
    MISSPELLED in the shipped header (missing the 'c') and is kept verbatim; and the
    UCI prose lists ``Motor_current`` before ``Oil_temperature`` while the FILE has
    them the other way round -- the file wins, and the header check enforces it.

The stream is SHIPPED decimated to ~10 s and is IRREGULAR (10 s x1.34M, 9 s x128k,
12 s x38k, plus much larger holes): ~17.6% of wall-clock time is simply ABSENT, with
no NaN row and no sentinel value. It is therefore never reindexed onto a fixed
frequency -- fabricating rows for absent time is exactly the failure mode the
min-samples rule below defends against.

**There is NO label column.** Ground truth is out-of-band: the four documented air-leak
events in ``config.METROPT_FAILURE_EVENTS`` (the UCI "Failure Information" table).

Adaptation to the pipeline's canonical (cycle-level) frame -- the whole point, so
everything downstream of ``data.load_prepared`` runs unchanged (mirrors XJTU §22 and
N-CMAPSS §27):
  * one "cycle" = one fixed ``config.metropt_cycle_minutes``-wide wall-clock bin
    (floored against the Unix epoch, so bin edges depend on nothing but the knob).
    Per bin: ``{analog}_mean``/``{analog}_std`` and ``{digital}_duty`` (a binary
    channel's mean IS the fraction of the bin it was active), in exactly
    ``config.metropt_feature_columns()`` order -- 22 channels. ``std`` is pandas'
    sample std (ddof=1), so a 1-row bin gives NaN -> 0.0, the N-CMAPSS convention.
  * a bin holding fewer than ``config.metropt_min_samples_per_cycle`` raw rows is
    DROPPED, not aggregated: with ~17.6% of time invisibly missing, a sparsely covered
    bin's mean/std is a different quantity from a full bin's, and nothing in the file
    marks it. This is the only defence against the invisible gaps.
  * one "unit" = one INTERVENTION RUN (RESEARCH_PLAN §4: "the clock resets at each
    maintenance intervention"). The 4 events cut the record into 5 runs: run k is the
    period ENDING at the start of event k (k = 1..4), and run 5 -- everything after
    event 4 -- is RIGHT-CENSORED (the record stops while the APU is still alive).
    ``unit_number`` IS the run number, so unit k names the event it ends at.
  * rows falling INSIDE a failure window belong to neither run (the APU is already
    failing, and the window is the intervention itself) and are dropped. A bin is
    grouped by (run, bin start), so a bin that straddles a boundary splits into two
    partial bins and each faces the min-samples rule on its own.
  * ``event_observed`` (``data.EVENT_OBSERVED_COLUMN``) is 1 for a run that ends at a
    documented event and 0 for the censored tail run; ``fault_type``/``fault_severity``
    carry the event table's ``failure``/``severity`` strings (all four events are
    "Air leak"/"High stress" -- the columns exist so the RQ-F probe and a future
    multi-fault-type dataset slot in unchanged) and are ``"none"`` when censored.
  * ``setting_1/2/3 = 0.0``: one APU, one operating point, no operating-point concept.
    ``condition_norm`` therefore resolves auto-OFF via the default (MetroPT-3 is in
    neither ``MULTI_CONDITION_DATASETS`` nor ``XJTU_DATASETS``); forcing it ON would
    normalize against a single all-zero condition group, which is a no-op group-wise
    z-score, not an error.

DECISION (uncited): ``time_cycles`` counts SURVIVING bins, renumbered 1..n
consecutively per run (the canonical frame requires consecutive 1-based cycles). So a
wall-clock gap collapses rather than leaving a hole, and RUL is measured in
*aggregated cycles*, not in hours -- with ~82% time coverage a 100-cycle RUL is ~122
wall-clock hours at the default 60 min bin. Every horizon/lead-time number for this
dataset is in cycles and must be read that way.

Split & test protocol (DECISION, uncited): ``config.metropt_test_runs`` (1-based run
numbers) are held out and each is truncated at ``config.metropt_test_truncation`` of
its length, so the pipeline's predict-at-last-observed-cycle protocol applies (the same
device as XJTU §22 / N-CMAPSS §27); ``rul_truth`` = remaining cycles at truncation. A
CENSORED run can never be a test run -- its remaining life is unknown, so "RUL at
truncation" does not exist for it -- and asking for one raises.

**Comparability warning:** MetroPT results in the literature are computed on the raw
~1 Hz/10 s stream under each author's own labelling of the failure reports (Davari et
al.'s 21 windows, per-second anomaly scores, sliding alarm windows, ...). These
hour-binned, intervention-run, censoring-aware numbers are NOT comparable to any of
them and must never share a table with published MetroPT results (RESEARCH_PLAN role:
same-protocol cross-model comparison for the censoring chapter, RQ-B/RQ-E, plus the
alarm/lead-time metric -- never the NASA score).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import (Config, INDEX_COLUMNS, SETTING_COLUMNS,
                      METROPT_DATASETS, METROPT_ANALOG_COLUMNS,
                      METROPT_DIGITAL_COLUMNS, METROPT_SIGNAL_COLUMNS,
                      METROPT_TIMESTAMP_COLUMN, METROPT_FEATURE_COLUMNS,
                      METROPT_FAILURE_EVENTS, METROPT_NOMINAL_CADENCE_S,
                      metropt_feature_columns)
from .base import resolve_data_dir

# Accepted subdirectory name(s) of ``config.data_root`` holding the CSV (flat). The UCI
# download folder is "MetroPT-3"; "MetroPT" is the common shortened rename (§26 style).
METROPT_SUBDIR = ("MetroPT-3", "MetroPT")

# Dataset names this family serves (the campaign sweeps these).
DATASETS = tuple(METROPT_DATASETS)

# Accepted file names, in PREFERENCE order. The first is the name UCI ships (literal
# parentheses); the others are the two renames forks use. Exactly one may be present --
# two copies could differ, and picking one silently is precisely what §7 forbids.
METROPT_CSV_NAMES = ("MetroPT3(AirCompressor).csv", "MetroPT3.csv", "metropt3.csv")

# pandas' name for the file's EMPTY first header cell (a meaningless row counter).
METROPT_ROW_INDEX_COLUMN = "Unnamed: 0"

# The timestamp column's exact strftime layout. Parsed with format= (never inferred):
# an inferred parse would silently accept a day/month-swapped fork.
METROPT_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

# The file's header, verbatim and in file order. Any drift fails loud.
METROPT_EXPECTED_HEADER = ((METROPT_ROW_INDEX_COLUMN, METROPT_TIMESTAMP_COLUMN)
                           + tuple(METROPT_SIGNAL_COLUMNS))

# The two RQ-F secondary-label columns the loader attaches to every row (RESEARCH_PLAN
# §4: adjustment vs. replacement). DECISION (uncited): they carry the event table's own
# ``failure``/``severity`` STRINGS rather than an invented ordinal, because MetroPT's
# four events are all one type -- inventing a severity ladder here would fabricate the
# very signal the RQ-F probe is supposed to measure. A censored run has no terminal
# event and is labelled ``NO_FAULT_LABEL``.
FAULT_TYPE_COLUMN = "fault_type"
FAULT_SEVERITY_COLUMN = "fault_severity"
NO_FAULT_LABEL = "none"

# Bump whenever the BINNING/RUN-SEGMENTATION LOGIC changes, so stale per-cycle caches
# (cache/metropt_agg_v<N>_*.npz) are invalidated -- the role CACHE_SCHEMA_VERSION plays
# for embeddings and NCMAPSS_AGG_VERSION for N-CMAPSS. The aggregate ALSO depends on the
# two binning knobs and on the failure-event table; all three are folded into the cache
# FILENAME (see ``_agg_cache_path``) rather than this version, so each combination is a
# separate, coexisting aggregate.
METROPT_AGG_VERSION = 1


def _canon_columns() -> list:
    """Canonical numeric-matrix column order used by the aggregate cache (the emitted
    frame appends the censoring/fault columns to this)."""
    return list(INDEX_COLUMNS) + list(SETTING_COLUMNS) + list(metropt_feature_columns())


_CANON_COLUMNS = _canon_columns()

_ANALOG = list(METROPT_ANALOG_COLUMNS)
_DIGITAL = list(METROPT_DIGITAL_COLUMNS)
_SIGNALS = list(METROPT_SIGNAL_COLUMNS)

# The only values a DIGITAL channel may take in the shipped file (as float64). A third
# value would make ``{digital}_duty`` -- a FRACTION-of-bin-active -- meaningless.
_DIGITAL_VALUE_SET = {0.0, 1.0}


def _event_observed_column() -> str:
    """``data.EVENT_OBSERVED_COLUMN``, imported lazily: ``src.data`` imports the dataset
    registry at module load, so a module-level import here would be circular. Resolved
    per call (never cached) so the constant has exactly one home."""
    from ..data import EVENT_OBSERVED_COLUMN
    return EVENT_OBSERVED_COLUMN


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
def _resolve_dir(config: Config) -> Path:
    return resolve_data_dir(config, METROPT_SUBDIR)


def _csv_candidates(root: Path) -> list:
    """Every accepted CSV name that actually exists under ``root``, in preference
    order. Empty when the folder is missing (no exception -- ``is_available`` calls
    this on a path the user may not have created yet)."""
    if not root.is_dir():
        return []
    return [root / name for name in METROPT_CSV_NAMES if (root / name).is_file()]


def _find_csv(root: Path) -> Path:
    """The one MetroPT-3 CSV under ``root``; raises if there are zero or several."""
    matches = _csv_candidates(root)
    if not matches:
        raise FileNotFoundError(
            f"no MetroPT-3 CSV under {root}; expected one of "
            f"{list(METROPT_CSV_NAMES)} (download UCI dataset 791 into that folder, "
            f"RESEARCH_PLAN §3).")
    if len(matches) > 1:
        raise ValueError(
            f"ambiguous MetroPT-3 files under {root}: {[m.name for m in matches]}; "
            f"keep exactly one (two copies can differ, and silently preferring one "
            f"would hide which readings were used).")
    return matches[0]


def is_available(config: Config) -> bool:
    """Cheap on-disk check: is at least one accepted MetroPT-3 CSV present?
    (The campaign skips unavailable datasets with a notice, CHANGES.md §24.)"""
    return bool(_csv_candidates(_resolve_dir(config)))


# ---------------------------------------------------------------------------
# Raw CSV -> validated signal frame + timestamps
# ---------------------------------------------------------------------------
def _read_raw(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Read the CSV and validate it byte-for-byte against the documented schema.

    Returns ``(signals, ts)``: a float frame of the 15 signal columns (file order) and
    the parsed ``datetime64[ns]`` Series. ``Unnamed: 0`` is never read (``usecols``),
    but the header check above still proves it was present and first -- so a fork that
    dropped it, or that shifted the columns, fails loud instead of feeding a row counter
    to a model. Every check names BOTH the expected and the observed value (§7).
    """
    header = list(pd.read_csv(csv_path, nrows=0).columns)
    if header != list(METROPT_EXPECTED_HEADER):
        raise ValueError(
            f"{csv_path.name}: header does not match the MetroPT-3 schema.\n"
            f"  expected: {list(METROPT_EXPECTED_HEADER)}\n"
            f"  observed: {header}\n"
            "Update METROPT_* in src/config.py to match the file (and bump "
            "METROPT_AGG_VERSION); do NOT silently reorder or rename.")
    # Read only timestamp + signals: the row counter can never reach a feature column.
    raw = pd.read_csv(csv_path, usecols=[METROPT_TIMESTAMP_COLUMN] + _SIGNALS)

    # Explicit format, never inferred (a swapped D/M fork would parse "silently right").
    ts = pd.to_datetime(raw[METROPT_TIMESTAMP_COLUMN],
                        format=METROPT_TIMESTAMP_FORMAT, errors="coerce")
    unparsed = raw.loc[ts.isna(), METROPT_TIMESTAMP_COLUMN]
    if len(unparsed):
        raise ValueError(
            f"{csv_path.name}: {len(unparsed)} timestamp(s) do not match the expected "
            f"format {METROPT_TIMESTAMP_FORMAT!r}; first offenders: "
            f"{list(unparsed.head(3))}")

    non_numeric = {c: str(raw[c].dtype) for c in _SIGNALS
                   if not pd.api.types.is_numeric_dtype(raw[c])}
    if non_numeric:
        raise ValueError(
            f"{csv_path.name}: signal column(s) did not parse as numbers (expected "
            f"float64 for all of {_SIGNALS}); observed dtypes {non_numeric}.")

    na_counts = raw[_SIGNALS].isna().sum()
    missing = {c: int(n) for c, n in na_counts.items() if n}
    if missing:
        raise ValueError(
            f"{csv_path.name}: the shipped file has NO missing signal values (absent "
            f"time is absent ROWS, not NaN cells), but these columns hold NaNs: "
            f"{missing}. Aggregating around them would silently change what a bin's "
            f"mean/duty measures.")

    off_domain = {c: sorted(set(pd.unique(raw[c].to_numpy())) - _DIGITAL_VALUE_SET)
                  for c in _DIGITAL}
    off_domain = {c: v for c, v in off_domain.items() if v}
    if off_domain:
        raise ValueError(
            f"{csv_path.name}: DIGITAL channels must be valued "
            f"{sorted(_DIGITAL_VALUE_SET)} (their per-bin mean IS the duty fraction), "
            f"but these carry other values: {off_domain}.")
    return raw[_SIGNALS], ts


# ---------------------------------------------------------------------------
# Intervention runs (the "unit" axis)
# ---------------------------------------------------------------------------
def _validate_events(events) -> tuple[np.ndarray, np.ndarray]:
    """Parse the failure table into sorted ``(starts, ends)`` datetime64 arrays.

    The windows must be well-formed (``start <= end``) and STRICTLY SEPARATED in
    chronological order (``end[k-1] < start[k]``): overlapping or unsorted windows would
    make "the k-th run" ambiguous and silently mis-assign rows to runs. The table is a
    code constant, so this is an invariant check on ``config.METROPT_FAILURE_EVENTS``
    itself, not on user input."""
    starts = np.array([np.datetime64(e["start"]) for e in events], dtype="datetime64[ns]")
    ends = np.array([np.datetime64(e["end"]) for e in events], dtype="datetime64[ns]")
    bad = [int(e["event"]) for e, s, t in zip(events, starts, ends) if s > t]
    if bad:
        raise ValueError(
            f"METROPT_FAILURE_EVENTS: event(s) {bad} have start > end; every failure "
            f"window must satisfy start <= end.")
    out_of_order = [i + 1 for i in range(1, len(events)) if ends[i - 1] >= starts[i]]
    if out_of_order:
        raise ValueError(
            f"METROPT_FAILURE_EVENTS: event(s) {out_of_order} start at or before the "
            f"previous event ends; the windows must be chronological and disjoint so "
            f"that 'run k ends at event k' is well defined.")
    return starts, ends


def _assign_runs(ts: pd.Series, starts: np.ndarray, ends: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Per-row ``(run_id, inside_failure_window)``.

    ``run_id`` = 1 + (number of failure windows that have already CLOSED), so run k is
    the period between event k-1's end and event k's start, i.e. the run that ends at
    event k. Rows with ``start_k <= t <= end_k`` for some k are INSIDE the intervention
    and belong to no run. Both are computed on the raw timestamps only, so the result
    does not depend on the file's row order."""
    t = ts.to_numpy("datetime64[ns]")
    inside = np.zeros(len(t), dtype=bool)
    for s, e in zip(starts, ends):
        inside |= (t >= s) & (t <= e)
    # searchsorted on the (validated, sorted) window ends: for a row inside window k the
    # count is k-1, but those rows are dropped, so only the between-window case matters.
    run = np.searchsorted(ends, t, side="left").astype(np.int64) + 1
    return run, inside


# ---------------------------------------------------------------------------
# Aggregation (irregular ~10 s rows -> fixed-width per-cycle bins)
# ---------------------------------------------------------------------------
def _aggregate(signals: pd.DataFrame, ts: pd.Series, events, cycle_minutes: int,
               min_samples: int, min_coverage: float = 0.0,
               verbose: bool = True) -> np.ndarray:
    """Bin the irregular stream into the canonical per-cycle numeric matrix
    ``(n_cycles, len(_CANON_COLUMNS) + 1)`` -- the canonical columns plus
    ``event_observed`` as the last column.

    Bins are fixed ``cycle_minutes``-wide wall-clock windows floored against the Unix
    epoch (``dt.floor``), grouped WITH the run id so a bin straddling a run boundary
    splits into two partial bins instead of mixing two lives into one row. Bins holding
    fewer than ``min_samples`` raw rows are dropped (the invisible-gap defence), and the
    survivors are renumbered 1..n consecutively within each run.
    """
    starts, ends = _validate_events(events)
    run, inside = _assign_runs(ts, starts, ends)

    work = signals.copy()
    work["__run"] = run
    # Bin START, not a bin index: dt.floor is epoch-anchored, so the edges depend only
    # on cycle_minutes -- never on where the file happens to begin.
    work["__bin"] = ts.dt.floor(f"{cycle_minutes}min").to_numpy()
    work = work.loc[~inside]

    g = work.groupby(["__run", "__bin"], sort=True)
    sizes = g.size()
    counts = sizes.to_numpy(np.int64)
    # The invisible-gap defence, in TWO parts. An absolute row floor alone is not
    # scale-invariant: metropt_cycle_minutes is this dataset's RQ-G sweep lever, so a
    # fixed count is ~140x stricter at 10-minute bins than at 1440-minute ones, and the
    # aggregation-granularity comparison would be confounded by a data-quality gradient
    # rather than measuring granularity. The COVERAGE fraction fixes the strictness at
    # every bin width; the absolute floor still guards the degenerate tiny-bin case.
    expected = cycle_minutes * 60.0 / METROPT_NOMINAL_CADENCE_S
    coverage_floor = int(np.ceil(min_coverage * expected))
    threshold = max(int(min_samples), coverage_floor)
    keep = counts >= threshold
    if not keep.any():
        raise ValueError(
            f"MetroPT-3: no {cycle_minutes}-minute bin reached the {threshold}-row "
            f"threshold (max of metropt_min_samples_per_cycle={min_samples} and "
            f"metropt_min_bin_coverage={min_coverage:g} x {expected:.0f} expected rows) "
            f"(rows outside the failure windows: {len(work)}; bins formed: "
            f"{len(counts)}; largest: {int(counts.max(initial=0))}). Lower either "
            f"threshold or widen metropt_cycle_minutes -- the loader will not "
            f"aggregate sparse bins.")
    dropped = int((~keep).sum())
    if verbose:
        # Report the defence's ACTIVITY: a silent filter that never fires reads exactly
        # like one that is working.
        print(f"[metropt] gap defence: kept {int(keep.sum())} of {len(counts)} bins "
              f"({dropped} dropped, {100.0 * dropped / max(len(counts), 1):.1f}%) at a "
              f"{threshold}-row threshold ({min_coverage:g} coverage of "
              f"{expected:.0f} expected rows per {cycle_minutes}-minute bin)")

    # ddof=1 like N-CMAPSS (§27): a 1-row bin has no sample std -> NaN -> 0.0.
    mean_arr = g[_ANALOG].mean().to_numpy(np.float64)[keep]
    std_arr = g[_ANALOG].std().fillna(0.0).to_numpy(np.float64)[keep]
    duty_arr = g[_DIGITAL].mean().to_numpy(np.float64)[keep]

    n_an = len(_ANALOG)
    feat = np.empty((int(keep.sum()), len(METROPT_FEATURE_COLUMNS)), np.float64)
    feat[:, 0:2 * n_an:2] = mean_arr          # {analog}_mean, interleaved ...
    feat[:, 1:2 * n_an:2] = std_arr           # ... with {analog}_std
    feat[:, 2 * n_an:] = duty_arr             # then {digital}_duty
    assert feat.shape[1] == len(METROPT_FEATURE_COLUMNS)

    idx = sizes.index[keep]
    runs = idx.get_level_values("__run").to_numpy(np.int64)
    # 1-based CONSECUTIVE cycles over the SURVIVING bins of each run (see the module
    # docstring's DECISION on what a gap does to the clock).
    cycles = (pd.Series(runs).groupby(runs).cumcount().to_numpy(np.int64) + 1)
    # A run is OBSERVED only if the record ACTUALLY REACHES the event it is supposed to
    # end at. Deriving observedness from the run INDEX alone (`runs <= len(events)`) is
    # the censoring bug §54 exists to prevent, committed inside the loader written to
    # prevent it: on a truncated mirror -- or the moment a 5th event is appended to the
    # table, which the module's own errors instruct -- the genuinely right-censored tail
    # run would silently become an "observed failure", pass the censored-test-run guard,
    # and be scored against a `rul_truth` for a failure the data never contains.
    t_max = ts.max()
    reached = np.array([s <= t_max for s in starts], dtype=bool)
    observed = np.zeros(len(runs), np.float64)
    in_range = runs <= len(events)
    observed[in_range] = reached[runs[in_range] - 1].astype(np.float64)
    unreached = [i + 1 for i, r in enumerate(reached) if not r]
    if unreached and verbose:
        print(f"[metropt] NOTE: the record ends at {t_max}, BEFORE documented event(s) "
              f"{unreached}; the run(s) ending at them are therefore RIGHT-CENSORED, "
              f"not observed failures. Use an observed run for metropt_test_runs.")
    zeros = np.zeros(len(runs), np.float64)
    return np.column_stack([runs.astype(np.float64), cycles.astype(np.float64),
                            zeros, zeros, zeros, feat, observed])


# ---------------------------------------------------------------------------
# Parsed-frame cache (binning 1.5M rows is slow; the aggregate is ~10^3 rows)
# ---------------------------------------------------------------------------
def _events_digest(events) -> str:
    """Short content hash of the failure table. It is a CODE constant rather than a
    Config field, so it cannot ride the config cache key -- but it fully reshapes the
    aggregate (run boundaries, dropped rows), so it rides the cache FILENAME instead."""
    blob = json.dumps([dict(e) for e in events], sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:8]


def _source_digest(csv_path: Path) -> str:
    """Short identity hash of the resolved CSV: its NAME, SIZE and MTIME.

    The cache is otherwise keyed only by knobs, and MetroPT has a single dataset name --
    so without this two different files (two data roots, a re-download, a swap between
    the accepted file names) collide on one cache file and the second load silently
    serves the FIRST file's readings. ``_find_csv`` refuses to choose between two copies
    precisely because they can differ; the cache must not then do it across time.
    Content is not hashed: the file is ~213 MB and (name, size, mtime) is the standard
    cheap staleness triple."""
    st = csv_path.stat()
    blob = f"{csv_path.name}|{st.st_size}|{int(st.st_mtime)}".encode()
    return hashlib.sha256(blob).hexdigest()[:8]


def _agg_cache_path(config: Config, events, csv_path: Path) -> Path:
    """Aggregate cache path. Every knob that SHAPES the aggregate is in the filename
    (bin width, min-samples, event-table digest, the SOURCE FILE's identity) plus the
    logic version; nothing about the file's LOCATION is (location-independent, like
    embeddings, §23 -- the source digest is content identity, not a path)."""
    return (Path(config.cache_dir)
            / f"metropt_agg_v{METROPT_AGG_VERSION}"
              f"_c{config.metropt_cycle_minutes}m"
              f"_n{config.metropt_min_samples_per_cycle}"
              f"_v{config.metropt_min_bin_coverage:g}"
              f"_e{_events_digest(events)}"
              f"_f{_source_digest(csv_path)}.npz")


def _load_or_build_aggregate(config: Config, verbose: bool = True) -> pd.DataFrame:
    """Return the UNTRUNCATED per-cycle frame for every run, from a versioned cache
    (parsing + binning 1.5M rows is tens of seconds; the aggregate is ~10^3 rows).
    Idempotent, and a pure function of the binning knobs + the event table, both of
    which are in the cache filename."""
    events = METROPT_FAILURE_EVENTS
    # The source file is resolved BEFORE the cache lookup: its identity is part of the
    # cache key, so two different CSVs can never share one aggregate.
    csv_path = _find_csv(_resolve_dir(config))
    cache_path = _agg_cache_path(config, events, csv_path)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as npz:
            mat = npz["cycles"]
        if verbose:
            print(f"[metropt] loaded cached aggregate {cache_path.name} "
                  f"({len(mat)} cycles, {len(np.unique(mat[:, 0]))} runs)")
    else:
        if verbose:
            print(f"[metropt] parsing {csv_path.name} (~10 s rows -> "
                  f"{config.metropt_cycle_minutes} min cycles; "
                  f"min_samples={config.metropt_min_samples_per_cycle})...")
        signals, ts = _read_raw(csv_path)
        mat = _aggregate(signals, ts, events, config.metropt_cycle_minutes,
                         config.metropt_min_samples_per_cycle,
                         config.metropt_min_bin_coverage, verbose=verbose)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: a killed run must not leave a truncated .npz that the next run
        # loads as a complete aggregate.
        # NOTE the name must still END in .npz: np.savez appends the extension when
        # it is absent, which would leave the temp file somewhere os.replace cannot see.
        tmp = cache_path.with_name(cache_path.name + ".tmp.npz")
        np.savez(tmp, cycles=mat.astype(np.float32))
        os.replace(tmp, cache_path)
        if verbose:
            print(f"[metropt] parsed {len(ts)} raw rows -> {len(mat)} cycles over "
                  f"{len(np.unique(mat[:, 0]))} runs; cached -> {cache_path.name}")
    return _frame_from_matrix(np.asarray(mat, np.float64), events)


def _frame_from_matrix(mat: np.ndarray, events) -> pd.DataFrame:
    """Numeric aggregate matrix -> the emitted frame: canonical columns, then
    ``event_observed`` and the two RQ-F fault-label columns (derived from the run id,
    since run k ends at the k-th chronological event)."""
    obs_col = _event_observed_column()
    df = pd.DataFrame(mat, columns=_CANON_COLUMNS + [obs_col])
    df["unit_number"] = df["unit_number"].astype(np.int64)
    df["time_cycles"] = df["time_cycles"].astype(np.int64)
    df[obs_col] = df[obs_col].astype(np.int64)
    pairs = list(zip(df["unit_number"].to_numpy(), df[obs_col].to_numpy()))
    df[FAULT_TYPE_COLUMN] = [events[u - 1]["failure"] if o else NO_FAULT_LABEL
                             for u, o in pairs]
    df[FAULT_SEVERITY_COLUMN] = [events[u - 1]["severity"] if o else NO_FAULT_LABEL
                                 for u, o in pairs]
    return df


# ---------------------------------------------------------------------------
# Truncation (test runs end at a real event; mirrors ncmapss._truncate_test, §27)
# ---------------------------------------------------------------------------
def _truncate_test(df_test_full: pd.DataFrame, config: Config
                   ) -> tuple[pd.DataFrame, dict]:
    """Truncate each test run at ``metropt_test_truncation`` of its length; return the
    truncated frame and ``{run: remaining_cycles}``. Same guards as N-CMAPSS/XJTU: keep
    at least ``window_size`` cycles (so the run yields one window) and always drop at
    least one cycle (so the provided RUL is > 0)."""
    frames, rul = [], {}
    for unit_id, unit_df in df_test_full.groupby("unit_number", sort=True):
        unit_df = unit_df.sort_values("time_cycles")
        n = len(unit_df)
        keep = int(np.floor(n * config.metropt_test_truncation))
        keep = max(config.window_size, min(keep, n - 1))
        if keep < 1 or keep >= n:
            raise ValueError(
                f"MetroPT-3 test run {unit_id}: cannot truncate {n} cycles to a valid "
                f"prefix (window_size={config.window_size}); run too short. Lower "
                f"metropt_cycle_minutes to cut finer cycles, or hold out another run.")
        frames.append(unit_df.iloc[:keep])
        rul[int(unit_id)] = n - keep
    return pd.concat(frames, ignore_index=True), rul


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------
def load_metropt(config: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load MetroPT-3 and return the canonical ``(df_train, df_test, rul_truth)``
    triple; ``rul_truth`` is indexed by unit_number (= run number) and holds the
    remaining cycles at each TEST run's last kept cycle."""
    df_all = _load_or_build_aggregate(config)
    obs_col = _event_observed_column()

    run_ids = [int(u) for u in np.sort(df_all["unit_number"].unique())]
    requested = sorted({int(r) for r in config.metropt_test_runs})
    missing = [r for r in requested if r not in run_ids]
    if missing:
        raise ValueError(
            f"metropt_test_runs {missing} do not exist; the record yields runs "
            f"{run_ids} at metropt_cycle_minutes={config.metropt_cycle_minutes} / "
            f"metropt_min_samples_per_cycle={config.metropt_min_samples_per_cycle} "
            f"(run k ends at documented failure event k).")
    censored = [int(u) for u, o in df_all.groupby("unit_number")[obs_col].max().items()
                if o == 0]
    picked_censored = [r for r in requested if r in censored]
    if picked_censored:
        raise ValueError(
            f"metropt_test_runs {picked_censored} are RIGHT-CENSORED runs (they do not "
            f"end at a documented failure -- observation simply stopped), so their true "
            f"RUL at truncation does not exist and they can never be scored under the "
            f"predict-at-last-observed-cycle protocol. Pick from the observed runs "
            f"{[r for r in run_ids if r not in censored]}; censored runs belong in "
            f"TRAIN, where they contribute genuine alarm-negative rows (§54).")
    train_runs = [r for r in run_ids if r not in requested]
    if not requested or not train_runs:
        raise ValueError(
            f"MetroPT-3 split produced an empty train or test set: runs {run_ids}, "
            f"metropt_test_runs={requested}. Hold out at least one run and leave at "
            f"least one for training.")

    is_test = df_all["unit_number"].isin(requested).to_numpy()
    df_train = df_all.loc[~is_test].reset_index(drop=True)
    df_test, rul = _truncate_test(df_all.loc[is_test].reset_index(drop=True), config)
    print(f"[metropt] runs {run_ids} ({len(run_ids) - len(censored)} observed, "
          f"{len(censored)} censored); train runs {train_runs} "
          f"({len(df_train)} cycles), test runs {requested} "
          f"({len(df_test)} cycles kept, RUL {rul})")
    rul_truth = pd.Series(rul, name="rul_truth").sort_index()
    rul_truth.index.name = "unit_number"
    return df_train, df_test, rul_truth
