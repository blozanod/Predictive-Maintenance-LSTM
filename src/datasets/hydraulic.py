"""UCI hydraulic-rig loader -> canonical C-MAPSS-shaped frames + RQ-F action labels.

Dataset (Helwig, Pignanelli & Schuetze 2015, "Condition monitoring of hydraulic
systems"; UCI ML Repository id 447): a real hydraulic test rig run through 2205
constant-load working cycles of 60 s each while four components are degraded on a
controlled schedule. The download is 18 TAB-delimited, HEADER-LESS text files, all with
one row per cycle::

    PS1..PS6.txt, EPS1.txt   6000 columns  (100 Hz x 60 s)   bar / W
    FS1.txt, FS2.txt          600 columns  ( 10 Hz x 60 s)   l/min
    TS1..TS4, VS1, CE, CP, SE  60 columns  (  1 Hz x 60 s)   degC / mm/s / % / kW / %
    profile.txt                 5 columns  (the per-cycle fault annotation)

**Row i of every file is cycle i -- positional alignment ONLY, there is no join key.**
Nothing here ever sorts, reindexes or drops rows in one file independently: one row mask
is derived once and applied to all 18. ``header=None`` is likewise mandatory (pandas'
default ``header='infer'`` would eat cycle 1 as column names and leave 2204 rows).

Adaptation to the pipeline's canonical frame (everything downstream of
``data.load_prepared`` then runs unchanged; same device as N-CMAPSS, CHANGES.md §27):
  * one "cycle" = one 60 s working cycle = one ROW of every sensor file;
  * "sensors"   = per-cycle SUMMARY STATISTICS of each sensor's intra-cycle samples,
    the statistic set chosen by ``config.hydraulic_agg_stats`` (reusing
    ``NCMAPSS_AGG_STAT_SETS``: ``mean_std`` -> 34 channels, ``mean_std_minmax_slope``
    -> 85), emitted in exactly ``config.hydraulic_feature_columns(...)`` order. This is
    what makes the three sampling rates commensurable without resampling anything;
  * one "unit"  = one maximal run of consecutive cycles sharing the same 4-tuple of
    component severities (see "Units" below); ``time_cycles`` restarts at 1 per unit;
  * ``setting_1/2/3 = 0.0`` -- the rig has ONE operating point (constant load), and the
    fault severities are LABELS: putting them in a setting column would leak the RQ-F
    target into the features. ``condition_norm`` therefore resolves auto-OFF.

Secondary labels (the reason this dataset is here -- RESEARCH_PLAN §2 RQ-F / §4). Every
row carries, for each of the four components:
  * ``severity_<comp>`` -- the ORDINAL index into ``HYDRAULIC_SEVERITY_ORDER[comp]``, so
    **0 = healthy and higher = worse for EVERY component**. This is a polarity FIX, not a
    relabelling: cooler / valve / accumulator degrade as their raw value DECREASES
    (100 -> 3 %, 100 -> 73 %, 130 -> 90 bar) while pump leakage degrades as its raw value
    INCREASES (0 -> 2). A single global polarity assumption would invert three of the
    four ladders;
  * ``action_<comp>``   -- the adjust-vs-replace taxonomy (``HYDRAULIC_ACTIONS``):
    0 = none (severity 0), 2 = replace (the component's WORST level), 1 = adjust
    (anything in between). The mapping itself is a config-level DECISION documented in
    src/config.py; ``severity_column`` / ``action_column`` name the columns for
    ``src/taxonomy.py``.

Units = contiguous LABEL BLOCKS (DECISION, uncited). The rig's faults are injected in
large contiguous BLOCKS under a nested factorial schedule (cooler outermost -> accumulator
-> pump -> valve innermost), and the file is NOT shuffled: cooler takes only three runs in
the whole record. A chronological split therefore makes the cooler state perfectly
separable -- a leak dressed up as a result. Segmenting into maximal same-severity runs
(~190 blocks of ~11 cycles, ~136 of them surviving the settling-cycle drop) is the only
segmentation that is both physically honest (one unit = one uninterrupted run of the rig
in one health state) and leakage-safe (no block can straddle a split). Blocks are cut on
the RAW cycle order, BEFORE the unstable rows are dropped, so a dropped settling row
shortens a unit instead of splitting one run into two. **Units are therefore SHORT: set
``window_size`` to ~4-6 for this dataset, not the C-MAPSS default 30.**

``config.hydraulic_drop_unstable`` (default True) drops the cycles the rig itself flags as
not-yet-settled. **Note the shipped polarity: the flag is 1 = NOT stable**, i.e. rows to
discard; 0 = stable.

**RUL WARNING -- read before tabling any RMSE from this dataset.** This is a cyclic,
controlled fault-INJECTION rig, not a run-to-failure fleet: nothing degrades within a
block, and a block ends because the experimenters changed the setting, not because a part
failed. ``time_cycles`` is emitted so the pipeline runs end-to-end and the test blocks are
truncated like every other family's, but the RUL arm is NOT the point of this dataset --
it is the **RQ-F adjust-vs-replace anchor**. Treat its RUL numbers as a plumbing check.

**Comparability warning:** published UCI-447 results are per-cycle CLASSIFICATION
accuracies over a random 2205-cycle split. The numbers produced here come from a
block-level (unit-disjoint) split of cycle-aggregated channels under this repo's
predict-at-the-last-observed-cycle protocol; they are NOT comparable to published figures
and must never share a table with them (role: same-protocol cross-model comparison for
RQ-F, exactly like XJTU-SY and N-CMAPSS).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import (Config, SETTING_COLUMNS, HYDRAULIC_DATASETS,
                      HYDRAULIC_SENSORS, HYDRAULIC_SENSOR_NAMES, HYDRAULIC_N_CYCLES,
                      HYDRAULIC_PROFILE_FILE, HYDRAULIC_PROFILE_COLUMNS,
                      HYDRAULIC_SEVERITY_ORDER, HYDRAULIC_COMPONENTS, HYDRAULIC_ACTIONS,
                      NCMAPSS_AGG_STAT_SETS, hydraulic_feature_columns)
from .base import resolve_data_dir

# Accepted subdirectory name(s) of ``config.data_root`` holding the 18 .txt files (flat).
# The documented layout is ``Hydraulic``; UCI's own download unzips to
# ``condition+monitoring+of+hydraulic+systems`` and users drop it verbatim -- accept both
# (resolve_data_dir picks the first that exists; CHANGES.md §26).
HYDRAULIC_SUBDIR = ("Hydraulic", "condition+monitoring+of+hydraulic+systems")

# Dataset names this family serves (the campaign sweeps these).
DATASETS = tuple(HYDRAULIC_DATASETS)

# Bump whenever the AGGREGATION LOGIC changes, so stale parsed-frame caches
# (cache/hydraulic_agg_v<N>_<stats>.npz) are invalidated -- the role
# CACHE_SCHEMA_VERSION plays for embeddings. The statistic set is in the cache FILENAME
# (each set is a separate, coexisting aggregate), so it is not part of this version.
HYDRAULIC_AGG_VERSION = 1

# One working cycle is 60 s by construction of the rig (Helwig et al. 2015): that is
# exactly why a 100 Hz sensor ships 6000 columns and a 1 Hz sensor 60. Used to put the
# intra-cycle sample index on a TRUE SECONDS axis. DECISION (uncited): the ``slope``
# statistic is therefore per-SECOND, not per-sample, so one slope means the same physical
# quantity at all three sampling rates (and under the down-scaled test fixture).
CYCLE_SECONDS = 60.0

# profile.txt's 5th column, verbatim from the dataset docs. NOTE THE INVERTED POLARITY:
# 1 = conditions were NOT yet stable (a settling cycle, dropped), 0 = stable (kept).
STABLE_FLAG_VALUES = (0, 1)
_STABLE_VALUE = 0

# DECISION (uncited): test blocks are truncated at 0.6 of their length, mirroring
# ``xjtu_test_truncation`` / ``ncmapss_test_truncation`` (§22, §27), so the pipeline's
# predict-at-the-last-observed-cycle protocol applies here too and ``rul_truth`` is the
# number of cycles the block still had left. It is a module constant rather than a Config
# field because this dataset's RUL arm is a plumbing check, not a result (see the
# module docstring) -- promote it to a keyed Config field before sweeping it.
HYDRAULIC_TEST_TRUNCATION = 0.6

# Action-taxonomy codes, resolved from the config-level names so no integer is magic.
_ACTION_NONE = HYDRAULIC_ACTIONS.index("none")
_ACTION_ADJUST = HYDRAULIC_ACTIONS.index("adjust")
_ACTION_REPLACE = HYDRAULIC_ACTIONS.index("replace")


# ---------------------------------------------------------------------------
# Secondary-label column names (src/taxonomy.py names its target through these)
# ---------------------------------------------------------------------------
def _check_component(component: str) -> str:
    if component not in HYDRAULIC_COMPONENTS:
        raise ValueError(
            f"unknown hydraulic component {component!r}; expected one of "
            f"{list(HYDRAULIC_COMPONENTS)}")
    return component


def severity_column(component: str) -> str:
    """Column holding ``component``'s ORDINAL severity (0 = healthy, higher = worse)."""
    return f"severity_{_check_component(component)}"


def action_column(component: str) -> str:
    """Column holding ``component``'s action code (index into ``HYDRAULIC_ACTIONS``:
    none / adjust / replace) -- the RQ-F few-shot probe's target."""
    return f"action_{_check_component(component)}"


SEVERITY_COLUMNS = [severity_column(c) for c in HYDRAULIC_COMPONENTS]
ACTION_COLUMNS = [action_column(c) for c in HYDRAULIC_COMPONENTS]
# Every non-canonical column the loader adds, in emission order.
HYDRAULIC_LABEL_COLUMNS = [c for comp in HYDRAULIC_COMPONENTS
                           for c in (severity_column(comp), action_column(comp))]


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
def _sensor_path(root: Path, name: str) -> Path:
    return root / f"{name}.txt"


def _has_profile(root: Path) -> bool:
    return (root / HYDRAULIC_PROFILE_FILE).is_file()


def _descend_to_data(root: Path, verbose: bool = True) -> Path:
    """Return the directory that directly holds the 18 .txt files.

    If ``root`` already holds ``profile.txt``, return it. Otherwise scan ``root``'s
    IMMEDIATE subdirectories (depth-1 only, no recursive walk) for one that does --
    absorbing the common zip-in-a-folder nesting UCI downloads produce. If none
    qualifies, return ``root`` unchanged so the caller's "not found" error names the
    documented path (CHANGES.md §26)."""
    if _has_profile(root) or not root.is_dir():
        return root
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        if _has_profile(child):
            if verbose:
                print(f"[hydraulic] descending into nested folder {child.name!r} "
                      f"({HYDRAULIC_PROFILE_FILE} found one level down)")
            return child
    return root


def _resolve_dir(config: Config) -> Path:
    return _descend_to_data(resolve_data_dir(config, HYDRAULIC_SUBDIR))


def _require_file(path: Path, what: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"missing hydraulic {what} file {path.name!r} under {path.parent}; the UCI "
            f"'Condition monitoring of hydraulic systems' (id 447) download ships "
            f"{len(HYDRAULIC_SENSOR_NAMES)} tab-delimited sensor files "
            f"({', '.join(HYDRAULIC_SENSOR_NAMES)}) plus {HYDRAULIC_PROFILE_FILE} -- "
            f"unzip all of them into that folder (RESEARCH_PLAN §3).")
    return path


def is_available(config: Config) -> bool:
    """Cheap on-disk check: profile.txt AND all 17 sensor files present. (The campaign
    skips unavailable datasets with a notice, CHANGES.md §24.)"""
    root = _descend_to_data(resolve_data_dir(config, HYDRAULIC_SUBDIR), verbose=False)
    if not _has_profile(root):
        return False
    return all(_sensor_path(root, name).is_file() for name in HYDRAULIC_SENSOR_NAMES)


# ---------------------------------------------------------------------------
# Reading + geometry validation (fail loud; never silently adapt)
# ---------------------------------------------------------------------------
def _read_sensor_matrix(path: Path) -> np.ndarray:
    """One sensor file -> ``(n_cycles, n_samples)`` float32.

    ``header=None`` is MANDATORY (the files carry no header row; pandas' default would
    consume cycle 1 as column names) and ``dtype=np.float32`` halves the peak footprint
    of the 100 Hz files (2205 x 6000 = 53 MB each, read one at a time). The shipped files
    have no missing values, so a non-finite entry means a corrupt/partial download and is
    an error rather than something to impute."""
    _require_file(path, "sensor")
    frame = pd.read_csv(path, sep="\t", header=None, dtype=np.float32)
    values = frame.to_numpy(np.float32)
    bad = int((~np.isfinite(values)).sum())
    if bad:
        raise ValueError(
            f"{path.name}: {bad} non-finite value(s) in a file that ships with none "
            f"(expected {values.size} finite readings); re-download the dataset rather "
            f"than imputing them.")
    return values


def _read_profile(path: Path) -> np.ndarray:
    """profile.txt -> ``(n_cycles, 5)`` int64 in ``HYDRAULIC_PROFILE_COLUMNS`` order."""
    _require_file(path, "annotation")
    frame = pd.read_csv(path, sep="\t", header=None, dtype=np.int64)
    if frame.shape[1] != len(HYDRAULIC_PROFILE_COLUMNS):
        raise ValueError(
            f"{path.name}: expected {len(HYDRAULIC_PROFILE_COLUMNS)} tab-separated "
            f"integer columns {list(HYDRAULIC_PROFILE_COLUMNS)}, observed "
            f"{frame.shape[1]}. The annotation is positional -- a different width means "
            f"a different file, not a file to reorder.")
    return frame.to_numpy(np.int64)


def _validate_geometry(shapes: dict, profile_rows: int) -> int:
    """Check the on-disk geometry against the documented schema; return the sample-rate
    SCALE FACTOR ``k`` (1 for the shipped dataset).

    Three things are asserted, each naming the expected AND the observed value:

    1. **Every file has the same number of rows** (row i of every file is cycle i --
       positional alignment only, so a disagreement silently mis-pairs sensors with
       labels and must never be "fixed" by truncation).
    2. **Each sensor's width matches its sampling rate**, up to ONE scale factor shared
       by all 17 files: ``HYDRAULIC_SENSORS[name][0] == observed * k``. A real download
       gives ``k = 1``; a uniformly down-scaled fixture (the CPU tests) gives ``k > 1``
       while still proving every rate ratio is intact. A per-file mismatch -- the shape
       drift this guard exists for -- cannot produce a common ``k`` and raises.
    3. **A full-rate dataset has exactly ``HYDRAULIC_N_CYCLES`` cycles** (``k == 1``), so
       a truncated download fails loud instead of quietly training on a prefix.
    """
    rows = {name: n_rows for name, (n_rows, _) in shapes.items()}
    distinct = set(rows.values()) | {profile_rows}
    if len(distinct) != 1:
        raise ValueError(
            "hydraulic files disagree on their row count; row i of every file IS cycle i "
            "(positional alignment only -- there is no join key), so they must match "
            f"exactly.\n  expected : one common row count\n"
            f"  observed : {dict(sorted(rows.items()))} + "
            f"{HYDRAULIC_PROFILE_FILE}={profile_rows}")
    n_cycles = distinct.pop()

    scales = {}
    for name, (_, n_samples) in shapes.items():
        expected = HYDRAULIC_SENSORS[name][0]
        if expected % n_samples:
            raise ValueError(
                f"{name}.txt has {n_samples} columns, which is not a whole down-scaling "
                f"of its documented width {expected} "
                f"({expected / CYCLE_SECONDS:g} Hz x {CYCLE_SECONDS:g} s); the column "
                f"count IS the sampling rate, so this file is not what it claims to be.")
        scales[name] = expected // n_samples
    if len(set(scales.values())) != 1:
        documented = {name: HYDRAULIC_SENSORS[name][0] for name in sorted(shapes)}
        observed = {name: shapes[name][1] for name in sorted(shapes)}
        raise ValueError(
            "hydraulic sensor widths do not share one sampling-rate scale: the 100/10/1 "
            "Hz ratio is broken.\n"
            f"  expected widths : {documented}\n"
            f"  observed widths : {observed}")
    scale = scales[HYDRAULIC_SENSOR_NAMES[0]]

    if scale == 1 and n_cycles != HYDRAULIC_N_CYCLES:
        raise ValueError(
            f"full-rate hydraulic files must hold exactly {HYDRAULIC_N_CYCLES} cycles "
            f"(the shipped record), observed {n_cycles}; the download is truncated.")
    return scale


# ---------------------------------------------------------------------------
# Per-cycle aggregation (intra-cycle samples -> channels)
# ---------------------------------------------------------------------------
def _aggregate_sensor(values: np.ndarray, stats: tuple) -> np.ndarray:
    """``(n_cycles, n_samples)`` -> ``(n_cycles, len(stats))``, one column per statistic
    in ``stats`` order.

    Conventions (mirroring the N-CMAPSS aggregation, §27/§53): ``std`` is the sample std
    (ddof=1) and a single-sample cycle gets 0.0 rather than NaN; ``slope`` is the
    least-squares slope against the intra-cycle time axis in SECONDS, via the algebraic
    identity ``cov(t, x) / var(t)`` (no per-row Python loop), and is likewise 0.0 when a
    single sample leaves it undefined."""
    n_cycles, n_samples = values.shape
    seconds = np.arange(n_samples, dtype=np.float64) * (CYCLE_SECONDS / n_samples)
    centered = seconds - seconds.mean()
    denom = float(centered @ centered)
    columns = []
    for stat in stats:
        if stat == "mean":
            columns.append(values.mean(axis=1, dtype=np.float64))
        elif stat == "std":
            columns.append(values.std(axis=1, ddof=1, dtype=np.float64)
                           if n_samples > 1 else np.zeros(n_cycles))
        elif stat == "min":
            columns.append(values.min(axis=1).astype(np.float64))
        elif stat == "max":
            columns.append(values.max(axis=1).astype(np.float64))
        else:   # "slope" (the stat sets are fixed in config.NCMAPSS_AGG_STAT_SETS)
            columns.append(values.astype(np.float64) @ centered / denom
                           if denom > 0 else np.zeros(n_cycles))
    return np.column_stack(columns)


def _build_aggregate(root: Path, agg_stats: str) -> tuple:
    """Parse the 18 files once -> ``(agg (n_cycles, n_channels) float64, profile)``.

    Sensors are read and reduced ONE FILE AT A TIME (the shipped text is 556 MB and would
    be ~740 MB as float64 in memory); only the tiny per-cycle statistics are kept. The
    geometry is validated BEFORE anything is concatenated, so a mis-shaped file raises
    instead of being silently broadcast against the others."""
    profile = _read_profile(root / HYDRAULIC_PROFILE_FILE)
    stats = NCMAPSS_AGG_STAT_SETS[agg_stats]
    shapes, blocks = {}, []
    for name in HYDRAULIC_SENSOR_NAMES:
        values = _read_sensor_matrix(_sensor_path(root, name))
        shapes[name] = values.shape
        blocks.append(_aggregate_sensor(values, stats))
        del values
    _validate_geometry(shapes, profile.shape[0])
    return np.concatenate(blocks, axis=1), profile


def _source_fingerprint(root: Path) -> str:
    """Short identity hash of the 18 shipped files: each ``(name, size)``.

    Without it the cache filename is a constant across every config but ``agg_stats``
    (this family has exactly one dataset name), so pointing the loader at a DIFFERENT
    hydraulic directory silently returns the previously cached dataset -- bypassing every
    geometry guard, including the 2205-cycle "fail loud on a truncated download" check.
    Sizes rather than content: the files total 556 MB and (name, size) is the standard
    cheap staleness triple for a read-only scientific download."""
    parts = []
    for name in list(HYDRAULIC_SENSOR_NAMES) + [HYDRAULIC_PROFILE_FILE.split(".")[0]]:
        path = root / (HYDRAULIC_PROFILE_FILE if name == "profile" else f"{name}.txt")
        parts.append(f"{path.name}:{path.stat().st_size if path.exists() else -1}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:8]


def _agg_cache_path(config: Config, root: Path) -> Path:
    """Aggregate cache path. Carries the logic version, the stat set AND a fingerprint of
    the source files, so two different hydraulic directories can never collide on one
    aggregate (location-independent: the fingerprint is content identity, not a path)."""
    return (Path(config.cache_dir)
            / f"hydraulic_agg_v{HYDRAULIC_AGG_VERSION}"
              f"_{config.hydraulic_agg_stats}"
              f"_f{_source_fingerprint(root)}.npz")


def _load_or_build_aggregate(config: Config, verbose: bool = True) -> tuple:
    """Return ``(agg, profile)`` for ALL cycles, through the versioned parsed-frame cache.

    Parsing 556 MB of text is minutes; the aggregate is 2205 x ~34 floats. The cache
    holds the UNFILTERED, UNSPLIT arrays, so ``hydraulic_drop_unstable`` /
    ``hydraulic_test_fraction`` re-apply from config without re-parsing."""
    root = _resolve_dir(config)
    cache_path = _agg_cache_path(config, root)
    columns = hydraulic_feature_columns(config.hydraulic_agg_stats)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as npz:
            agg, profile = npz["agg"], npz["profile"]
        if agg.shape != (profile.shape[0], len(columns)):
            raise ValueError(
                f"cached hydraulic aggregate {cache_path.name} has shape {agg.shape}, "
                f"expected ({profile.shape[0]}, {len(columns)}) for agg_stats="
                f"{config.hydraulic_agg_stats!r}; the cache is stale or corrupt -- "
                f"delete it (and bump HYDRAULIC_AGG_VERSION if the logic changed).")
        if verbose:
            print(f"[hydraulic] loaded cached aggregate {cache_path.name} "
                  f"({agg.shape[0]} cycles x {agg.shape[1]} channels)")
    else:
        if verbose:
            print(f"[hydraulic] parsing {len(HYDRAULIC_SENSOR_NAMES)} sensor files under "
                  f"{root} ({CYCLE_SECONDS:g} s cycles -> per-cycle aggregate; "
                  f"stats={config.hydraulic_agg_stats})...")
        agg, profile = _build_aggregate(root, config.hydraulic_agg_stats)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Store float32 AND return it, so a cold run and a warm run of the same config
        # produce byte-identical windows -- otherwise a resume after Stage A would train
        # on values that differ from the ones the cached embeddings were computed from.
        agg = agg.astype(np.float32)
        np.savez(cache_path, agg=agg, profile=profile.astype(np.int64))
        if verbose:
            print(f"[hydraulic] parsed {agg.shape[0]} cycles x {agg.shape[1]} channels; "
                  f"cached -> {cache_path.name}")
    return np.asarray(agg, np.float64), np.asarray(profile, np.int64)


# ---------------------------------------------------------------------------
# Labels: ordinal severities, the action taxonomy, the stability flag
# ---------------------------------------------------------------------------
def _severity_ordinals(profile: np.ndarray) -> np.ndarray:
    """``(n_cycles, 4)`` int64 ordinal severities in ``HYDRAULIC_COMPONENTS`` order.

    Each raw annotation value is looked up in that component's own HEALTHY->WORST ladder
    (``HYDRAULIC_SEVERITY_ORDER``), which is what puts all four components on ONE polarity
    (0 = healthy, higher = worse) despite three of them counting DOWN in raw units and the
    pump counting UP. A value outside a component's documented set raises: it means the
    annotation schema changed, and guessing where it sits on the ladder would silently
    invert or flatten the RQ-F labels."""
    columns = []
    for component in HYDRAULIC_COMPONENTS:
        raw = profile[:, HYDRAULIC_PROFILE_COLUMNS.index(component)]
        ladder = HYDRAULIC_SEVERITY_ORDER[component]
        lookup = {int(value): rank for rank, value in enumerate(ladder)}
        unknown = sorted(set(int(v) for v in raw) - set(lookup))
        if unknown:
            raise ValueError(
                f"{HYDRAULIC_PROFILE_FILE}: {component} annotation holds value(s) "
                f"{unknown} that are not in its documented set {list(ladder)} "
                f"(healthy -> worst). The severity ladder is ORDERED -- an unknown level "
                f"has no defined rank, so it cannot be mapped.")
        columns.append(np.array([lookup[int(v)] for v in raw], np.int64))
    return np.column_stack(columns)


def _action_codes(severity: np.ndarray, component: str) -> np.ndarray:
    """Ordinal severities of ONE component -> action codes (``HYDRAULIC_ACTIONS`` index):
    0 = none (healthy), 2 = replace (the component's worst level), 1 = adjust (in
    between). The thresholding itself is the config-level DECISION (src/config.py)."""
    worst = len(HYDRAULIC_SEVERITY_ORDER[component]) - 1
    codes = np.full(severity.shape, _ACTION_ADJUST, np.int64)
    codes[severity == 0] = _ACTION_NONE
    codes[severity == worst] = _ACTION_REPLACE
    return codes


def _stable_mask(profile: np.ndarray) -> np.ndarray:
    """Boolean keep-mask from the stability flag. **The shipped polarity is inverted with
    respect to the column's name: 1 = NOT stable** (a settling cycle), 0 = stable. The
    value set is validated whether or not the caller drops anything, so a schema change
    can never pass silently."""
    flags = profile[:, HYDRAULIC_PROFILE_COLUMNS.index("stable_flag")]
    unknown = sorted(set(int(v) for v in flags) - set(STABLE_FLAG_VALUES))
    if unknown:
        raise ValueError(
            f"{HYDRAULIC_PROFILE_FILE}: stable_flag holds value(s) {unknown}; expected "
            f"only {list(STABLE_FLAG_VALUES)} (1 = NOT stable, 0 = stable).")
    return flags == _STABLE_VALUE


# ---------------------------------------------------------------------------
# Units = contiguous label blocks -> the canonical frame
# ---------------------------------------------------------------------------
def _block_ids(severity: np.ndarray) -> np.ndarray:
    """Maximal runs of consecutive cycles sharing the same severity 4-tuple, numbered
    0.. in cycle order (the "unit" of this dataset -- see the module docstring)."""
    changed = np.any(severity[1:] != severity[:-1], axis=1)
    return np.concatenate([[0], np.cumsum(changed)]).astype(np.int64)


def _canonical_frame(agg: np.ndarray, profile: np.ndarray,
                     config: Config) -> pd.DataFrame:
    """Aggregate + annotation -> the canonical frame (one row per RETAINED cycle).

    Blocks are cut on the RAW cycle order first, then the unstable rows are dropped from
    within them, then ``time_cycles`` is renumbered 1..n over what survives -- so a
    settling row shortens a unit instead of splitting one physical run in two. A block
    that loses ALL of its rows (the 211-cycle warm-ups at the start of each cooler regime
    swallow whole blocks) contributes no unit at all; ``unit_number`` numbers the
    SURVIVING blocks 1..U in cycle order."""
    severity = _severity_ordinals(profile)
    stable = _stable_mask(profile)          # validated either way (see _stable_mask)
    keep = stable if config.hydraulic_drop_unstable else np.ones(len(profile), bool)
    blocks = _block_ids(severity)

    selected, units, cycles = [], [], []
    for block in range(int(blocks[-1]) + 1):
        rows = np.flatnonzero((blocks == block) & keep)
        if rows.size == 0:
            continue
        selected.append(rows)
        units.append(np.full(rows.size, len(selected), np.int64))
        cycles.append(np.arange(1, rows.size + 1, dtype=np.int64))
    if not selected:
        raise ValueError(
            f"every one of {len(profile)} hydraulic cycles was discarded as not-stable "
            f"(stable_flag == 1); set hydraulic_drop_unstable=False or check "
            f"{HYDRAULIC_PROFILE_FILE} -- the flag is 1 = NOT stable.")
    rows = np.concatenate(selected)

    frame = pd.DataFrame(agg[rows],
                         columns=hydraulic_feature_columns(config.hydraulic_agg_stats))
    frame.insert(0, "unit_number", np.concatenate(units))
    frame.insert(1, "time_cycles", np.concatenate(cycles))
    for offset, column in enumerate(SETTING_COLUMNS):
        # One operating point; the severities are LABELS and must not leak in here.
        frame.insert(2 + offset, column, 0.0)
    for index, component in enumerate(HYDRAULIC_COMPONENTS):
        component_severity = severity[rows, index]
        frame[severity_column(component)] = component_severity
        frame[action_column(component)] = _action_codes(component_severity, component)
    return frame


# ---------------------------------------------------------------------------
# Split: stratified, deterministic, block-level
# ---------------------------------------------------------------------------
def _select_test_units(frame: pd.DataFrame, fraction: float, min_cycles: int,
                       taxonomy_component: str = "valve",
                       verbose: bool = True) -> np.ndarray:
    """Held-out unit ids: a ``fraction`` of BLOCKS, stratified by
    ``(cooler severity, <taxonomy component> severity)``.

    DECISION (uncited). The block order is a nested factorial with COOLER OUTERMOST --
    only three cooler runs exist in the whole record -- so a tail/random-tail split hands
    the test set exactly one cooler regime and makes that component trivially separable.
    Stratifying on cooler ALONE is not enough either, and the failure is subtle:
    systematic sampling inside a stratum ALIASES against the inner factorial. Valve has
    period 4 in block index, so at ``fraction = 0.25`` every span midpoint lands on the
    same residue mod 4 and the test set collapses onto ONE valve level -- destroying the
    very RQ-F probe this dataset exists for, silently, at the most natural sweep value of
    a field meant to be swept. The strata are therefore the CROSS of the outermost factor
    and the component the RQ-F probe targets, which makes every level of that component
    present on both sides by construction. The split then:

      * takes only blocks of at least ``min_cycles`` cycles as ELIGIBLE for test. A block
        too short to be truncated into (a window, a positive RUL) cannot serve as a test
        unit, and short blocks are structural here -- the 211-cycle warm-ups clip whichever
        block they land in. Such a block is not dropped: it stays in TRAIN, where a unit
        shorter than the window simply yields no windows;
      * within a stratum, takes blocks by SYSTEMATIC sampling at the requested rate
        (midpoints of ``k`` equal spans), so the extreme first/last blocks of a regime are
        never systematically preferred;
      * keeps at least one train block per stratum (``k <= n - 1``) and takes at least one
        test block (``k >= 1``) -- unless the stratum holds a single eligible block, which
        stays in TRAIN so no stratum is represented ONLY in test. Skipped strata are
        REPORTED, not silently dropped.

    No RNG is involved: the split is a pure function of the block layout, the fraction
    (a keyed Config field), the window size and the targeted component."""
    if not 0.0 < fraction < 1.0:
        raise ValueError(
            f"hydraulic_test_fraction must be in (0, 1) -- it is the fraction of label "
            f"BLOCKS held out -- got {fraction}.")
    by_unit = frame.groupby("unit_number", sort=True)
    lengths = by_unit.size()
    eligible = (lengths >= min_cycles).to_numpy()
    unit_ids = lengths.index.to_numpy()[eligible]
    cooler = by_unit[severity_column("cooler")].first().to_numpy()[eligible]
    target = by_unit[severity_column(taxonomy_component)].first().to_numpy()[eligible]

    def _sample(strata: np.ndarray):
        """Systematic pick per stratum; returns (chosen blocks, skipped strata)."""
        chosen, skipped = [], []
        for level in np.unique(strata, axis=0):
            mask = np.all(strata == level, axis=1)
            stratum = unit_ids[mask]                  # ascending == block order
            n_blocks = stratum.size
            if n_blocks < 2:
                skipped.append((tuple(int(v) for v in level), int(n_blocks)))
                continue                              # single-block stratum -> train
            n_test = min(max(int(round(n_blocks * fraction)), 1), n_blocks - 1)
            spans = (np.arange(n_test) + 0.5) * n_blocks / n_test   # span midpoints
            chosen.append(stratum[np.floor(spans).astype(np.int64)])
        return chosen, skipped

    target_by_unit = dict(zip(unit_ids, target))
    all_levels = {int(v) for v in target}

    def _coverage(test_units: np.ndarray) -> tuple[set, set]:
        """(levels of the RQ-F target on the TEST side, on the TRAIN side)."""
        held = set(int(u) for u in test_units)
        return ({int(target_by_unit[u]) for u in test_units},
                {int(v) for u, v in target_by_unit.items() if int(u) not in held})

    def _scoreable(test_units: np.ndarray) -> bool:
        """Can the RQ-F probe be scored on this split? It needs >= 2 target levels on
        the test side and a non-empty train side -- unless the record genuinely holds
        only one level, in which case no split can do better."""
        if len(all_levels) < 2:
            return True
        test_levels, train_levels = _coverage(test_units)
        return len(test_levels) >= 2 and bool(train_levels)

    # Strategy order. The CROSS is preferred (it controls the outermost leak-prone factor
    # AND the RQ-F target); if it cannot produce a SCOREABLE split -- not merely a
    # non-empty one -- fall back to the TARGET alone. Never to cooler alone: cooler-only
    # is precisely the stratification whose systematic sampling aliases against the inner
    # factorial and collapses the test set onto a single target level.
    strategies = ((f"(cooler, {taxonomy_component})", np.stack([cooler, target], axis=1)),
                  (f"{taxonomy_component}-only", target[:, None]))
    picked = None
    for name, strata in strategies:
        chosen, skipped = _sample(strata)
        if not chosen:
            continue
        candidate = np.sort(np.concatenate(chosen))
        if _scoreable(candidate):
            picked = (name, candidate, skipped)
            break
        if verbose:
            test_levels, _train = _coverage(candidate)
            print(f"[hydraulic] NOTE: {name} stratification yields a test set covering "
                  f"only severity_{taxonomy_component} level(s) {sorted(test_levels)} of "
                  f"{sorted(all_levels)} (systematic sampling aliased against the nested "
                  f"factorial); trying a coarser stratification.")
    if picked is None:
        test_levels, train_levels = ((set(), set()) if not chosen
                                     else _coverage(np.sort(np.concatenate(chosen))))
        raise ValueError(
            f"the hydraulic split could not produce a scoreable test set: of "
            f"{len(lengths)} label block(s) (lengths {int(lengths.min())}.."
            f"{int(lengths.max())} cycles) only {len(unit_ids)} reach the {min_cycles} "
            f"cycles a truncated test unit needs, and no stratification puts >= 2 of the "
            f"severity_{taxonomy_component} levels {sorted(all_levels)} on the test side "
            f"(best attempt: test {sorted(test_levels)}, train {sorted(train_levels)}). "
            f"Lower window_size for this dataset (its blocks are ~10 cycles), adjust "
            f"hydraulic_test_fraction (currently {fraction}), or set "
            f"hydraulic_drop_unstable=False.")
    name, test_units, skipped = picked
    if verbose:
        print(f"[hydraulic] test split stratified by {name}")
        if skipped:
            # A silently-dropped stratum reads exactly like one that was represented.
            print(f"[hydraulic] NOTE: stratum/strata {skipped} hold fewer than 2 "
                  f"eligible blocks and contribute NO test block (they stay wholly in "
                  f"train); the test split does not cover them.")
    return test_units


def _truncate_test(df_test_full: pd.DataFrame, config: Config) -> tuple:
    """Truncate each test block at ``HYDRAULIC_TEST_TRUNCATION`` of its length; return the
    truncated frame and ``{unit: remaining_cycles}``. Same guards as XJTU/N-CMAPSS: at
    least ``window_size`` cycles kept (so the unit yields a window) and at least one cycle
    dropped (so the provided RUL is > 0). ``_select_test_units`` only ever hands over
    blocks long enough for both, so the raise here is a defensive backstop -- it fires if
    a caller truncates a frame the eligibility rule never approved."""
    frames, rul = [], {}
    for unit_id, unit_df in df_test_full.groupby("unit_number", sort=True):
        unit_df = unit_df.sort_values("time_cycles")
        n_cycles = len(unit_df)
        keep = int(np.floor(n_cycles * HYDRAULIC_TEST_TRUNCATION))
        keep = max(config.window_size, min(keep, n_cycles - 1))
        if keep < 1 or keep >= n_cycles:
            raise ValueError(
                f"hydraulic test unit {unit_id}: cannot truncate {n_cycles} cycles to a "
                f"valid prefix (window_size={config.window_size}); label blocks here are "
                f"short (~10 cycles) -- lower window_size for this dataset.")
        frames.append(unit_df.iloc[:keep])
        rul[int(unit_id)] = n_cycles - keep
    return pd.concat(frames, ignore_index=True), rul


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------
def load_hydraulic(config: Config) -> tuple:
    """Load the UCI hydraulic rig and return the canonical ``(df_train, df_test,
    rul_truth)`` triple.

    Both frames carry the canonical columns (``unit_number``, ``time_cycles``, the three
    settings, then ``config.default_sensor_columns()``) plus the RQ-F secondary labels
    ``severity_<comp>`` / ``action_<comp>`` for all four components.  ``rul_truth`` is
    indexed by unit_number = cycles remaining at each TEST block's last kept cycle -- and
    is a plumbing quantity, not a degradation measurement (see the module docstring)."""
    agg, profile = _load_or_build_aggregate(config)
    frame = _canonical_frame(agg, profile, config)
    # A test block must survive truncation: >= window_size kept cycles AND >= 1 dropped.
    test_units = _select_test_units(frame, config.hydraulic_test_fraction,
                                    config.window_size + 1,
                                    config.hydraulic_taxonomy_component)
    is_test = frame["unit_number"].isin(test_units).to_numpy()
    df_train = frame.loc[~is_test].reset_index(drop=True)
    df_test, rul = _truncate_test(frame.loc[is_test].reset_index(drop=True), config)
    print(f"[hydraulic] {frame['unit_number'].nunique()} label blocks -> "
          f"{df_train['unit_number'].nunique()} train / {len(test_units)} test, "
          f"stratified by (cooler, {config.hydraulic_taxonomy_component}) "
          f"(blocks under {config.window_size + 1} cycles stay in train -- too short "
          f"to truncate)")
    rul_truth = pd.Series(rul, name="rul_truth").sort_index()
    rul_truth.index.name = "unit_number"

    # NO BLOCK ENDS IN A FAILURE. A block ends because the experimenter changed the
    # set-point, so every unit is right-censored in the literal sense: nothing was
    # observed to fail. Emitting this makes `data.add_alarm_label` treat the rows
    # honestly, and `config.is_classification_dataset()` routes the campaign to the RQ-F
    # probe instead of a RUL sweep (CHANGES.md §55).
    from ..data import EVENT_OBSERVED_COLUMN
    for out in (df_train, df_test):
        out[EVENT_OBSERVED_COLUMN] = 0.0

    # A degenerate RUL target is the EXPECTED outcome here, not a bug -- uniformly-sized
    # blocks truncated at a fixed fraction give every test unit the same remaining count,
    # so the predict-the-mean floor scores a perfect 0.0 that no model can beat. Say so
    # loudly rather than letting an RMSE column be tabled beside C-MAPSS's.
    if len(rul_truth) and rul_truth.nunique() == 1:
        block_lengths = frame.groupby("unit_number").size()
        print(f"[hydraulic] WARNING: rul_truth is CONSTANT ({int(rul_truth.iloc[0])} "
              f"cycles for all {len(rul_truth)} test blocks) because the blocks are "
              f"uniformly sized (lengths {int(block_lengths.min())}.."
              f"{int(block_lengths.max())}). Its RUL/NASA numbers are degenerate and "
              f"must NOT be tabled against a run-to-failure dataset -- this dataset's "
              f"deliverable is the RQ-F taxonomy probe (src/taxonomy.py).")
    return df_train, df_test, rul_truth
