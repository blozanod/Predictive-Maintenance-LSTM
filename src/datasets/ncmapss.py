"""N-CMAPSS run-to-failure loader -> canonical C-MAPSS-shaped frames.

Dataset (Arias Chao et al. 2021, "Aircraft Engine Run-to-Failure Dataset under Real
Flight Conditions"; NASA Prognostics Data Repository): a small fleet of turbofan
engines simulated with C-MAPSS under REAL recorded flight profiles, one ``.h5`` file
per sub-dataset (DS01..DS08d). Each file holds, for dev and test splits:
  * ``W``   -- 4 flight-condition scenario descriptors (alt, Mach, TRA, T2);
  * ``X_s`` -- 14 measured sensors (the condition-monitoring signals);
  * ``X_v``, ``T``, ``Y`` -- virtual sensors, health-parameter ground truth, and
    per-row RUL. These are SIMULATION ORACLES and are never read here (RUL is
    re-derived from cycle counts by ``data.add_train_rul``, exactly as for C-MAPSS);
  * ``A``   -- auxiliary: ``unit, cycle, Fc`` (flight class 1/2/3), ``hs`` (health).
The raw data is 1 Hz WITHIN each flight; one flight = one cycle, thousands of rows.

Adaptation to the pipeline's canonical (cycle-level) frame -- the whole point, so
everything downstream of ``data.load_prepared`` runs unchanged (mirrors the XJTU
indicator-trend design, CHANGES.md §22, §27):
  * one "cycle" = one flight; ``time_cycles`` = the flight index (``A.cycle``);
  * one "unit"  = one engine (``A.unit``; dev/test ids are disjoint within a file);
  * "sensors"   = per-cycle SUMMARY STATISTICS of the 18 raw channels
    (``mean`` + ``std`` of each W and X_s channel) plus ``cycle_len_s`` = the number
    of 1 Hz rows in the flight (observable flight duration) -- ``NCMAPSS_FEATURE_COLUMNS``,
    37 channels. NOT the raw 1 Hz sub-cycle series (Chronos-2 contexts are per-cycle
    multivariate series; cycle aggregation is the standard cycle-level formulation);
  * ``setting_1`` = ``Fc`` (flight class, constant per unit), ``setting_2/3`` = 0.

Condition normalization resolves auto-OFF for N-CMAPSS (flight conditions are
CONTINUOUS, not a discrete grid, and the aggregates already carry them as channels).
Because ``setting_1 = Fc`` you MAY force ``condition_norm=True`` for per-flight-class
normalization; the default stays OFF.

Split & test protocol (DECISION, uncited -- CHANGES.md §27): train = the file's
``*_dev`` units (full run-to-failure); test = the file's ``*_test`` units, TRUNCATED
at ``config.ncmapss_test_truncation`` of their life (same device as XJTU, §22) so the
pipeline's predict-at-last-observed-cycle protocol applies; provided RUL = remaining
cycles at truncation. ``max_rul`` (default 125) is effectively INACTIVE here (N-CMAPSS
end-of-life is ~60-100 cycles, so the piecewise cap never binds -> plain linear RUL,
which matches N-CMAPSS community practice).

**Comparability warning:** published N-CMAPSS RMSEs are computed on 1 Hz sub-cycle
windows over full test trajectories. These cycle-aggregated, truncation-protocol
numbers are NOT comparable to them and must never share a table (RESEARCH_PLAN role:
same-protocol cross-model comparison for RQ1/RQ4, exactly like XJTU-SY).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import (Config, INDEX_COLUMNS, SETTING_COLUMNS,
                      NCMAPSS_DATASETS, NCMAPSS_W_VARS, NCMAPSS_XS_VARS,
                      NCMAPSS_FEATURE_COLUMNS)
from .base import resolve_data_dir

# Accepted subdirectory name(s) of ``config.data_root`` holding the .h5 files (flat).
NCMAPSS_SUBDIR = ("N-CMAPSS",)

# Dataset names this family serves (the campaign sweeps these). DSALL is the combined
# all-files fleet (§28).
DATASETS = tuple(NCMAPSS_DATASETS)

# Bump whenever the AGGREGATION LOGIC changes, so stale per-file aggregate caches
# (cache/ncmapss_agg_<ds>_v<N>.npz) are invalidated -- the role CACHE_SCHEMA_VERSION
# plays for embeddings. The aggregate is otherwise config-INDEPENDENT (no knobs).
NCMAPSS_AGG_VERSION = 1

# Canonical numeric-matrix column order used by the aggregate cache.
_CANON_COLUMNS = list(INDEX_COLUMNS) + list(SETTING_COLUMNS) + list(NCMAPSS_FEATURE_COLUMNS)

_CONFIG_VARS = list(NCMAPSS_W_VARS) + list(NCMAPSS_XS_VARS)   # W then X_s, config order


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
def _resolve_dir(config: Config) -> Path:
    return resolve_data_dir(config, NCMAPSS_SUBDIR)


def _glob_for(ds: str) -> str:
    # Filenames carry a generation suffix (e.g. N-CMAPSS_DS02-006.h5); glob it.
    return f"N-CMAPSS_{ds}*.h5"


def _find_h5(root: Path, ds: str) -> Path:
    matches = sorted(root.glob(_glob_for(ds)))
    if not matches:
        raise FileNotFoundError(
            f"no N-CMAPSS file matching {_glob_for(ds)!r} under {root}; download the "
            f"dataset's .h5 files into that folder (RESEARCH_PLAN §3).")
    if len(matches) > 1:
        raise ValueError(
            f"ambiguous N-CMAPSS files for {ds}: {[m.name for m in matches]}; keep "
            f"exactly one file per sub-dataset in {root}.")
    return matches[0]


def is_available(config: Config) -> bool:
    """Cheap on-disk check. Per-file: at least one matching .h5. DSALL: at least two
    ``N-CMAPSS_DS*.h5`` present (a 1-file union is just that file -- §28)."""
    root = _resolve_dir(config)
    if not root.is_dir():
        return False
    if config.dataset == "DSALL":
        return len(sorted(root.glob("N-CMAPSS_DS*.h5"))) >= 2
    return bool(sorted(root.glob(_glob_for(config.dataset))))


# ---------------------------------------------------------------------------
# Aggregation (1 Hz rows -> per-cycle summary statistics)
# ---------------------------------------------------------------------------
def _decode_vars(arr) -> list[str]:
    """Decode an h5 ``*_var`` byte-string array (any of shape (n,), (n,1), (1,n)) to a
    flat list of stripped python strings."""
    return [str(x).strip() for x in np.array(arr).astype("U20").ravel()]


def _aggregate_split(W, X_s, A, w_var, xs_var) -> np.ndarray:
    """Aggregate one split's 1 Hz rows to the canonical per-cycle numeric matrix
    ``(n_cycles, len(_CANON_COLUMNS))``. Column order == ``_CANON_COLUMNS``."""
    file_vars = list(w_var) + list(xs_var)
    if set(file_vars) != set(_CONFIG_VARS):
        raise ValueError(
            "N-CMAPSS channel names in the file do not match the config schema.\n"
            f"  file W+X_s : {sorted(file_vars)}\n"
            f"  config     : {sorted(_CONFIG_VARS)}\n"
            "Update NCMAPSS_W_VARS/NCMAPSS_XS_VARS in src/config.py to match the file "
            "(and bump NCMAPSS_AGG_VERSION); do NOT silently reorder.")
    # Reorder raw channels (file order) into config order (W then X_s).
    order = [file_vars.index(v) for v in _CONFIG_VARS]
    raw = np.concatenate([np.asarray(W, np.float64), np.asarray(X_s, np.float64)],
                         axis=1)[:, order]

    df = pd.DataFrame(raw, columns=_CONFIG_VARS)
    # A columns: unit, cycle, Fc, hs -- fixed order per the dataset docs.
    df["__unit"] = np.asarray(A[:, 0], np.int64)
    df["__cycle"] = np.asarray(A[:, 1], np.int64)
    df["__fc"] = np.asarray(A[:, 2], np.float64)

    g = df.groupby(["__unit", "__cycle"], sort=True)
    means = g[_CONFIG_VARS].mean()
    stds = g[_CONFIG_VARS].std().fillna(0.0)   # 1-row cycles -> NaN std -> 0
    counts = g.size().to_numpy(np.float64)     # cycle_len_s (rows per flight)
    fc_first = g["__fc"].first().to_numpy(np.float64)

    # Interleave mean/std per variable to match NCMAPSS_FEATURE_COLUMNS ordering.
    feat = np.empty((len(means), 2 * len(_CONFIG_VARS)), np.float64)
    for j, v in enumerate(_CONFIG_VARS):
        feat[:, 2 * j] = means[v].to_numpy()
        feat[:, 2 * j + 1] = stds[v].to_numpy()
    feat = np.concatenate([feat, counts[:, None]], axis=1)   # + cycle_len_s

    midx = means.index
    unit_arr = midx.get_level_values(0).to_numpy(np.float64)
    cycle_arr = midx.get_level_values(1).to_numpy(np.float64)
    zeros = np.zeros_like(fc_first)
    canon = np.column_stack([unit_arr, cycle_arr, fc_first, zeros, zeros, feat])
    assert canon.shape[1] == len(_CANON_COLUMNS)
    return canon.astype(np.float64)


def _read_and_aggregate(h5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Open one .h5, read ONLY W/X_s/A (+ W/X_s name arrays), aggregate dev and test to
    canonical numeric matrices. X_v/T/Y (oracles) are never touched. A's numeric column
    order (unit, cycle, Fc, hs) is fixed by the dataset docs."""
    import h5py

    with h5py.File(h5_path, "r") as h:
        def rd(k):
            return np.asarray(h[k], np.float32)
        W_dev, X_s_dev, A_dev = rd("W_dev"), rd("X_s_dev"), rd("A_dev")
        W_test, X_s_test, A_test = rd("W_test"), rd("X_s_test"), rd("A_test")
        w_var, xs_var = _decode_vars(h["W_var"]), _decode_vars(h["X_s_var"])

    train = _aggregate_split(W_dev, X_s_dev, A_dev, w_var, xs_var)
    test = _aggregate_split(W_test, X_s_test, A_test, w_var, xs_var)
    # Dev/test units MUST be disjoint within a file (guard, not assumed).
    dev_u, te_u = set(train[:, 0].astype(int)), set(test[:, 0].astype(int))
    if dev_u & te_u:
        raise ValueError(f"{h5_path.name}: dev and test share unit ids {dev_u & te_u}")
    return train, test


def _agg_cache_path(config: Config, ds: str) -> Path:
    return Path(config.cache_dir) / f"ncmapss_agg_{ds}_v{NCMAPSS_AGG_VERSION}.npz"


def _load_or_build_aggregate(config: Config, ds: str, verbose: bool = True
                             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the UNTRUNCATED (df_train_full, df_test_full) canonical frames for one
    per-file dataset, using a versioned per-file aggregate cache (parsing 1-3 GB of
    h5 is minutes; the aggregate is ~10^2-10^3 rows). Idempotent; config-independent
    aside from NCMAPSS_AGG_VERSION."""
    cache_path = _agg_cache_path(config, ds)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as npz:
            train, test = npz["train"], npz["test"]
        if verbose:
            print(f"[ncmapss] loaded cached aggregate {cache_path.name} "
                  f"({len(np.unique(train[:, 0]))} dev + "
                  f"{len(np.unique(test[:, 0]))} test units)")
    else:
        root = _resolve_dir(config)
        h5_path = _find_h5(root, ds)
        if verbose:
            print(f"[ncmapss] parsing {h5_path.name} (1 Hz -> per-cycle aggregate)...")
        train, test = _read_and_aggregate(h5_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, train=train.astype(np.float32), test=test.astype(np.float32))
        if verbose:
            print(f"[ncmapss] parsed {ds}: {len(np.unique(train[:, 0]))} dev units, "
                  f"{len(train)} train cycles; {len(np.unique(test[:, 0]))} test units; "
                  f"cached -> {cache_path.name}")
    df_train = pd.DataFrame(train, columns=_CANON_COLUMNS)
    df_test = pd.DataFrame(test, columns=_CANON_COLUMNS)
    for df in (df_train, df_test):
        df["unit_number"] = df["unit_number"].astype(np.int64)
        df["time_cycles"] = df["time_cycles"].astype(np.int64)
    return df_train, df_test


# ---------------------------------------------------------------------------
# Truncation (test units are run-to-failure; mirror XJTU's protocol, §22)
# ---------------------------------------------------------------------------
def _truncate_test(df_test_full: pd.DataFrame, config: Config
                   ) -> tuple[pd.DataFrame, dict]:
    """Truncate each test unit at ``ncmapss_test_truncation`` of its life; return the
    truncated frame and ``{unit: remaining_cycles}``. Same guards as XJTU."""
    frames, rul = [], {}
    for unit_id, unit_df in df_test_full.groupby("unit_number", sort=True):
        unit_df = unit_df.sort_values("time_cycles")
        n = len(unit_df)
        keep = int(np.floor(n * config.ncmapss_test_truncation))
        keep = max(config.window_size, min(keep, n - 1))
        if keep < 1 or keep >= n:
            raise ValueError(
                f"N-CMAPSS test unit {unit_id}: cannot truncate {n} cycles to a valid "
                f"prefix (window_size={config.window_size}); trajectory too short.")
        frames.append(unit_df.iloc[:keep])
        rul[int(unit_id)] = n - keep
    return pd.concat(frames, ignore_index=True), rul


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------
def load_ncmapss(config: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load ``config.dataset`` (per-file DS0x, or the combined DSALL fleet -- §28) and
    return the canonical ``(df_train, df_test, rul_truth)`` triple; rul_truth indexed
    by unit_number = remaining cycles at each test unit's last kept flight."""
    if config.dataset == "DSALL":
        return _load_dsall(config)
    df_train, df_test_full = _load_or_build_aggregate(config, config.dataset)
    df_test, rul = _truncate_test(df_test_full, config)
    rul_truth = pd.Series(rul, name="rul_truth").sort_index()
    rul_truth.index.name = "unit_number"
    return df_train, df_test, rul_truth


def _load_dsall(config: Config):
    raise NotImplementedError(
        "DSALL is implemented in CHANGES.md §28 (Task 4 of DATASET_EXPANSION_PLAN.md).")
