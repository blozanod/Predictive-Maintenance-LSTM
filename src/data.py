"""C-MAPSS loading, RUL labels, unit-level splits, per-fraction unit subsampling,
and windowing.

Leakage rules enforced here (Task 2.4):
  * Splits are BY ENGINE UNIT. ``make_windows`` returns a ``unit_ids`` array
    aligned to the windows so callers can guarantee no unit's windows straddle a
    split (verified in tests).
  * Data-efficiency subsampling is BY UNIT, not by row, and is seeded
    (RESEARCH_PLAN sec.6). Sampled unit IDs are returned for saving to the run
    directory.
  * The test set is loaded here but its labels are only ever consumed by
    evaluate.py -- nothing in the training path reads test labels.
  * Any fitted scaler (baselines) is fit on the TRAIN split of the current data
    fraction only; see ``fit_channel_scaler`` / ``apply_channel_scaler``.

Chronos-2 instance-normalizes each series internally (embed() returns per-series
loc/scale), so the TSFM path needs no fitted scaler and its cached embeddings are
fraction-independent -- computed once over all units (Stage A).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .config import Config, SETTING_COLUMNS, CONDITION_SETTING_DECIMALS
from . import datasets as datasets_pkg
# Re-export the per-dataset raw loaders so ``data.load_cmapss`` / ``data.load_xjtu``
# stay valid entry points (data.py is the data facade); their implementations live
# in src/datasets/ (one module per dataset family).
from .datasets.cmapss import load_cmapss  # noqa: F401
from .datasets.xjtu import load_xjtu  # noqa: F401
from .datasets.ncmapss import load_ncmapss  # noqa: F401


# ---------------------------------------------------------------------------
# Unified entry point (all pipeline stages load through here)
# ---------------------------------------------------------------------------
def load_prepared(config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load ``config.dataset``, attach RUL labels, and apply condition-wise
    normalization when resolved ON -- the ONE loading path every pipeline stage
    (Stage A caches, baselines, fairness arms) uses, so no stage can disagree
    about preprocessing.

    Returns (df_train, df_test), both carrying ``actual_rul``/``clipped_rul`` and
    the canonical C-MAPSS-shaped columns (unit_number, time_cycles, settings,
    sensor channels)."""
    # Raw loading is dispatched by dataset family through the src/datasets/ registry;
    # every family emits the same canonical frame shape.
    df_train, df_test, rul_truth = datasets_pkg.load_raw(config)
    df_train = add_train_rul(df_train, config)
    df_test = add_test_rul(df_test, rul_truth, config)
    if config.effective_condition_norm():
        df_train, df_test = condition_normalize(df_train, df_test, config)
    # RQ-H sim-only perturbation: applied AFTER labels + normalization, BEFORE
    # windowing (CHANGES.md §38). No-op unless config.noise_injection is set.
    if config.noise_injection:
        df_train, df_test = apply_noise_injection(df_train, df_test, config)
    return df_train, df_test


# ---------------------------------------------------------------------------
# Sim-only noise/drift injection (RQ-H, perturbative; RESEARCH_PLAN §1; CHANGES.md §38)
# ---------------------------------------------------------------------------
def inject_noise(
    df: pd.DataFrame, sensor_columns: list, spec: dict, seed: int
) -> pd.DataFrame:
    """Return a copy of ``df`` with ``sensor_columns`` perturbed per ``spec`` (a
    ``config.noise_injection`` dict). Deterministic in ``seed`` -- a fixed seed + spec
    reproduces the same perturbed series. Perturbation magnitudes are expressed in
    per-channel std units, so the effect is comparable across channels and datasets.
    DECISION (uncited): the three kinds and their default parameters.

      * ``gaussian`` -- additive white noise at ``snr_db`` (default 20): per-channel
        noise std = channel_std / sqrt(10**(snr_db/10)).
      * ``drift``    -- a per-unit linear bias ramp reaching ``magnitude`` (default 1)
        channel-std by each unit's last cycle (a slow calibration drift).
      * ``dropout``  -- each (row, channel) is blanked to 0 with probability ``rate``
        (default 0.1), the random sensor-dropout failure mode.
    """
    kind = spec["kind"]
    rng = np.random.default_rng(seed)
    out = df.copy()
    X = out[sensor_columns].to_numpy(np.float64, copy=True)  # (rows, C)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0                                    # guard flat channels
    if kind == "gaussian":
        snr_db = float(spec.get("snr_db", 20.0))
        noise_std = std / np.sqrt(10.0 ** (snr_db / 10.0))
        X = X + rng.normal(0.0, 1.0, size=X.shape) * noise_std
    elif kind == "drift":
        magnitude = float(spec.get("magnitude", 1.0))
        # per-unit 0..1 progress by cycle order (order-independent), scaled to std.
        rank = out.groupby("unit_number")["time_cycles"].rank(method="first") - 1.0
        size = out.groupby("unit_number")["time_cycles"].transform("size")
        frac = np.where(size > 1, rank / (size - 1).clip(lower=1), 0.0)
        X = X + frac[:, None] * (magnitude * std)
    else:  # "dropout" (kinds are validated in Config.__post_init__)
        rate = float(spec.get("rate", 0.1))
        X[rng.random(X.shape) < rate] = 0.0
    out[sensor_columns] = X
    return out


def apply_noise_injection(
    df_train: pd.DataFrame, df_test: pd.DataFrame, config: Config
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Guarded application of ``config.noise_injection`` to both frames. RAISES on a
    REAL dataset -- perturbing real sensor readings is out of scope by design
    (RESEARCH_PLAN §1); the guard reports both the expected (simulated) families and
    the observed dataset. Train/test get distinct-but-reproducible seeds so the two
    perturbations are independent yet deterministic."""
    if not config.is_simulated_dataset():
        from .config import SIMULATED_DATASET_KINDS
        raise ValueError(
            f"noise_injection is sim-only (RQ-H): allowed for simulated families "
            f"{SIMULATED_DATASET_KINDS}, but dataset {config.dataset!r} is real "
            f"(kind {config.dataset_kind()!r}). Perturbing real readings is out of "
            f"scope -- cover the noise axis observationally on real data instead.")
    spec = config.noise_injection
    base = config.effective_noise_seed()   # same resolution the cache key captures (§40)
    cols = config.sensor_columns
    return (inject_noise(df_train, cols, spec, base),
            inject_noise(df_test, cols, spec, base + 1))


# ---------------------------------------------------------------------------
# Condition-wise normalization (plan §6; CHANGES.md §21)
# ---------------------------------------------------------------------------
def condition_keys(
    df: pd.DataFrame,
    setting_columns: list = SETTING_COLUMNS,
    decimals=CONDITION_SETTING_DECIMALS,
) -> np.ndarray:
    """Discrete operating-condition KEY per row: the settings snapped onto their
    (per-column-rounded) grid, one row per condition combination -- shape
    ``(n_rows, n_settings)`` float64. Keys are VALUES, not per-frame ranks, so
    train and test rows at the same operating point always share a key even when
    the two frames saw different condition sets. No fitting, hence no leakage."""
    rounded = np.column_stack([
        np.round(df[c].to_numpy(np.float64), d) for c, d in zip(setting_columns, decimals)
    ])
    return rounded + 0.0  # -0.0 -> 0.0 so a rounded zero's sign never splits a condition


def identify_conditions(df: pd.DataFrame, **kw) -> np.ndarray:
    """Per-frame integer condition IDs (rank of the rounded setting tuple within
    THIS frame) -- for counting/diagnostics only; normalization uses the value
    keys from ``condition_keys`` so cross-frame alignment never depends on ranks."""
    _, inverse = np.unique(condition_keys(df, **kw), axis=0, return_inverse=True)
    return inverse.astype(np.int64)


def fit_condition_scaler(
    df_train: pd.DataFrame, sensor_columns: list, keys: np.ndarray
) -> dict:
    """{condition key tuple: (mean, std)} of each sensor channel, fit on TRAIN
    rows only. Channels flat within a condition get std=1 (they normalize to ~0
    and carry no signal -- e.g. the 7 conventionally-dropped C-MAPSS sensors).
    Also stores a ``None``-keyed GLOBAL fallback for operating points never seen
    in training."""
    X = df_train[sensor_columns].to_numpy(np.float64)
    scaler: dict = {}
    for key in np.unique(keys, axis=0):
        rows = X[np.all(keys == key, axis=1)]
        mean, std = rows.mean(axis=0), rows.std(axis=0)
        std[std < 1e-8] = 1.0
        scaler[tuple(key)] = (mean, std)
    g_mean, g_std = X.mean(axis=0), X.std(axis=0)
    g_std[g_std < 1e-8] = 1.0
    scaler[None] = (g_mean, g_std)
    return scaler


def apply_condition_scaler(
    df: pd.DataFrame, sensor_columns: list, keys: np.ndarray, scaler: dict
) -> pd.DataFrame:
    """Z-normalize each row's sensor channels with its condition's train-fit
    statistics (global fallback for conditions unseen in training). Returns a
    copy."""
    out = df.copy()
    # copy=True: pandas copy-on-write may hand back a read-only view otherwise
    X = out[sensor_columns].to_numpy(np.float64, copy=True)
    for key in np.unique(keys, axis=0):
        mean, std = scaler.get(tuple(key), scaler[None])
        m = np.all(keys == key, axis=1)
        X[m] = (X[m] - mean) / std
    out[sensor_columns] = X
    return out


def condition_normalize(
    df_train: pd.DataFrame, df_test: pd.DataFrame, config: Config
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Condition-wise z-normalization of both frames, statistics fit on the FULL
    train frame once (never on test). Fitting once over all train units -- rather
    than per data fraction -- is a deliberate, documented deviation (CHANGES.md
    §21): it keeps the single-pass embedding cache, and condition statistics are
    properties of the operating points, insensitive to which units are sampled."""
    cols = config.sensor_columns
    k_tr, k_te = condition_keys(df_train), condition_keys(df_test)
    scaler = fit_condition_scaler(df_train, cols, k_tr)
    return (apply_condition_scaler(df_train, cols, k_tr, scaler),
            apply_condition_scaler(df_test, cols, k_te, scaler))


# ---------------------------------------------------------------------------
# RUL labels
# ---------------------------------------------------------------------------
def add_train_rul(df_train: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Add ``actual_rul`` and ``clipped_rul`` to the training frame.

    Training RUL = (unit's max cycle) - (current cycle), clipped at max_rul.
    Piecewise-linear target, community convention (Heimes 2008; Li et al. 2018).
    """
    df = df_train.copy()
    max_cycle = df.groupby("unit_number")["time_cycles"].transform("max")
    df["actual_rul"] = max_cycle - df["time_cycles"]
    df["clipped_rul"] = df["actual_rul"].clip(upper=config.max_rul)
    return df


def add_test_rul(
    df_test: pd.DataFrame, rul_truth: pd.Series, config: Config
) -> pd.DataFrame:
    """Add ``actual_rul`` (unclipped) and ``clipped_rul`` (clipped at max_rul) to
    the test frame.

    For a test unit whose series ends before failure, RUL at a given cycle is
    provided_remaining_cycles + (unit's last observed cycle - current cycle).
    The cache stores the UNCLIPPED ``actual_rul`` as the test label; evaluate.py
    reports BOTH protocols (clipped at max_rul and unclipped) from it (Task 1.4).
    """
    df = df_test.copy()
    max_cycle = df.groupby("unit_number")["time_cycles"].transform("max")
    remaining = df["unit_number"].map(rul_truth)
    df["actual_rul"] = remaining + (max_cycle - df["time_cycles"])
    df["clipped_rul"] = df["actual_rul"].clip(upper=config.max_rul)
    return df


# ---------------------------------------------------------------------------
# Unit-level splits & subsampling (BY UNIT, seeded)
# ---------------------------------------------------------------------------
def unit_ids_of(df: pd.DataFrame) -> np.ndarray:
    return np.sort(df["unit_number"].unique())


def subsample_units(
    all_units: np.ndarray, n_units: int, seed: int
) -> np.ndarray:
    """Sample ``n_units`` engine units without replacement, seeded and sorted.

    By UNIT, never by row (RESEARCH_PLAN sec.6). Deterministic in (units, seed).
    """
    all_units = np.asarray(all_units)
    if n_units >= len(all_units):
        return np.sort(all_units.copy())
    rng = np.random.default_rng(seed)
    chosen = rng.choice(all_units, size=n_units, replace=False)
    return np.sort(chosen)


def unit_train_val_split(
    units: np.ndarray, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split units into (train_units, val_units) by unit, seeded.

    Guarantees disjoint unit sets so no unit's windows cross the split.
    """
    units = np.sort(np.asarray(units))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(units)
    n_val = max(1, int(round(len(units) * val_fraction))) if len(units) > 1 else 0
    val_units = np.sort(perm[:n_val])
    train_units = np.sort(perm[n_val:])
    return train_units, val_units


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------
def make_windows(
    df: pd.DataFrame,
    sensor_columns: list,
    window_size: int,
    target_col: str = "clipped_rul",
    units: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sliding windows over each unit's trajectory.

    Returns:
        windows:  (N, window_size, n_channels) float32
        labels:   (N,) float32 -- target at the LAST cycle of each window
                  (RUL is defined at the window end; verified in tests).
        unit_ids: (N,) int -- the unit each window belongs to (for split checks).

    Windows never span two units. Units shorter than window_size yield no windows.
    """
    if units is not None:
        units = set(int(u) for u in units)
    windows, labels, win_units = [], [], []
    channels = list(sensor_columns)
    for unit_id, unit_df in df.groupby("unit_number", sort=True):
        if units is not None and int(unit_id) not in units:
            continue
        sensors = unit_df[channels].to_numpy(dtype=np.float32)
        targets = unit_df[target_col].to_numpy(dtype=np.float32)
        n = len(unit_df)
        for i in range(n - window_size + 1):
            windows.append(sensors[i : i + window_size])
            labels.append(targets[i + window_size - 1])  # label at window END
            win_units.append(int(unit_id))
    if not windows:
        n_ch = len(channels)
        return (
            np.empty((0, window_size, n_ch), np.float32),
            np.empty((0,), np.float32),
            np.empty((0,), np.int64),
        )
    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
        np.asarray(win_units, dtype=np.int64),
    )


def make_test_last_windows(
    df_test: pd.DataFrame,
    sensor_columns: list,
    window_size: int,
    target_col: str = "clipped_rul",
    pad_short: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One FIXED-shape window per test unit: the last ``window_size`` cycles.

    C-MAPSS last-cycle test protocol (RESEARCH_PLAN sec.6): predict RUL at each
    test unit's final observed cycle. Units shorter than window_size are
    left-padded by repeating the first cycle when ``pad_short`` is True. This is the
    BASELINE path; the TSFM path uses ``make_test_last_contexts`` (no padding).
    The padding is always at the FRONT, so ``win[-1]`` is the true last cycle.
    """
    channels = list(sensor_columns)
    windows, labels, win_units = [], [], []
    for unit_id, unit_df in df_test.groupby("unit_number", sort=True):
        sensors = unit_df[channels].to_numpy(dtype=np.float32)
        last_target = float(unit_df[target_col].to_numpy()[-1])
        n = len(unit_df)
        if n >= window_size:
            win = sensors[n - window_size :]
        elif pad_short:
            pad = np.repeat(sensors[:1], window_size - n, axis=0)  # left-pad w/ first cycle
            win = np.concatenate([pad, sensors], axis=0)
        else:
            continue
        windows.append(win)
        labels.append(last_target)
        win_units.append(int(unit_id))
    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
        np.asarray(win_units, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Variable-length TSFM contexts (Task 1.2)
#
# The TSFM path replaces fixed-shape padded windows with embed()'s native
# variable-length input: a LIST of per-window arrays. Two guarantees make it a
# drop-in for the fixed path in the sweep:
#   * ``make_windows_varlen`` iterates units and cycles IDENTICALLY to
#     ``make_windows``, so window i, its label, and its unit_id match 1:1 -- the
#     head trains on the same rows the baselines do (verified in tests).
#   * ``make_test_last_contexts`` yields one context per test unit in the same
#     unit order as ``make_test_last_windows``, with the same last-cycle label.
# The only difference is the CONTENT: instead of a fixed window_size slice, each
# context is up to ``tsfm_context_length`` real cycles of history, never padded.
# Short test histories (FD001: 37/100 test units < 120 cycles) stay short and are
# left-pad-MASKED inside embed(), not fabricated (Task 1.4 padding hazard).
# ---------------------------------------------------------------------------
def make_windows_varlen(
    df: pd.DataFrame,
    sensor_columns: list,
    window_size: int,
    tsfm_context_length: int,
    target_col: str = "clipped_rul",
    units: Optional[np.ndarray] = None,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Variable-length TSFM contexts aligned 1:1 to ``make_windows``.

    For each prediction cycle c (the SAME set as make_windows: window_size..n), the
    context is the last ``min(c, tsfm_context_length)`` cycles ending at cycle c.
    Returns (contexts, labels, unit_ids) where ``contexts`` is a list of
    ``(L_i, n_channels)`` float32 arrays; ``labels``/``unit_ids`` are identical to
    ``make_windows(..., window_size, target_col, units)``.
    """
    if units is not None:
        units = set(int(u) for u in units)
    contexts: list[np.ndarray] = []
    labels, win_units = [], []
    channels = list(sensor_columns)
    for unit_id, unit_df in df.groupby("unit_number", sort=True):
        if units is not None and int(unit_id) not in units:
            continue
        sensors = unit_df[channels].to_numpy(dtype=np.float32)
        targets = unit_df[target_col].to_numpy(dtype=np.float32)
        n = len(unit_df)
        for i in range(n - window_size + 1):
            end = i + window_size          # cycles 1..end covered so far (1-indexed end)
            start = max(0, end - tsfm_context_length)
            contexts.append(sensors[start:end])
            labels.append(targets[end - 1])   # label at prediction cycle == window END
            win_units.append(int(unit_id))
    return (
        contexts,
        np.asarray(labels, dtype=np.float32),
        np.asarray(win_units, dtype=np.int64),
    )


def make_test_last_contexts(
    df_test: pd.DataFrame,
    sensor_columns: list,
    tsfm_context_length: int,
    target_col: str = "clipped_rul",
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """One variable-length context per test unit: the last
    ``min(n, tsfm_context_length)`` cycles ending at the final observed cycle.

    Same unit order and last-cycle labels as ``make_test_last_windows`` but NEVER
    padded -- short histories stay short for embed()'s masked left-padding.
    """
    channels = list(sensor_columns)
    contexts: list[np.ndarray] = []
    labels, win_units = [], []
    for unit_id, unit_df in df_test.groupby("unit_number", sort=True):
        sensors = unit_df[channels].to_numpy(dtype=np.float32)
        last_target = float(unit_df[target_col].to_numpy()[-1])
        n = len(unit_df)
        start = max(0, n - tsfm_context_length)
        contexts.append(sensors[start:])
        labels.append(last_target)
        win_units.append(int(unit_id))
    return (
        contexts,
        np.asarray(labels, dtype=np.float32),
        np.asarray(win_units, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Channel scaler for baselines (fit on the CURRENT fraction's train split only)
# ---------------------------------------------------------------------------
def fit_channel_scaler(windows_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-channel standardization statistics on training windows only.

    Returns (mean, std) of shape (n_channels,). No leakage: caller passes only
    the current data fraction's TRAIN windows (Task 2.4). Chronos-2's own
    per-series normalization means the TSFM path does not use this.
    """
    flat = windows_train.reshape(-1, windows_train.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)  # guard flat channels
    return mean.astype(np.float32), std.astype(np.float32)


def apply_channel_scaler(
    windows: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    return ((windows - mean) / std).astype(np.float32)
