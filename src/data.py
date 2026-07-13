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

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import Config, ALL_COLUMNS


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_cmapss(config: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load train, test, and ground-truth test-RUL for ``config.dataset``.

    Returns (df_train, df_test, rul_truth) where rul_truth is indexed by
    unit_number and holds the provided remaining-cycles-at-last-observed-cycle.
    """
    data_dir = Path(config.data_dir)
    ds = config.dataset
    df_train = _read_cmapss_txt(data_dir / f"train_{ds}.txt")
    df_test = _read_cmapss_txt(data_dir / f"test_{ds}.txt")

    rul_path = data_dir / f"RUL_{ds}.txt"
    rul_values = pd.read_csv(rul_path, header=None).iloc[:, 0].to_numpy()
    # RUL_FDxxx.txt is ordered by unit number 1..N (Saxena et al. 2008).
    rul_truth = pd.Series(
        rul_values, index=pd.RangeIndex(1, len(rul_values) + 1, name="unit_number"),
        name="rul_truth",
    )
    return df_train, df_test, rul_truth


def _read_cmapss_txt(path: Path) -> pd.DataFrame:
    # 26 whitespace-separated columns, trailing whitespace tolerated.
    df = pd.read_csv(path, sep=r"\s+", header=None, names=ALL_COLUMNS)
    return df


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
