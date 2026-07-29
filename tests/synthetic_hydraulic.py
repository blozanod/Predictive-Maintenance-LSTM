"""Synthetic UCI hydraulic rig (UCI 447) in the REAL on-disk format, scaled down.

``src/datasets/hydraulic.py`` reads 18 TAB-delimited, HEADER-LESS text files whose column
count IS the sensor's sampling rate and whose row i IS cycle i. This fixture writes
exactly that -- same delimiter, no header, no index column, one row per cycle,
``profile.txt`` with its 5 integer columns in the shipped order -- so the CPU tests
exercise the real parse path (including ``header=None``, which is load-bearing) without
downloading 556 MB.

Two knobs keep it fast while preserving everything the loader validates:
  * ``samples_scale`` divides every documented width by the SAME factor, so the
    100/10/1 Hz RATIO is intact (default 60 -> 100/10/1 samples instead of 6000/600/60);
    the loader accepts one common scale and rejects any per-file mismatch.
  * ``cycles`` sets the record length (the shipped file has 2205).

The layout mirrors the real one: a NESTED FACTORIAL with cooler outermost (three regimes,
in the file's own worst->healthy value order) and valve innermost, contiguous label blocks
of ``block_len`` cycles, an unstable warm-up at the start of each cooler regime plus an
isolated single-cycle transition marker (``stable_flag`` 1 = NOT stable).

Fault-injection hooks (``wrong_columns``, ``row_counts``, ``bad_profile``,
``nan_sensors``) write deliberately broken files so every fail-loud branch of the loader
is reachable from a CPU test.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import (HYDRAULIC_COMPONENTS, HYDRAULIC_PROFILE_COLUMNS,
                        HYDRAULIC_PROFILE_FILE, HYDRAULIC_SENSORS,
                        HYDRAULIC_SENSOR_NAMES, HYDRAULIC_SEVERITY_ORDER)


def synthetic_profile(cycles: int, block_len: int, unstable_warmup: int) -> np.ndarray:
    """``(cycles, 5)`` int64 annotation in ``HYDRAULIC_PROFILE_COLUMNS`` order.

    Cooler splits the record into three equal regimes (its shipped value order 3 -> 20 ->
    100, i.e. worst -> healthy), and inside each regime the accumulator/pump/valve
    factorial cycles with ``block_len`` consecutive cycles per combination -- valve
    innermost, exactly like the shipped schedule. ``stable_flag`` is 1 (NOT stable) for
    the first ``unstable_warmup`` cycles of every regime plus one isolated marker cycle.
    """
    cooler_levels = tuple(reversed(HYDRAULIC_SEVERITY_ORDER["cooler"]))
    inner = list(product(HYDRAULIC_SEVERITY_ORDER["accumulator"],
                         HYDRAULIC_SEVERITY_ORDER["pump"],
                         HYDRAULIC_SEVERITY_ORDER["valve"]))
    profile = np.zeros((cycles, len(HYDRAULIC_PROFILE_COLUMNS)), np.int64)
    for cooler, segment in zip(cooler_levels,
                               np.array_split(np.arange(cycles), len(cooler_levels))):
        if segment.size == 0:
            continue
        for position, row in enumerate(segment):
            accumulator, pump, valve = inner[(position // block_len) % len(inner)]
            profile[row] = (cooler, valve, pump, accumulator, 0)
        profile[segment[:unstable_warmup], -1] = 1          # warm-up block
        marker = min(segment.size - 1, unstable_warmup + block_len + block_len // 2)
        profile[segment[marker], -1] = 1                    # isolated transition marker
    return profile


def severity_ordinals(profile: np.ndarray) -> np.ndarray:
    """``(cycles, 4)`` healthy->worst ranks of a clean profile (test-side reference
    implementation of the loader's polarity fix)."""
    return np.column_stack([
        [HYDRAULIC_SEVERITY_ORDER[component].index(int(value))
         for value in profile[:, HYDRAULIC_PROFILE_COLUMNS.index(component)]]
        for component in HYDRAULIC_COMPONENTS])


def write_synthetic_hydraulic(
    data_dir: Path,
    cycles: int = 240,
    block_len: int = 8,
    samples_scale: int = 60,
    unstable_warmup: int = 8,
    seed: int = 0,
    wrong_columns: Optional[dict] = None,
    row_counts: Optional[dict] = None,
    bad_profile: Optional[tuple] = None,
    nan_sensors: tuple = (),
) -> Path:
    """Write ``profile.txt`` + the 17 sensor files into ``data_dir``; return it.

    Sensor readings are ``base(sensor) + fault_level(cycle) + within-cycle ramp + noise``:
    the fault term makes the four severity ladders linearly separable (so the RQ-F probe
    has something to find), and the ramp gives the ``slope`` statistic a non-degenerate
    value whenever a sensor has more than one sample per cycle.

    Break-it hooks (each writes a file that is malformed in exactly one way):
      * ``wrong_columns={"PS2": 37}`` -- that sensor gets 37 columns instead of its
        rate-scaled width;
      * ``row_counts={"FS1": 12}``    -- that sensor gets 12 rows instead of ``cycles``;
      * ``bad_profile=("valve", 55)`` -- cycle 1's annotation for that profile column is
        set to an undocumented value;
      * ``nan_sensors=("TS1",)``      -- that sensor's first reading is written as ``nan``.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    bad_scale = [name for name in HYDRAULIC_SENSOR_NAMES
                 if HYDRAULIC_SENSORS[name][0] % samples_scale]
    if bad_scale:
        raise ValueError(
            f"samples_scale={samples_scale} does not divide the documented width of "
            f"{bad_scale}; it must divide 60 so the 100/10/1 Hz ratio survives.")
    wrong_columns = dict(wrong_columns or {})
    row_counts = dict(row_counts or {})
    rng = np.random.default_rng(seed)

    profile = synthetic_profile(cycles, block_len, unstable_warmup)
    # Per-cycle fault level: a weighted sum of the four ordinal severities, so every
    # component is recoverable from the sensors (and none is a constant offset).
    weights = np.array([0.5, 2.0, 1.5, 1.0])
    fault = severity_ordinals(profile) @ weights

    for index, name in enumerate(HYDRAULIC_SENSOR_NAMES):
        width = int(wrong_columns.get(name, HYDRAULIC_SENSORS[name][0] // samples_scale))
        n_rows = int(row_counts.get(name, cycles))
        level = np.resize(fault, n_rows)                     # tiles/truncates cleanly
        ramp = np.linspace(0.0, 1.0, width) if width > 1 else np.zeros(width)
        values = (10.0 + 5.0 * index + level[:, None]
                  + (1.0 + 0.1 * index) * ramp[None, :]
                  + rng.normal(0.0, 0.05, size=(n_rows, width)))
        if name in nan_sensors:
            values[0, 0] = np.nan
        np.savetxt(data_dir / f"{name}.txt", values, delimiter="\t", fmt="%.4f")

    if bad_profile is not None:
        column, value = bad_profile
        profile[0, HYDRAULIC_PROFILE_COLUMNS.index(column)] = value
    np.savetxt(data_dir / HYDRAULIC_PROFILE_FILE, profile, delimiter="\t", fmt="%d")
    return data_dir
