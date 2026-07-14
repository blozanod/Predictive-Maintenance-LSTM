"""XJTU-SY bearing run-to-failure loader -> canonical C-MAPSS-shaped frames.

Dataset (Wang et al. 2020, IEEE TR 69(1); download: https://biaowang.tech/xjtu-sy-
bearing-datasets/ or the mirrors linked there): 15 rolling bearings run to failure
under 3 operating conditions, horizontal+vertical accelerometers sampled at
25.6 kHz for 1.28 s once per minute. Expected directory layout under
``config.data_dir``::

    35Hz12kN/Bearing1_1/{1.csv, 2.csv, ...}   # one CSV per minute, 2 columns
    37.5Hz11kN/Bearing2_1/...                 # (Horizontal_..., Vertical_...)
    40Hz12kN/Bearing3_1/...

Adaptation to the pipeline's canonical frame (the whole point -- everything
downstream of ``data.load_prepared`` runs unchanged):
  * one "cycle"  = one 1-minute snapshot; ``time_cycles`` = snapshot index;
  * one "unit"   = one bearing (unit_number = global 1..15 in sorted name order);
  * "sensors"    = per-snapshot condition indicators per axis
    (``XJTU_FEATURE_COLUMNS``): classic time-domain bearing features, computed
    per snapshot -- NOT the raw 32768-sample waveform (Chronos-2 contexts are
    per-cycle series, and minute-level indicator trends are the standard
    formulation for XJTU RUL);
  * ``setting_1`` = condition index (0..2), ``setting_2`` = speed (Hz),
    ``setting_3`` = radial force (kN), so condition-wise normalization (data.py)
    groups by operating condition exactly as for FD002/FD004.

Split protocol (DECISION, uncited -- no community standard; CHANGES.md §22):
``config.xjtu_test_bearings`` (default the last 2 of 5 per condition) are held
out; each test bearing is truncated at ``config.xjtu_test_truncation`` of its
life (>= window_size cycles kept) to mimic the C-MAPSS "predict at the last
observed cycle" protocol, with provided RUL = remaining minutes at truncation.
``max_rul`` is in MINUTES here; the FD-convention 125 is arbitrary for bearings
(lifetimes span ~35 min to ~42 h) -- set it deliberately per experiment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

# Time-domain condition indicators per axis (h_ = horizontal, v_ = vertical).
_BASE_FEATURES = ["rms", "kurtosis", "skewness", "peak", "p2p",
                  "crest", "impulse", "shape"]
XJTU_FEATURE_COLUMNS = [f"{ax}_{f}" for ax in ("h", "v") for f in _BASE_FEATURES]

# Condition folder -> (index, speed Hz, radial force kN). Folder names as shipped.
XJTU_CONDITIONS = {
    "35Hz12kN": (0, 35.0, 12.0),
    "37.5Hz11kN": (1, 37.5, 11.0),
    "40Hz12kN": (2, 40.0, 12.0),
}


def snapshot_features(x: np.ndarray) -> list[float]:
    """``_BASE_FEATURES`` of one axis' snapshot (1D array of samples)."""
    x = np.asarray(x, np.float64)
    n = x.size
    mean = x.mean()
    xc = x - mean
    rms = float(np.sqrt(np.mean(x * x)))
    std = float(xc.std()) or 1e-12
    abs_x = np.abs(x)
    peak = float(abs_x.max())
    mean_abs = float(abs_x.mean()) or 1e-12
    kurt = float(np.mean(xc**4) / std**4 - 3.0)
    skew = float(np.mean(xc**3) / std**3)
    p2p = float(x.max() - x.min())
    crest = peak / (rms or 1e-12)
    impulse = peak / mean_abs
    shape = (rms or 1e-12) / mean_abs
    return [rms, kurt, skew, peak, p2p, crest, impulse, shape]


def _bearing_frame(bearing_dir: Path, unit_id: int, cond: tuple) -> pd.DataFrame:
    """All snapshots of one bearing -> rows of the canonical frame."""
    cond_idx, speed, force = cond
    files = sorted(bearing_dir.glob("*.csv"), key=lambda p: int(p.stem))
    if not files:
        raise FileNotFoundError(f"no snapshot CSVs in {bearing_dir}")
    rows = []
    for i, f in enumerate(files, start=1):
        snap = pd.read_csv(f).to_numpy(np.float64)
        if snap.ndim != 2 or snap.shape[1] < 2:
            raise ValueError(f"{f}: expected 2 columns (horizontal, vertical), "
                             f"got shape {snap.shape}")
        rows.append([unit_id, i, float(cond_idx), speed, force,
                     *snapshot_features(snap[:, 0]), *snapshot_features(snap[:, 1])])
    cols = (["unit_number", "time_cycles", "setting_1", "setting_2", "setting_3"]
            + XJTU_FEATURE_COLUMNS)
    return pd.DataFrame(rows, columns=cols)


def load_xjtu(config: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Walk ``config.data_dir``, build the canonical frames, apply the fixed
    split + test-truncation protocol. Mirrors ``data.load_cmapss``'s return
    contract: (df_train, df_test, rul_truth), rul_truth indexed by unit_number =
    remaining cycles (minutes) at each TEST unit's last kept snapshot."""
    root = Path(config.data_dir)
    found = {}
    for cond_name, cond in XJTU_CONDITIONS.items():
        cond_dir = root / cond_name
        if not cond_dir.is_dir():
            continue
        for bdir in sorted(p for p in cond_dir.iterdir() if p.is_dir()):
            found[bdir.name] = (bdir, cond)
    if not found:
        raise FileNotFoundError(
            f"no XJTU-SY condition folders ({list(XJTU_CONDITIONS)}) under {root}")
    unknown = set(config.xjtu_test_bearings) - set(found)
    if unknown:
        raise ValueError(f"xjtu_test_bearings not on disk: {sorted(unknown)}; "
                         f"found {sorted(found)}")

    train_frames, test_frames, rul = [], [], {}
    for unit_id, name in enumerate(sorted(found), start=1):
        bdir, cond = found[name]
        frame = _bearing_frame(bdir, unit_id, cond)
        if name in config.xjtu_test_bearings:
            n = len(frame)
            # keep >= window_size cycles so the unit yields at least one window,
            # and always truncate at least 1 cycle so RUL truth is > 0.
            keep = int(np.floor(n * config.xjtu_test_truncation))
            keep = max(config.window_size, min(keep, n - 1))
            if keep < 1 or keep >= n:
                raise ValueError(
                    f"{name}: cannot truncate {n} snapshots to a valid test "
                    f"prefix (window_size={config.window_size}); bearing too short.")
            test_frames.append(frame.iloc[:keep])
            rul[unit_id] = n - keep
        else:
            train_frames.append(frame)
    if not train_frames or not test_frames:
        raise ValueError("XJTU split produced an empty train or test set; check "
                         "xjtu_test_bearings against the folders on disk.")
    df_train = pd.concat(train_frames, ignore_index=True)
    df_test = pd.concat(test_frames, ignore_index=True)
    rul_truth = pd.Series(rul, name="rul_truth").sort_index()
    rul_truth.index.name = "unit_number"
    return df_train, df_test, rul_truth
