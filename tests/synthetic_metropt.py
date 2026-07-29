"""Tiny synthetic MetroPT-3 generator, in the REAL shipped on-disk format.

Lets the CPU tests exercise ``src.datasets.metropt`` without downloading UCI 791
(1516948 x 17). What "real format" means here, and why each detail matters:

  * ONE comma-delimited, LF-terminated, UTF-8 CSV named like the shipped file
    (``MetroPT3(AirCompressor).csv``, literal parentheses).
  * ONE header row whose FIRST CELL IS EMPTY -- pandas names it ``Unnamed: 0`` -- then
    ``timestamp`` and the 15 signals in FILE order (``Oil_temperature`` BEFORE
    ``Motor_current``, unlike the UCI prose), including the shipped MISSPELLING
    ``DV_eletric``. The loader's header check is byte-exact, so a fixture that "fixed"
    any of this would make the tests pass against a file the loader rejects.
  * The row counter under the empty header is 0, 10, 20, ... exactly as shipped.
  * ``'%Y-%m-%d %H:%M:%S'`` naive timestamps on an IRREGULAR ~10 s grid (the real
    cadence mix: mostly 10 s, some 9 s, a few 12 s) -- plus configurable HOLES, because
    ~17.6% of the real record's wall-clock time is simply absent with no NaN row and no
    sentinel. The default hole is sized to leave one partially-covered bin, which is
    what ``metropt_min_samples_per_cycle`` exists to drop.
  * 7 analog channels that DEGRADE monotonically toward each injected failure event
    (so there is learnable RUL signal) and 8 digital channels written as 0.0/1.0
    FLOATS whose duty cycle also rises toward the event.

``events`` is configurable so a test can inject 2-3 events into a short window (and
monkeypatch ``metropt.METROPT_FAILURE_EVENTS`` to match) or replay the real four-event
table over the real date range at a coarser step.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (METROPT_ANALOG_COLUMNS, METROPT_DIGITAL_COLUMNS,
                        METROPT_SIGNAL_COLUMNS)

# Three air-leak events inside a 3-day window: 4 runs (3 observed + 1 censored tail),
# same dict shape as ``config.METROPT_FAILURE_EVENTS`` so a test can monkeypatch the
# loader's table with this one verbatim.
SYNTHETIC_METROPT_EVENTS = (
    {"event": 1, "start": "2020-02-01 18:00:00", "end": "2020-02-01 20:00:00",
     "failure": "Air leak", "severity": "High stress"},
    {"event": 2, "start": "2020-02-02 18:00:00", "end": "2020-02-02 20:00:00",
     "failure": "Air leak", "severity": "Medium stress"},
    {"event": 3, "start": "2020-02-03 12:00:00", "end": "2020-02-03 13:00:00",
     "failure": "Oil leak", "severity": "High stress"},
)

# One invisible hole, deliberately NOT aligned to a bin edge: it starts 40 s into the
# 02:00 bin, so that bin keeps only a handful of rows and must be DROPPED by the
# min-samples rule, while the bins fully inside the hole never exist at all. The window
# sits inside both the synthetic 3-day range and the real Feb-Sep 2020 range.
SYNTHETIC_METROPT_GAPS = (("2020-02-02 02:00:40", "2020-02-02 07:50:00"),)

# Per-channel (healthy level, drift toward the event, noise sigma). DECISION (fixture
# only): signs follow the air-leak story -- the compressor runs harder and hotter while
# the downstream pressures sag -- so degradation is monotone and every channel carries
# some RUL signal.
_ANALOG_PROFILE = {
    "TP2": (0.70, 2.00, 0.05),              # compressor outlet pressure (bar)
    "TP3": (9.00, -0.80, 0.04),             # pneumatic panel pressure (bar)
    "H1": (8.50, -2.00, 0.06),              # pressure drop at the cyclonic separator
    "DV_pressure": (0.02, 0.40, 0.01),      # air-dryer towers pressure drop
    "Reservoirs": (9.00, -0.70, 0.04),      # downstream reservoir pressure
    "Oil_temperature": (55.0, 12.0, 0.50),  # compressor oil temperature (degC)
    "Motor_current": (2.00, 2.50, 0.10),    # compressor motor current (A)
}
# Per-channel (healthy duty, duty added by end of life) for the 0/1 channels.
_DIGITAL_PROFILE = {
    "COMP": (0.30, 0.50), "DV_eletric": (0.25, 0.55), "Towers": (0.50, 0.00),
    "MPG": (0.30, 0.40), "LPS": (0.02, 0.15), "Pressure_switch": (0.05, 0.05),
    "Oil_level": (0.03, 0.04), "Caudal_impulses": (0.10, 0.20),
}


def write_synthetic_metropt(
    data_dir: Path,
    filename: str = "MetroPT3(AirCompressor).csv",
    start: str = "2020-02-01 00:00:00",
    end: str = "2020-02-04 00:00:00",
    events=SYNTHETIC_METROPT_EVENTS,
    gaps=SYNTHETIC_METROPT_GAPS,
    step_seconds: int = 10,
    seed: int = 0,
) -> Path:
    """Write one real-format MetroPT-3 CSV into ``data_dir`` and return its path.

    ``events`` (same dict shape as ``config.METROPT_FAILURE_EVENTS``) drives BOTH the
    degradation ramps and the run segmentation the loader will re-derive; ``gaps`` is a
    sequence of ``(start, end)`` ISO pairs whose rows are omitted entirely -- no NaN
    row, no sentinel, exactly like the shipped file. ``step_seconds`` sets the nominal
    cadence (10 s = the shipped decimation; raise it to cover a long date range with few
    rows). Rows falling INSIDE a failure window are emitted at full degradation, because
    the real file contains them too and the loader must drop them itself.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    t0 = np.datetime64(start, "s")
    t_end = np.datetime64(end, "s")
    span = int((t_end - t0) / np.timedelta64(1, "s"))
    # Irregular cadence in the shipped proportions (~10 s x1.34M, 9 s x128k, 12 s x38k).
    n_steps = int(span // max(step_seconds - 1, 1)) + 2
    steps = rng.choice([step_seconds, step_seconds - 1, step_seconds + 2],
                       size=n_steps, p=[0.880, 0.085, 0.035])
    offsets = np.concatenate([[0], np.cumsum(steps)])
    t = t0 + offsets.astype("timedelta64[s]")
    t = t[t <= t_end]
    for g_start, g_end in gaps:                     # invisible holes: rows just absent
        t = t[(t < np.datetime64(g_start, "s")) | (t > np.datetime64(g_end, "s"))]
    n = len(t)

    # Run boundaries, mirroring the loader's rule: run i (0-based) spans from the end of
    # event i-1 (or the record start) to the start of event i (or the record end).
    ev_start = np.array([np.datetime64(e["start"], "s") for e in events],
                        dtype="datetime64[s]")
    ev_end = np.array([np.datetime64(e["end"], "s") for e in events],
                      dtype="datetime64[s]")
    bounds_start = np.concatenate([np.array([t0], "datetime64[s]"), ev_end])
    bounds_end = np.concatenate([ev_start, np.array([t_end], "datetime64[s]")])
    run = np.searchsorted(ev_end, t, side="left")
    elapsed = (t - bounds_start[run]) / np.timedelta64(1, "s")
    length = np.maximum((bounds_end[run] - bounds_start[run]) / np.timedelta64(1, "s"), 1)
    frac = np.clip(elapsed / length, 0.0, 1.0)      # 0 = fresh after an intervention

    cols = {"": (np.arange(n, dtype=np.int64) * 10),
            "timestamp": pd.to_datetime(t).strftime("%Y-%m-%d %H:%M:%S")}
    for name in METROPT_ANALOG_COLUMNS:
        base, drift, sigma = _ANALOG_PROFILE[name]
        cols[name] = np.round(base + drift * frac + rng.normal(0.0, sigma, n), 6)
    for name in METROPT_DIGITAL_COLUMNS:
        healthy, rise = _DIGITAL_PROFILE[name]
        cols[name] = (rng.random(n) < healthy + rise * frac).astype(np.float64)

    df = pd.DataFrame(cols)
    assert list(df.columns) == ["", "timestamp"] + list(METROPT_SIGNAL_COLUMNS)
    path = data_dir / filename
    df.to_csv(path, index=False, lineterminator="\n")
    return path
