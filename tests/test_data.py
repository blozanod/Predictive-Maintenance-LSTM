"""Data-layer correctness: label alignment, split integrity, unit subsampling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src import data as D


def _linear_unit(unit_id, n_cycles, n_sensors=3):
    """One unit whose sensors equal the cycle index (so we can check alignment)."""
    rows = {"unit_number": unit_id, "time_cycles": np.arange(1, n_cycles + 1)}
    for j in range(n_sensors):
        rows[f"s_{j+2}"] = np.arange(1, n_cycles + 1, dtype=float) + j
    return pd.DataFrame(rows)


def test_train_rul_and_window_label_alignment():
    cfg = Config(window_size=5, max_rul=1000)  # no clipping so RUL is exact
    df = _linear_unit(1, 20)
    df = D.add_train_rul(df, cfg)
    # RUL at cycle t = max_cycle - t = 20 - t.
    assert df.loc[df["time_cycles"] == 20, "actual_rul"].item() == 0
    assert df.loc[df["time_cycles"] == 1, "actual_rul"].item() == 19

    sensors = ["s_2", "s_3", "s_4"]
    win, lab, units = D.make_windows(df, sensors, cfg.window_size, "clipped_rul")
    # Window i covers cycles i+1..i+window_size; its label is RUL at the LAST cycle.
    assert win.shape == (20 - 5 + 1, 5, 3)
    for i in range(len(win)):
        last_cycle = i + cfg.window_size  # cycles are 1-indexed
        assert lab[i] == pytest.approx(20 - last_cycle)
        # first sensor equals cycle index => last row equals last cycle
        assert win[i, -1, 0] == pytest.approx(last_cycle)
    assert set(units.tolist()) == {1}


def test_clipping_applied():
    cfg = Config(window_size=5, max_rul=10)
    df = D.add_train_rul(_linear_unit(1, 30), cfg)
    assert df["clipped_rul"].max() == 10
    assert df["actual_rul"].max() == 29


def test_no_unit_crosses_split():
    cfg = Config(window_size=5, val_fraction=0.5)
    df = pd.concat([_linear_unit(u, 15) for u in range(1, 7)], ignore_index=True)
    df = D.add_train_rul(df, cfg)
    units = D.unit_ids_of(df)
    train_u, val_u = D.unit_train_val_split(units, cfg.val_fraction, seed=0)
    # disjoint unit sets
    assert set(train_u).isdisjoint(set(val_u))
    assert set(train_u) | set(val_u) == set(units.tolist())
    # windows restricted to each split contain only that split's units
    _, _, tr_units = D.make_windows(df, ["s_2"], cfg.window_size, units=train_u)
    _, _, va_units = D.make_windows(df, ["s_2"], cfg.window_size, units=val_u)
    assert set(np.unique(tr_units)).issubset(set(train_u.tolist()))
    assert set(np.unique(va_units)).issubset(set(val_u.tolist()))
    assert set(np.unique(tr_units)).isdisjoint(set(np.unique(va_units)))


def test_subsample_units_by_unit_seeded():
    units = np.arange(1, 101)
    a = D.subsample_units(units, 10, seed=0)
    b = D.subsample_units(units, 10, seed=0)
    c = D.subsample_units(units, 10, seed=1)
    assert len(a) == 10 and len(set(a)) == 10           # by unit, no repeats
    assert np.array_equal(a, b)                          # deterministic in seed
    assert not np.array_equal(a, c)                      # seed changes selection
    assert set(a).issubset(set(units.tolist()))
    # nested-ish: asking for all returns all
    assert np.array_equal(D.subsample_units(units, 100, 0), units)


def test_test_last_windows_one_per_unit_and_padding():
    cfg = Config(window_size=10)
    frames = []
    for u, n in [(1, 4), (2, 25)]:  # unit 1 is shorter than the window
        frames.append(_linear_unit(u, n))
    df = pd.concat(frames, ignore_index=True)
    rul = pd.Series([7, 12], index=pd.RangeIndex(1, 3, name="unit_number"))
    df = D.add_test_rul(df, rul, cfg)
    win, lab, units = D.make_test_last_windows(df, ["s_2", "s_3", "s_4"],
                                               cfg.window_size, pad_short=True)
    assert win.shape == (2, 10, 3)          # exactly one window per test unit
    assert list(units) == [1, 2]
    # short unit left-padded with its first cycle value (sensor s_2 first cycle = 1)
    assert win[0, 0, 0] == pytest.approx(1.0)
    # label at last cycle: RUL provided + 0 = provided value
    assert lab[0] == pytest.approx(7)
    assert lab[1] == pytest.approx(12)


def test_channel_scaler_fit_on_train_only():
    rng = np.random.default_rng(0)
    train = rng.normal(5, 2, size=(50, 8, 3)).astype(np.float32)
    mean, std = D.fit_channel_scaler(train)
    scaled = D.apply_channel_scaler(train, mean, std)
    assert mean.shape == (3,) and std.shape == (3,)
    assert np.allclose(scaled.reshape(-1, 3).mean(0), 0, atol=1e-4)
    assert np.allclose(scaled.reshape(-1, 3).std(0), 1, atol=1e-2)
