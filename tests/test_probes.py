"""Factor-probe harness + sim-only noise injection (§38)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src import data as D
from src import probes as P
from tests.synthetic import write_synthetic_cmapss, MockEmbedder


# ---------------------------------------------------------------------------
# noise injection: sim-only guard, determinism, per-kind effect, cache key
# ---------------------------------------------------------------------------
def _toy_df(n_units=3, n_cycles=10, channels=("s_2", "s_3", "s_4"), seed=0):
    rng = np.random.default_rng(seed)
    blocks = []
    for u in range(1, n_units + 1):
        d = {"unit_number": u, "time_cycles": np.arange(1, n_cycles + 1)}
        for c in channels:
            d[c] = rng.normal(500, 10, n_cycles)
        blocks.append(pd.DataFrame(d))
    return pd.concat(blocks, ignore_index=True)


def test_noise_guard_raises_on_real_dataset():
    cfg = Config(dataset="XJTU-SY", noise_injection={"kind": "gaussian", "snr_db": 20})
    # guard fires on is_simulated_dataset() before touching the frames
    with pytest.raises(ValueError, match="sim-only"):
        D.apply_noise_injection(pd.DataFrame(), pd.DataFrame(), cfg)


def test_noise_allowed_on_simulated_kinds():
    for ds in ("FD001", "DS02"):
        assert Config(dataset=ds).is_simulated_dataset()
    assert not Config(dataset="XJTU-SY").is_simulated_dataset()


def test_gaussian_noise_is_deterministic_and_changes_signal():
    cols = ["s_2", "s_3", "s_4"]
    df = _toy_df(channels=cols)
    spec = {"kind": "gaussian", "snr_db": 10}
    a = D.inject_noise(df, cols, spec, seed=7)
    b = D.inject_noise(df, cols, spec, seed=7)
    c = D.inject_noise(df, cols, spec, seed=8)
    assert np.array_equal(a[cols].to_numpy(), b[cols].to_numpy())      # same seed => same series
    assert not np.allclose(a[cols].to_numpy(), df[cols].to_numpy())    # signal perturbed
    assert not np.array_equal(a[cols].to_numpy(), c[cols].to_numpy())  # different seed differs
    # labels / index columns untouched
    assert np.array_equal(a["time_cycles"].to_numpy(), df["time_cycles"].to_numpy())


def test_drift_grows_toward_end_of_unit():
    cols = ["s_2", "s_3", "s_4"]
    df = _toy_df(n_units=1, n_cycles=20, channels=cols)
    out = D.inject_noise(df, cols, {"kind": "drift", "magnitude": 5.0}, seed=0)
    delta = (out[cols].to_numpy() - df[cols].to_numpy())
    # the ramp is 0 at the first cycle and largest at the last
    assert np.allclose(delta[0], 0.0, atol=1e-9)
    assert np.all(np.abs(delta[-1]) > np.abs(delta[1]))


def test_dropout_blanks_some_entries():
    cols = ["s_2", "s_3", "s_4"]
    df = _toy_df(n_units=4, n_cycles=25, channels=cols)
    out = D.inject_noise(df, cols, {"kind": "dropout", "rate": 0.5}, seed=1)
    zeros = (out[cols].to_numpy() == 0.0).mean()
    assert 0.3 < zeros < 0.7                     # ~half blanked at rate 0.5


def test_noise_rekeys_windows_only_when_set():
    base = Config(dataset="FD001")
    noisy = Config(dataset="FD001", noise_injection={"kind": "gaussian", "snr_db": 20})
    assert noisy.window_cache_key() != base.window_cache_key()
    assert noisy.embedding_cache_key() != base.embedding_cache_key()
    assert Config(dataset="FD001", noise_injection={}).window_cache_key() == base.window_cache_key()


def test_load_prepared_applies_noise(tmp_path):
    clean = Config(dataset="FD001", data_dir=str(tmp_path / "CMAPSSData"),
                   sensor_columns=["s_2", "s_3", "s_4"], max_rul=40)
    write_synthetic_cmapss(Path(clean.data_dir), n_train_units=4, n_test_units=3)
    noisy = clean.replace(noise_injection={"kind": "gaussian", "snr_db": 5})
    tr_clean, _ = D.load_prepared(clean)
    tr_noisy, _ = D.load_prepared(noisy)
    assert not np.allclose(tr_clean[clean.sensor_columns].to_numpy(),
                           tr_noisy[noisy.sensor_columns].to_numpy())
    # RUL labels are unaffected by the perturbation
    assert np.array_equal(tr_clean["clipped_rul"].to_numpy(),
                          tr_noisy["clipped_rul"].to_numpy())


# ---------------------------------------------------------------------------
# _level_overrides
# ---------------------------------------------------------------------------
def test_level_overrides_channels_noise_generic():
    assert P._level_overrides("channels", ["s_2", "s_3"]) == {"sensor_columns": ["s_2", "s_3"]}
    assert P._level_overrides("noise", {"kind": "gaussian"}) == {
        "noise_injection": {"kind": "gaussian"}}
    assert P._level_overrides("aggregation", {"window_size": 20}) == {"window_size": 20}
    with pytest.raises(ValueError, match="dict of config overrides"):
        P._level_overrides("aggregation", 5)


# ---------------------------------------------------------------------------
# run_factor_probe
# ---------------------------------------------------------------------------
def _probe_cfg(tmp_path: Path) -> Config:
    return Config(
        dataset="FD001", data_dir=str(tmp_path / "CMAPSSData"),
        cache_dir=str(tmp_path / "cache"), results_dir=str(tmp_path / "results"),
        window_size=12, sensor_columns=["s_2", "s_3", "s_4", "s_7", "s_9"],
        max_rul=40, head_hidden_dim=16, head_batch_size=32, head_max_epochs=3,
        head_early_stopping_patience=2, sweep_seeds=[0, 1])


def test_run_factor_probe_channels(tmp_path):
    cfg = _probe_cfg(tmp_path)
    write_synthetic_cmapss(Path(cfg.data_dir), n_train_units=6, n_test_units=4)
    out = P.run_factor_probe(
        cfg, factor="channels",
        levels={"all": ["s_2", "s_3", "s_4", "s_7", "s_9"], "few": ["s_2", "s_3"]},
        models=["amazon/chronos-2"], baselines=["predict_mean"], seeds=[0],
        embedder_factory=lambda c: MockEmbedder(feature_dim=16))
    rows = list(csv.DictReader(open(out)))
    assert out.name == "probe_channels.csv"
    assert {r["factor"] for r in rows} == {"channels"}
    assert {r["level"] for r in rows} == {"all", "few"}
    assert {r["model"] for r in rows} == {"chronos-2_mlp", "predict_mean"}
    for r in rows:
        assert np.isfinite(float(r["nasa_clipped"]))


def test_run_factor_probe_noise_levels_use_distinct_caches(tmp_path):
    cfg = _probe_cfg(tmp_path)
    write_synthetic_cmapss(Path(cfg.data_dir), n_train_units=6, n_test_units=4)
    out = P.run_factor_probe(
        cfg, factor="noise",
        levels={"snr20": {"kind": "gaussian", "snr_db": 20},
                "snr5": {"kind": "gaussian", "snr_db": 5}},
        models=["amazon/chronos-2"], baselines=["predict_mean"], seeds=[0],
        embedder_factory=lambda c: MockEmbedder(feature_dim=16))
    rows = list(csv.DictReader(open(out)))
    assert {r["level"] for r in rows} == {"snr20", "snr5"}
    # two SNR levels => two distinct embedding caches on disk
    caches = list((Path(cfg.cache_dir)).glob("emb_FD001_*.npz"))
    assert len(caches) == 2


def test_run_factor_probe_restartable(tmp_path):
    cfg = _probe_cfg(tmp_path)
    write_synthetic_cmapss(Path(cfg.data_dir), n_train_units=6, n_test_units=4)
    kw = dict(factor="channels", levels={"all": ["s_2", "s_3", "s_4", "s_7", "s_9"]},
              models=["amazon/chronos-2"], baselines=["predict_mean"], seeds=[0],
              embedder_factory=lambda c: MockEmbedder(feature_dim=16))
    out = P.run_factor_probe(cfg, **kw)
    n1 = len(list(csv.DictReader(open(out))))
    P.run_factor_probe(cfg, **kw)                        # rerun: skip completed cells
    assert len(list(csv.DictReader(open(out)))) == n1


# ---------------------------------------------------------------------------
# probe_roster
# ---------------------------------------------------------------------------
def test_probe_roster_picks_best_per_category(tmp_path):
    from src.evaluate import append_result_row
    csv_path = tmp_path / "tier1_results_v2.csv"
    # lower nasa_clipped = better; give each model a distinct mean
    scores = {"chronos-2_mlp": 10, "moirai-2_mlp": 12, "moment-1-large_mlp": 15,
              "gbm": 20, "minirocket": 18, "catch22_gbm": 25,
              "cnn": 30, "lstm": 28, "predict_mean": 50}
    for model, nasa in scores.items():
        for seed in (0, 1):
            append_result_row(csv_path, {"model": model, "dataset": "FD001",
                                         "n_units": 100, "seed": seed, "loss": "mse",
                                         "nasa_clipped": nasa + seed})
    top2_tsfm, top2_foil, best_nn = P.probe_roster(csv_path)
    assert top2_tsfm == ["chronos-2_mlp", "moirai-2_mlp"]      # 2 best TSFMs
    assert top2_foil == ["minirocket", "gbm"]                  # 2 best foils
    assert best_nn == "lstm"                                   # best NN (28 < 30)


def test_probe_roster_handles_missing_categories(tmp_path):
    from src.evaluate import append_result_row
    csv_path = tmp_path / "only_tsfm_results_v2.csv"
    append_result_row(csv_path, {"model": "chronos-2_mlp", "dataset": "FD001",
                                 "n_units": 100, "seed": 0, "loss": "mse",
                                 "nasa_clipped": 10})
    # a row missing the ranking metric is skipped, not crashed on
    append_result_row(csv_path, {"model": "gbm", "dataset": "FD001", "n_units": 100,
                                 "seed": 0, "loss": "native", "nasa_clipped": ""})
    top2_tsfm, top2_foil, best_nn = P.probe_roster(csv_path)
    assert top2_tsfm == ["chronos-2_mlp"] and top2_foil == [] and best_nn is None
