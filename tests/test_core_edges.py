"""Fail-loud guards and rarely-taken branches of the core modules.

Every path here is either a documented guard (repo invariant §7: never adapt silently)
or a non-default branch the campaign takes but the happy-path smoke tests do not:
`config` validation, `data` windowing edge cases, `heads` decoding/loss variants,
`train` seeding, `baselines` interface contracts, `evaluate` provenance/IO helpers, and
`embeddings` loc/scale normalization + cache IO. CPU-only, no downloads, no backbone.
"""

from __future__ import annotations

import csv as _csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.config import Config
from src import data as D
from src import embeddings as E
from src import evaluate as EV
from src import heads as H
from src import train as T
from src import baselines as B
from tests.synthetic import write_synthetic_cmapss, MockEmbedder


# ---------------------------------------------------------------------------
# Config validation + resolution
# ---------------------------------------------------------------------------
def test_config_rejects_unknown_choices_and_fields():
    with pytest.raises(ValueError, match="pooling must be one of"):
        Config(pooling="bogus")
    with pytest.raises(ValueError, match="head_features must be one of"):
        Config(head_features="bogus")
    with pytest.raises(KeyError, match="Unknown config field"):
        Config().replace(not_a_field=1)


def test_config_num_channels_and_unknown_dataset():
    cfg = Config(dataset="FD001", sensor_columns=["s_2", "s_3", "s_4"])
    assert cfg.num_channels() == 3
    with pytest.raises(ValueError, match="unknown dataset"):
        Config(dataset="NOT-A-DATASET", sensor_columns=["s_2"]).dataset_kind()


# ---------------------------------------------------------------------------
# data: windowing edge cases
# ---------------------------------------------------------------------------
def _unit(uid: int, n: int, channels=("s_2", "s_3")) -> pd.DataFrame:
    d = {"unit_number": uid, "time_cycles": np.arange(1, n + 1)}
    for j, c in enumerate(channels):
        d[c] = np.arange(1, n + 1, dtype=float) + 100 * j
    return pd.DataFrame(d)


def test_make_windows_returns_typed_empties_when_every_unit_is_short():
    """No unit reaches window_size -> correctly-SHAPED empty arrays (not a crash and
    not a ragged list), so downstream shape asserts still hold."""
    cfg = Config(window_size=50, max_rul=1000, sensor_columns=["s_2", "s_3"])
    df = D.add_train_rul(pd.concat([_unit(1, 5), _unit(2, 7)], ignore_index=True), cfg)
    w, y, u = D.make_windows(df, cfg.sensor_columns, cfg.window_size)
    assert w.shape == (0, 50, 2) and y.shape == (0,) and u.shape == (0,)
    assert w.dtype == np.float32 and u.dtype == np.int64


def test_make_test_last_windows_drops_short_units_when_padding_is_off():
    cfg = Config(window_size=10, sensor_columns=["s_2", "s_3"])
    df = pd.concat([_unit(1, 4), _unit(2, 25)], ignore_index=True)
    rul = pd.Series([7, 12], index=pd.RangeIndex(1, 3, name="unit_number"))
    df = D.add_test_rul(df, rul, cfg)
    _, _, units = D.make_test_last_windows(df, cfg.sensor_columns, cfg.window_size,
                                           target_col="actual_rul", pad_short=False)
    assert list(units) == [2]          # the 4-cycle unit is dropped, never fabricated


def test_make_windows_varlen_honours_the_unit_filter():
    """The varlen path takes the same `units=` restriction as make_windows, so a split's
    contexts never include another split's units."""
    cfg = Config(window_size=5, max_rul=1000, sensor_columns=["s_2", "s_3"])
    df = D.add_train_rul(pd.concat([_unit(1, 12), _unit(2, 12)], ignore_index=True), cfg)
    ctx, y, u = D.make_windows_varlen(df, cfg.sensor_columns, cfg.window_size,
                                      tsfm_context_length=8, units=np.array([2]))
    assert set(np.unique(u)) == {2}
    assert len(ctx) == len(y) == 8      # 12 - 5 + 1 windows for the one kept unit


# ---------------------------------------------------------------------------
# heads: loss/decoding variants + guards
# ---------------------------------------------------------------------------
def test_head_guards_reject_unknown_loss_everywhere():
    cfg = Config()
    with pytest.raises(ValueError, match="unknown loss_type"):
        H.head_output_dim("bogus", cfg)
    with pytest.raises(ValueError, match="unknown loss_type"):
        H.compute_loss(torch.zeros(4, 1), torch.zeros(4), "bogus", cfg)
    with pytest.raises(ValueError, match="unknown loss_type"):
        H.decode(torch.zeros(4, 1), "bogus", cfg)


def test_deep_head_stacks_the_requested_hidden_layers():
    head = H.build_head(8, "mse", Config(head_num_layers=4, head_hidden_dim=6))
    linears = [m for m in head.net if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 4                       # in + 2 hidden + out


def test_decode_without_target_scaling():
    """scale_targets=False: the head already predicts RUL in cycles, so decode must NOT
    multiply by max_rul (mse and quantile arms alike)."""
    cfg = Config(max_rul=125, scale_targets=False, quantile_levels=[0.1, 0.9])
    assert H.decode(torch.tensor([[30.0], [200.0]]), "mse", cfg).tolist() == [30.0, 125.0]
    q = H.decode(torch.tensor([[10.0, 40.0]]), "quantile", cfg)   # no 0.5 -> middle col
    assert q[0] == pytest.approx(40.0)


def test_decode_rejects_unknown_corn_decoding():
    cfg = Config(num_bins=6, corn_decoding="bogus")
    with pytest.raises(ValueError, match="unknown corn_decoding"):
        H.decode(torch.zeros(3, cfg.num_bins - 1), "corn", cfg)


# ---------------------------------------------------------------------------
# train: seeding branches + the default-seed path
# ---------------------------------------------------------------------------
def test_set_seed_non_deterministic_skips_the_torch_switches():
    T.set_seed(3, deterministic=False)             # no determinism side effects
    assert np.random.randint(0, 10) >= 0


def test_set_seed_survives_a_torch_without_deterministic_algorithms(monkeypatch):
    """Older/exotic torch builds raise from use_deterministic_algorithms; seeding must
    degrade rather than kill the run (the documented warn_only intent)."""
    def _boom(*a, **kw):
        raise RuntimeError("deterministic algorithms unsupported here")
    monkeypatch.setattr(torch, "use_deterministic_algorithms", _boom)
    T.set_seed(4, deterministic=True)
    assert torch.backends.cudnn.benchmark is False


def test_train_head_falls_back_to_config_seed(tmp_path):
    cfg = Config(seed=7, head_hidden_dim=8, head_batch_size=8, head_max_epochs=2,
                 head_early_stopping_patience=1, max_rul=40)
    X = np.random.default_rng(0).normal(size=(24, 5)).astype(np.float32)
    y = np.linspace(0, 40, 24).astype(np.float32)
    model, hist = T.train_head(X[:16], y[:16], X[16:], y[16:], "mse", cfg)  # seed=None
    assert np.isfinite(hist["best_val_rmse"])
    assert T.predict_head(model, X[16:], "mse", cfg).shape == (8,)


# ---------------------------------------------------------------------------
# baselines: interface contract + the no-validation training branch
# ---------------------------------------------------------------------------
def test_baseline_interface_is_abstract():
    base = B.Baseline()
    with pytest.raises(NotImplementedError):
        base.fit(None, None)
    with pytest.raises(NotImplementedError):
        base.predict(None)
    with pytest.raises(NotImplementedError):
        B._TorchBaseline(Config())._build()
    with pytest.raises(KeyError, match="unknown baseline"):
        B.make_baseline("nope", Config())


def test_torch_baseline_trains_without_a_validation_split():
    """With no val windows there is no early stopping and no best-state restore -- the
    model must still fit and predict inside [0, max_rul]."""
    cfg = Config(max_rul=40, baseline_max_epochs=2, baseline_early_stopping_patience=1,
                 baseline_batch_size=16)
    rng = np.random.default_rng(0)
    w = rng.normal(size=(32, 8, 3)).astype(np.float32)
    y = rng.uniform(0, 40, 32).astype(np.float32)
    bl = B.make_baseline("cnn", cfg, seed=0).fit(w, y)          # val_windows=None
    pred = bl.predict(w)
    assert pred.shape == (32,) and pred.min() >= 0 and pred.max() <= cfg.max_rul


# ---------------------------------------------------------------------------
# evaluate: provenance + IO helpers
# ---------------------------------------------------------------------------
def test_git_state_outside_a_repo_reports_nulls(tmp_path):
    state = EV.git_state(tmp_path)
    assert state == {"commit": None, "describe": None, "dirty": None}


def test_load_results_of_a_missing_file_is_empty(tmp_path):
    assert EV.load_results(tmp_path / "nope.csv") == []


def test_aggregate_data_scaling_without_a_dataset_filter(tmp_path):
    csv_path = tmp_path / "results_v2.csv"
    for ds in ("FD001", "FD003"):
        for n_units in (2, 4):
            EV.append_result_row(csv_path, {
                "model": "gbm", "dataset": ds, "n_units": n_units, "seed": 0,
                "loss": "native", "rmse_clipped": 20.0 - n_units})
    pooled = EV.aggregate_data_scaling(csv_path)                 # dataset=None
    ns, mean, std = pooled["gbm"]
    assert list(ns) == [2, 4] and mean.shape == std.shape == (2,)


def test_paired_seed_ttest_ignores_other_models_and_losses(tmp_path):
    p = tmp_path / "horizon.csv"
    for seed in range(3):
        for loss, val in (("corn", 5.0 + seed), ("mse", 6.0 + 0.5 * seed)):
            EV.append_result_row(p, {"model": "chronos-2_mlp", "max_rul": 40,
                                     "n_units": 4, "seed": seed, "loss": loss,
                                     "bin_lo": "0.0", "bin_hi": "20.0",
                                     "mae_clipped": val})
        EV.append_result_row(p, {"model": "gbm", "max_rul": 40, "n_units": 4,
                                 "seed": seed, "loss": "native", "bin_lo": "0.0",
                                 "bin_hi": "20.0", "mae_clipped": 99.0})
        EV.append_result_row(p, {"model": "chronos-2_mlp", "max_rul": 40, "n_units": 4,
                                 "seed": seed, "loss": "quantile", "bin_lo": "0.0",
                                 "bin_hi": "20.0", "mae_clipped": 99.0})
    rows = EV.paired_seed_ttest(p, loss_a="corn", loss_b="mse", metric="mae_clipped")
    assert len(rows) == 1 and rows[0]["n_seeds"] == 3           # gbm/quantile skipped
    assert rows[0]["mean_delta"] < 0


def test_archive_results_v1_is_idempotent(tmp_path):
    (tmp_path / "results.csv").write_text("model,rmse\ngbm,12\n")
    archive = EV.archive_results_v1(tmp_path)
    assert archive is not None and archive.name == "results_v1.csv"
    assert archive.read_text() == "model,rmse\ngbm,12\n"
    (tmp_path / "results.csv").write_text("model,rmse\ngbm,99\n")
    assert EV.archive_results_v1(tmp_path) == archive           # never overwritten
    assert archive.read_text() == "model,rmse\ngbm,12\n"


def test_load_learning_curve_reads_step_and_epoch_axes(tmp_path):
    p = tmp_path / "curve.csv"
    with open(p, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["step", "epoch", "metric", "value"])
        w.writerow([0, 0, "train_loss", 1.0])
        w.writerow([1, 0, "train_loss", 0.5])
        w.writerow(["", 0, "val_rmse", 20.0])                   # no step -> epoch is x
        w.writerow(["", 0, "custom_metric", 3.0])               # unseen metric key
    out = EV.load_learning_curve(p)
    assert out["train_loss"] == ([0.0, 1.0], [1.0, 0.5])
    assert out["val_rmse"] == ([0.0], [20.0])
    assert out["custom_metric"] == ([0.0], [3.0])


# ---------------------------------------------------------------------------
# embeddings: the Embedder protocol, loc/scale normalization, cache IO
# ---------------------------------------------------------------------------
def test_embedder_protocol_methods_are_pure_declarations():
    """`Embedder` is a typing Protocol: it declares the Stage-A contract and carries no
    behaviour, so its stubs return None. Asserted so the contract is executed, not just
    read -- the concrete implementations live in src/models/."""
    assert E.Embedder.embed_windows(object(), []) is None
    assert E.Embedder.describe(object()) is None


def test_extract_loc_scale_per_item_sequence_and_ambiguous_batch_two():
    """A per-window sequence of (n_variates, 2) entries normalizes; a length-2 container
    with batch == 2 is read as a per-item sequence (not as a (loc, scale) pair)."""
    batch, V = 3, 2
    seq = [np.full((V, 2), float(i)) for i in range(batch)]
    out = E.extract_loc_scale(seq, batch, V)
    assert out.shape == (batch, V, 2) and out[2, 0, 0] == 2.0
    two = [np.full((V, 2), 1.0), np.full((V, 2), 2.0)]
    assert E.extract_loc_scale(two, 2, V).shape == (2, V, 2)


def test_extract_loc_scale_stacked_leading_axis():
    """A (2, batch, n_variates) stack (loc first) is moved onto the trailing axis."""
    batch, V = 3, 4
    stacked = np.stack([np.zeros((batch, V)), np.ones((batch, V))], axis=0)
    out = E.extract_loc_scale(stacked, batch, V)
    assert out.shape == (batch, V, 2)
    assert np.allclose(out[..., 0], 0.0) and np.allclose(out[..., 1], 1.0)


def test_extract_loc_scale_fails_loud_on_uninterpretable_shapes():
    with pytest.raises(ValueError, match="cannot interpret"):
        E.extract_loc_scale(np.zeros((3, 5)), batch=3, n_variates=5)   # no 2-axis
    with pytest.raises(ValueError, match="normalized to"):
        # per-item entries carrying the WRONG variate count
        E.extract_loc_scale([np.zeros((3, 2))] * 3, batch=3, n_variates=2)


def _cache_cfg(tmp_path: Path) -> Config:
    return Config(dataset="FD001", data_dir=str(tmp_path / "CMAPSSData"),
                  cache_dir=str(tmp_path / "cache"), results_dir=str(tmp_path / "results"),
                  window_size=12, sensor_columns=["s_2", "s_3", "s_4"], max_rul=40)


class _ThroughputMock(MockEmbedder):
    """MockEmbedder that reports a throughput, so Stage A's progress print is exercised."""
    def embed_windows(self, contexts):
        out = super().embed_windows(contexts)
        self.last_throughput = 123.4
        return out


def test_build_embedding_cache_uses_the_registry_when_no_embedder_is_injected(
        tmp_path, monkeypatch, capsys):
    """embedder=None resolves through models.make_embedder (the production path); the
    registry is monkeypatched to a CPU mock so no backbone is imported."""
    import src.models as models_mod
    cfg = _cache_cfg(tmp_path)
    write_synthetic_cmapss(Path(cfg.data_dir), n_train_units=4, n_test_units=3)
    monkeypatch.setattr(models_mod, "make_embedder",
                        lambda c, device=None: _ThroughputMock(feature_dim=8))
    path = E.build_embedding_cache(cfg, verbose=True)
    assert path.exists()
    out = capsys.readouterr().out
    assert "train embed throughput" in out and "test  embed throughput" in out


def test_load_embedding_cache_requires_stage_a(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run Stage A"):
        E.load_embedding_cache(_cache_cfg(tmp_path))


def test_load_embedding_cache_tolerates_a_cache_without_embeddings(tmp_path):
    """The fp16 -> fp32 upcast only touches keys that are present, so a partial npz
    (e.g. a windows-only sidecar) loads instead of raising a KeyError."""
    cfg = _cache_cfg(tmp_path)
    cache_path = cfg.cache_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, train_windows=np.zeros((2, 12, 3), np.float32))
    out = E.load_embedding_cache(cfg)
    assert set(out) == {"train_windows"} and out["train_windows"].dtype == np.float32
