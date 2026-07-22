"""CPU tests for the four v2 TSFM embedders + the RQ-M fairness machinery (§34/§35).

No backbone is ever imported: the shared plain-patch base (``models/base.py``) is
exercised through a fake ``_encode_batch`` returning synthetic canonical tensors, the
concrete embedders are checked only for identity/registry/describe (their backbone
methods are the sanctioned ``# pragma: no cover`` boundary), and
``run_representation_fairness`` runs on two differently-shaped ``MockEmbedder``s.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from src.config import Config
from src import embeddings as E
from src.models import (EMBEDDERS, make_embedder, ChronosEmbedder, MoiraiEmbedder,
                        MomentEmbedder, TimesFMEmbedder, TTMEmbedder)
from src.models.base import TSFMEmbedderBase
from tests.synthetic import write_synthetic_cmapss, MockEmbedder


# ---------------------------------------------------------------------------
# Pooling contract across BOTH layout kinds (n_special_tokens 2 vs 0)
# ---------------------------------------------------------------------------
def test_pool_patches_chronos_layout_maps_special_tokens():
    """n_special_tokens=2 (Chronos-2): content = [:-2], forecast=-1, last_content=-3."""
    V, P, Dm = 3, 6, 4  # 4 content + REG(-2) + forecast(-1)
    emb = np.arange(V * P * Dm, dtype=np.float32).reshape(V, P, Dm)
    assert np.array_equal(E.pool_patches(emb, "forecast_token", 2), emb[:, -1, :])
    assert np.array_equal(E.pool_patches(emb, "last_content", 2), emb[:, -3, :])
    assert np.allclose(E.pool_patches(emb, "mean", 2), emb[:, :-2, :].mean(axis=1))
    assert np.array_equal(E.pool_patches(emb, "flatten", 2),
                          emb[:, :-2, :].reshape(V, -1))


def test_pool_patches_plain_layout_has_no_special_tokens():
    """n_special_tokens=0 (MOMENT/TimesFM/Moirai/TTM): every position is content, so
    forecast_token == last_content == the last patch, and mean is over ALL patches."""
    V, P, Dm = 3, 5, 4
    emb = np.arange(V * P * Dm, dtype=np.float32).reshape(V, P, Dm)
    assert np.array_equal(E.pool_patches(emb, "forecast_token", 0), emb[:, -1, :])
    assert np.array_equal(E.pool_patches(emb, "last_content", 0), emb[:, -1, :])
    assert np.allclose(E.pool_patches(emb, "mean", 0), emb.mean(axis=1))
    assert np.array_equal(E.pool_patches(emb, "flatten", 0), emb.reshape(V, -1))


def test_pool_patches_needs_a_content_patch():
    emb = np.zeros((3, 2, 4), np.float32)     # 2 positions, 2 special -> 0 content
    with pytest.raises(ValueError, match="content patch"):
        E.pool_patches(emb, "mean", 2)
    with pytest.raises(ValueError):
        E.pool_patches(np.zeros((3, 4), np.float32), "mean", 0)   # not 3-D
    with pytest.raises(ValueError, match="pooling"):
        E.pool_patches(np.zeros((3, 4, 2), np.float32), "bogus", 0)


def test_channel_aggregation_concat_vs_mean_dims():
    V, P, Dm = 4, 5, 6
    emb = np.random.default_rng(0).normal(size=(V, P, Dm)).astype(np.float32)
    concat = E.pool_window_embedding(emb, "forecast_token", "concat", 0)
    mean = E.pool_window_embedding(emb, "forecast_token", "mean", 0)
    assert concat.shape == (V * Dm,)
    assert mean.shape == (Dm,)
    assert np.allclose(mean, emb[:, -1, :].mean(axis=0))
    with pytest.raises(ValueError, match="channel_aggregation"):
        E.aggregate_variates(emb[:, -1, :], "bogus")


def test_torch_pooling_matches_numpy_all_layouts():
    import torch
    rng = np.random.default_rng(1)
    emb = rng.normal(size=(3, 6, 5)).astype(np.float32)
    for n_special in (0, 2):
        for strat in ["forecast_token", "last_content", "mean", "flatten"]:
            for agg in ["concat", "mean"]:
                np_out = E.pool_window_embedding(emb, strat, agg, n_special)
                t_out = E._pool_one_torch(torch.from_numpy(emb), strat, agg,
                                          n_special).numpy()
                assert np.allclose(np_out, t_out, atol=1e-5), (strat, agg, n_special)
    with pytest.raises(ValueError, match="channel_aggregation"):
        E._pool_one_torch(torch.from_numpy(emb), "mean", "bogus", 0)
    with pytest.raises(ValueError, match="pooling"):
        E._pool_one_torch(torch.from_numpy(emb), "bogus", "concat", 0)


# ---------------------------------------------------------------------------
# Shared base embed_windows (fake backbone -> no import)
# ---------------------------------------------------------------------------
class _FakeEmbedder(TSFMEmbedderBase):
    """Plain-patch fake: returns a fixed (C, patches, d) canonical tensor per window,
    so the base's batching/pooling/assembly path runs without any backbone."""
    embedder_name = "_FakeEmbedder"

    def __init__(self, config, device=None, patches=3, d_model=5):
        super().__init__(config, device=device)
        self._patches, self._d = patches, d_model

    def _encode_batch(self, batch):
        canon = [np.random.default_rng(i).normal(
            size=(w.shape[1], self._patches, self._d)).astype(np.float32)
            for i, w in enumerate(batch)]
        return canon, self.loc_scale_from_contexts(batch)


def _ctx(n=5, length=6, channels=4, seed=1):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(length, channels)).astype(np.float32) for _ in range(n)]


def test_base_embed_windows_concat_and_locscale_shape():
    cfg = Config(dataset="FD001", embed_batch_size=2)  # forces >1 batch over 5 windows
    fe = _FakeEmbedder(cfg, d_model=5)
    ctx = _ctx(n=5, channels=4)
    emb, ls = fe.embed_windows(ctx)
    assert emb.shape == (5, 4 * 5)               # concat: n_variates * d_model
    assert ls.shape == (5, 4, 2)                 # per-channel [mean, std]
    assert fe.last_throughput is not None
    # loc/scale is the input mean/std (fallback), so it matches the context stats
    assert np.allclose(ls[0, :, 0], ctx[0].mean(axis=0), atol=1e-5)
    assert np.allclose(ls[0, :, 1], ctx[0].std(axis=0), atol=1e-5)


def test_base_embed_windows_mean_aggregation_shrinks_F():
    cfg = Config(dataset="FD001", channel_aggregation="mean")
    emb, _ = _FakeEmbedder(cfg, d_model=5).embed_windows(_ctx(n=4, channels=6))
    assert emb.shape == (4, 5)                    # mean over variates -> d_model


def test_base_embed_windows_empty():
    emb, ls = _FakeEmbedder(Config(dataset="FD001")).embed_windows([])
    assert emb.shape == (0, 0) and ls.shape == (0, 0, 2)


def test_base_accepts_fixed_array_input():
    cfg = Config(dataset="FD001")
    arr = np.random.default_rng(0).normal(size=(3, 6, 4)).astype(np.float32)
    emb, ls = _FakeEmbedder(cfg, d_model=5).embed_windows(arr)
    assert emb.shape == (3, 4 * 5) and ls.shape == (3, 4, 2)


# ---------------------------------------------------------------------------
# Registry + describe (concrete embedders, no backbone touched)
# ---------------------------------------------------------------------------
def test_all_five_embedders_registered_and_selectable():
    expected = {
        "amazon/chronos-2": ChronosEmbedder,
        "Salesforce/moirai-2": MoiraiEmbedder,
        "AutonLab/MOMENT-1-large": MomentEmbedder,
        "google/timesfm-2.5": TimesFMEmbedder,
        "ibm-granite/granite-timeseries-ttm-r2": TTMEmbedder,
    }
    assert EMBEDDERS == expected
    for name, cls in expected.items():
        e = make_embedder(Config(dataset="FD001", model_name=name))
        assert isinstance(e, cls)


def test_embedder_registry_never_drifts():
    """Every registered class is a real embedder exposing the protocol, and
    make_embedder round-trips every key -- the models twin of the datasets drift test."""
    for name, cls in EMBEDDERS.items():
        e = make_embedder(Config(dataset="FD001", model_name=name))
        assert hasattr(e, "embed_windows") and hasattr(e, "describe")
        assert e.__class__ is cls
        d = e.describe()
        assert d["model_name"] == name
        assert set(d) >= {"embedder", "model_name", "pooling", "channel_aggregation"}


def test_make_embedder_unknown_raises():
    with pytest.raises(KeyError, match="no embedder registered"):
        make_embedder(Config(dataset="FD001", model_name="nope/not-a-model"))


def test_config_channel_aggregation_guard():
    Config(dataset="FD001", channel_aggregation="mean")   # valid
    with pytest.raises(ValueError, match="channel_aggregation"):
        Config(dataset="FD001", channel_aggregation="bogus")


def test_config_noise_injection_kind_guard():
    Config(dataset="FD001", noise_injection={"kind": "drift", "magnitude": 2})  # valid
    with pytest.raises(ValueError, match="noise_injection"):
        Config(dataset="FD001", noise_injection={"kind": "bogus"})


def test_new_embedder_describe_reflects_layout_and_aggregation():
    c = Config(dataset="FD001", model_name="AutonLab/MOMENT-1-large",
               channel_aggregation="mean", pooling="mean")
    d = make_embedder(c).describe()
    assert d["layout"] == "univariate" and d["n_special_tokens"] == 0
    assert d["channel_aggregation"] == "mean" and d["pooling"] == "mean"
    # multivariate-native members carry the multivariate label
    for name in ("Salesforce/moirai-2", "ibm-granite/granite-timeseries-ttm-r2"):
        assert make_embedder(Config(dataset="FD001", model_name=name)).describe()[
            "layout"] == "multivariate"


# ---------------------------------------------------------------------------
# Chronos channel_aggregation is threaded (describe records it; key changes)
# ---------------------------------------------------------------------------
def test_chronos_threads_channel_aggregation():
    c = Config(dataset="FD001", channel_aggregation="mean")
    assert make_embedder(c).describe()["channel_aggregation"] == "mean"
    # the mean control re-keys the embedding cache (RQ-M control is a distinct cache)
    assert (c.embedding_cache_key()
            != Config(dataset="FD001").embedding_cache_key())


# ---------------------------------------------------------------------------
# RQ-M representation fairness (two differently-shaped mock layouts)
# ---------------------------------------------------------------------------
def _fairness_cfg(tmp_path: Path) -> Config:
    return Config(
        dataset="FD001", data_dir=str(tmp_path / "CMAPSSData"),
        cache_dir=str(tmp_path / "cache"), results_dir=str(tmp_path / "results"),
        window_size=12, sensor_columns=["s_2", "s_3", "s_4", "s_7", "s_9"],
        max_rul=40, head_hidden_dim=16, head_batch_size=32, head_max_epochs=3,
        head_early_stopping_patience=2,
    )


def _layout_factory(cfg: Config):
    """A mock whose layout mirrors the real model family, and whose aggregation
    tracks cfg.channel_aggregation -- so native/common genuinely change the width."""
    univariate = cfg.model_name in ("AutonLab/MOMENT-1-large", "google/timesfm-2.5")
    return MockEmbedder(feature_dim=8,
                        layout="univariate" if univariate else "multivariate",
                        channel_aggregation=cfg.channel_aggregation)


def test_representation_fairness_runs_native_and_common(tmp_path):
    from src.sweep import run_representation_fairness
    cfg = _fairness_cfg(tmp_path)
    write_synthetic_cmapss(Path(cfg.data_dir), n_train_units=6, n_test_units=4)
    out = run_representation_fairness(
        cfg, models=["amazon/chronos-2", "AutonLab/MOMENT-1-large"],
        seeds=[0, 1], embedder_factory=_layout_factory)
    rows = list(csv.DictReader(open(out)))
    modes = {r["mode"] for r in rows}
    aggs = {r["channel_aggregation"] for r in rows}
    assert modes == {"native", "common"}
    assert aggs == {"concat", "mean"}
    # 2 models x 2 modes x 2 seeds
    assert len(rows) == 8
    for r in rows:
        assert np.isfinite(float(r["rmse_clipped"]))
        assert r["loss"] == "mse"


def test_representation_fairness_is_restartable(tmp_path):
    from src.sweep import run_representation_fairness
    cfg = _fairness_cfg(tmp_path)
    write_synthetic_cmapss(Path(cfg.data_dir), n_train_units=6, n_test_units=4)
    kw = dict(models=["amazon/chronos-2"], seeds=[0], embedder_factory=_layout_factory)
    out = run_representation_fairness(cfg, **kw)
    n1 = len(list(csv.DictReader(open(out))))
    run_representation_fairness(cfg, **kw)              # rerun skips completed cells
    n2 = len(list(csv.DictReader(open(out))))
    assert n1 == n2 == 2                                 # 1 model x 2 modes x 1 seed
