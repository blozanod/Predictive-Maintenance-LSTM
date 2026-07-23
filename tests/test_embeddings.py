"""Embedding/pooling/cache correctness for the Task 1/2 fixes:

  * pooling excludes the 2 special tokens (REG, forecast) from content poolings,
    and indexes forecast_token (-1) / last_content (-3) correctly;
  * the on-device torch pooling matches the numpy reference;
  * the variable-length TSFM path yields the SAME label/unit alignment as the fixed
    window path (so the head trains on the same rows the baselines do);
  * the float16 embedding cache round-trips losslessly on load;
  * evaluate reports BOTH test-label protocols.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.config import Config
from src import embeddings as E
from src import data as D
from src.evaluate import evaluate_predictions, rmse
from tests.synthetic import write_synthetic_cmapss, MockEmbedder


# ---------------------------------------------------------------------------
# Pooling excludes special tokens (Task 1.3)
# ---------------------------------------------------------------------------
def test_pooling_excludes_special_tokens():
    V, P, D_ = 3, 6, 4  # P = 4 content patches + REG(-2) + forecast(-1)
    emb = np.arange(V * P * D_, dtype=np.float32).reshape(V, P, D_)

    ft = E.pool_window_embedding(emb, "forecast_token")
    assert np.array_equal(ft, emb[:, -1, :].reshape(-1))        # masked output patch

    lc = E.pool_window_embedding(emb, "last_content")
    assert np.array_equal(lc, emb[:, -3, :].reshape(-1))        # last real content patch

    mn = E.pool_window_embedding(emb, "mean")
    assert np.allclose(mn, emb[:, :-2, :].mean(axis=1).reshape(-1))  # content only
    # the fix: mean must NOT average the 2 special tokens in
    assert not np.allclose(mn, emb.mean(axis=1).reshape(-1))

    fl = E.pool_window_embedding(emb, "flatten")
    assert fl.shape[0] == V * (P - 2) * D_                      # content patches only


def test_pooling_needs_min_positions():
    emb = np.zeros((3, 2, 4), np.float32)  # only 2 positions: no content patch left
    with pytest.raises(ValueError):
        E.pool_window_embedding(emb, "last_content")


# ---------------------------------------------------------------------------
# loc/scale normalization (defensive extractor)
# ---------------------------------------------------------------------------
def test_extract_loc_scale_shapes():
    batch, V = 4, 3
    # single (batch, V, 2) tensor-like
    a = np.random.default_rng(0).normal(size=(batch, V, 2)).astype(np.float32)
    out = E.extract_loc_scale(a, batch, V)
    assert out.shape == (batch, V, 2) and np.allclose(out, a)
    # (loc, scale) pair
    loc = np.ones((batch, V), np.float32)
    scale = 2 * np.ones((batch, V), np.float32)
    out2 = E.extract_loc_scale((loc, scale), batch, V)
    assert out2.shape == (batch, V, 2)
    assert np.allclose(out2[..., 0], 1) and np.allclose(out2[..., 1], 2)


# ---------------------------------------------------------------------------
# Variable-length TSFM path == fixed path alignment (Task 1.2)
# ---------------------------------------------------------------------------
def _unit(uid, n, nch=4):
    import pandas as pd
    rows = {"unit_number": uid, "time_cycles": np.arange(1, n + 1)}
    for j in range(nch):
        rows[f"s_{j+2}"] = np.arange(1, n + 1, dtype=float) + 100 * j
    return pd.DataFrame(rows)


def test_varlen_matches_fixed_label_alignment():
    import pandas as pd
    cfg = Config(window_size=5, max_rul=1000)
    df = pd.concat([_unit(1, 20), _unit(2, 13)], ignore_index=True)
    df = D.add_train_rul(df, cfg)
    cols = ["s_2", "s_3", "s_4", "s_5"]

    fw, fy, fu = D.make_windows(df, cols, cfg.window_size, "clipped_rul")
    ctx, vy, vu = D.make_windows_varlen(df, cols, cfg.window_size, tsfm_context_length=8,
                                        target_col="clipped_rul")
    # same number of windows, identical labels + unit ids, identical order
    assert len(ctx) == len(fw)
    assert np.array_equal(fy, vy)
    assert np.array_equal(fu, vu)
    # each context ends at the SAME last cycle as the fixed window and is capped at 8
    for i in range(len(ctx)):
        assert ctx[i].shape[0] <= 8
        assert np.array_equal(ctx[i][-1], fw[i][-1])   # same last-cycle sensors


def test_varlen_context_length_caps_history():
    cfg = Config(window_size=5)
    df = D.add_train_rul(_unit(1, 30), cfg)
    cols = ["s_2", "s_3", "s_4", "s_5"]
    ctx, _, _ = D.make_windows_varlen(df, cols, cfg.window_size, tsfm_context_length=10,
                                      target_col="clipped_rul")
    # first prediction cycle (=window_size=5) has only 5 cycles of history;
    # later cycles saturate at the cap of 10.
    assert ctx[0].shape[0] == 5
    assert ctx[-1].shape[0] == 10


def test_test_contexts_not_padded_and_aligned():
    import pandas as pd
    cfg = Config(window_size=10)
    df = pd.concat([_unit(1, 4), _unit(2, 25)], ignore_index=True)  # unit 1 short
    rul = pd.Series([7, 12], index=pd.RangeIndex(1, 3, name="unit_number"))
    df = D.add_test_rul(df, rul, cfg)
    cols = ["s_2", "s_3", "s_4", "s_5"]
    fw, fy, fu = D.make_test_last_windows(df, cols, cfg.window_size,
                                          target_col="actual_rul", pad_short=True)
    ctx, vy, vu = D.make_test_last_contexts(df, cols, tsfm_context_length=cfg.window_size,
                                            target_col="actual_rul")
    assert np.array_equal(fu, vu) and np.array_equal(fy, vy)  # same units + labels
    # fixed path fabricates cycles for the short unit; varlen keeps its true length
    assert fw[0].shape[0] == 10
    assert ctx[0].shape[0] == 4     # NOT padded (Task 1.4 padding hazard fix)
    assert ctx[1].shape[0] == 10


# ---------------------------------------------------------------------------
# float16 cache round-trip (Task 2)
# ---------------------------------------------------------------------------
def _cache_cfg(tmp_path: Path, **kw) -> Config:
    return Config(
        dataset="FD001", data_dir=str(tmp_path / "CMAPSSData"),
        cache_dir=str(tmp_path / "cache"), results_dir=str(tmp_path / "results"),
        window_size=12, sensor_columns=["s_2", "s_3", "s_4", "s_7", "s_9"],
        max_rul=40, **kw,
    )


def test_fp16_cache_roundtrip(tmp_path):
    cfg = _cache_cfg(tmp_path)  # embedding_storage_dtype defaults to float16
    write_synthetic_cmapss(Path(cfg.data_dir), n_train_units=6, n_test_units=4)
    path = E.build_embedding_cache(cfg, embedder=MockEmbedder(feature_dim=16))
    with np.load(path) as z:
        stored_emb = z["train_emb"]
        stored_win = z["train_windows"]
        stored_ls = z["train_locscale"]
    assert stored_emb.dtype == np.float16       # embeddings stored fp16 (I/O saver)
    assert stored_win.dtype == np.float32        # raw windows stay fp32
    assert stored_ls.dtype == np.float32         # loc/scale stays fp32
    loaded = E.load_embedding_cache(cfg)
    assert loaded["train_emb"].dtype == np.float32          # upcast on load
    assert np.array_equal(loaded["train_emb"], stored_emb.astype(np.float32))  # lossless upcast


def test_fp32_cache_option(tmp_path):
    cfg = _cache_cfg(tmp_path, embedding_storage_dtype="float32")
    write_synthetic_cmapss(Path(cfg.data_dir), n_train_units=6, n_test_units=4)
    path = E.build_embedding_cache(cfg, embedder=MockEmbedder(feature_dim=16))
    with np.load(path) as z:
        assert z["train_emb"].dtype == np.float32


# ---------------------------------------------------------------------------
# Both-protocol metrics (Task 1.4)
# ---------------------------------------------------------------------------
def test_evaluate_both_protocols():
    y_true = np.array([150.0, 100.0, 50.0])   # first unit's true RUL exceeds max_rul
    y_pred = np.array([125.0, 90.0, 55.0])
    m = evaluate_predictions(y_true, y_pred, max_rul=125)
    assert m["rmse_clipped"] == pytest.approx(rmse(np.array([125.0, 100.0, 50.0]), y_pred))
    assert m["rmse_unclipped"] == pytest.approx(rmse(y_true, y_pred))
    assert m["rmse_unclipped"] > m["rmse_clipped"]  # unclipped protocol is inflated
    assert m["n"] == 3


# ---------------------------------------------------------------------------
# MockEmbedder layout / channel-aggregation parametrization (M0.3)
# ---------------------------------------------------------------------------
def _ctx(n_windows=4, length=6, n_channels=5, seed=1):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(length, n_channels)).astype(np.float32) for _ in range(n_windows)]


def test_mock_default_is_backward_compatible():
    """Defaults (multivariate, concat) reproduce the ORIGINAL fixture byte-for-byte:
    F == feature_dim regardless of channel count, and the exact tanh(mean @ proj)."""
    C, F = 5, 16
    ctx = _ctx(n_windows=3, n_channels=C)
    emb, ls = MockEmbedder(feature_dim=F, seed=0).embed_windows(ctx)
    assert emb.shape == (3, F)                      # width == feature_dim, NOT C * F
    assert ls.shape == (3, C, 2)
    # reference: the original single-vector formula
    proj = np.random.default_rng(0).normal(0, 1, size=(C, F)).astype(np.float32)
    ref = np.stack([np.tanh(np.asarray(w, np.float32).mean(0) @ proj) for w in ctx])
    assert np.allclose(emb, ref)


def test_mock_univariate_concat_grows_with_channels():
    """Univariate layout embeds per channel -> concat width is C * feature_dim
    (the property that distinguishes MOMENT/TimesFM-like backbones)."""
    C, F = 5, 8
    ctx = _ctx(n_windows=4, n_channels=C)
    emb, ls = MockEmbedder(feature_dim=F, layout="univariate",
                           channel_aggregation="concat").embed_windows(ctx)
    assert emb.shape == (4, C * F)
    assert ls.shape == (4, C, 2)


def test_mock_univariate_mean_collapses_variate_axis():
    """Mean aggregation collapses the variate axis to a common representation of width
    feature_dim (the RQ-M fairness control)."""
    C, F = 5, 8
    ctx = _ctx(n_windows=4, n_channels=C)
    emb, _ = MockEmbedder(feature_dim=F, layout="univariate",
                          channel_aggregation="mean").embed_windows(ctx)
    assert emb.shape == (4, F)


def test_mock_multivariate_aggregation_modes_coincide():
    """The multivariate joint summary is already channel-collapsed, so concat/mean give
    the same width feature_dim (documented mock simplification)."""
    C, F = 6, 12
    ctx = _ctx(n_channels=C)
    concat, _ = MockEmbedder(feature_dim=F, channel_aggregation="concat").embed_windows(ctx)
    mean, _ = MockEmbedder(feature_dim=F, channel_aggregation="mean").embed_windows(ctx)
    assert concat.shape == mean.shape == (len(ctx), F)
    assert np.allclose(concat, mean)


def test_mock_empty_context_and_describe():
    for layout in ("multivariate", "univariate"):
        m = MockEmbedder(feature_dim=10, layout=layout)
        emb, ls = m.embed_windows([])
        assert emb.shape == (0, 10) and ls.shape == (0, 0, 2)
        d = m.describe()
        assert set(d) >= {"embedder", "feature_dim", "seed", "layout", "channel_aggregation"}
        assert d["layout"] == layout


def test_mock_rejects_bad_params():
    with pytest.raises(ValueError):
        MockEmbedder(layout="bogus")
    with pytest.raises(ValueError):
        MockEmbedder(channel_aggregation="bogus")
