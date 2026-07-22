"""Frozen-TSFM embedding infrastructure: pooling, loc/scale handling + disk cache.

Model-agnostic plumbing shared by every TSFM. The concrete backbones live in
``src/models/`` (one module per model, e.g. ``models/chronos.py``) and are selected
by ``config.model_name`` via ``models.make_embedder``; each returns, per window, a
tensor of shape ``(n_variates, num_patches + 2, d_model)`` AND per-window loc/scale
from its internal instance normalization. ``pool_window_embedding`` reduces the
embedding to one feature vector. Chronos-2 uses the OFFICIAL
``Chronos2Pipeline.embed()`` -- we do not reimplement embedding (Task 2.1).

Token layout (Task 1.3): the last two positions are special tokens embed() appends
-- index ``-1`` is the masked output/forecast patch (a CLS-like window summary),
index ``-2`` is the REG token. Content patches are ``emb[:, :-2, :]``. Poolings:
  * ``forecast_token`` -> ``emb[:, -1, :]``   (masked output patch; default)
  * ``last_content``   -> ``emb[:, -3, :]``   (last real content patch)
  * ``mean``           -> ``emb[:, :-2, :].mean(1)``  (content patches only)
  * ``flatten``        -> ``emb[:, :-2, :].reshape(-1)`` (fixed-length contexts only)

Why loc/scale is cached (Task 1.1): Chronos-2 normalizes each window internally and
the per-window loc/scale is the ONLY carrier of the slow degradation-level signal
that a normalized 30-cycle window otherwise erases. It is cached per window as
``(n_windows, n_variates, 2)`` and optionally fused into the head input.

The backbone is frozen, so every window is embedded exactly once and cached, keyed
by ``config.embedding_cache_key()`` (window size, pooling, model, TSFM context
length, cache schema version). Stage A builds the cache; Stage B/sweep only loads
it. Any code path that re-embeds during a sweep is a bug (Task 3).

The embedder is injectable: ``build_embedding_cache(config, embedder=...)`` accepts
any object exposing ``embed_windows(contexts) -> (embeddings, loc_scale)`` and
``describe() -> dict``, so CPU-only smoke tests pass a mock and never import chronos
or hit a GPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Protocol, Sequence, Union

import numpy as np

from .config import Config

# A batch of TSFM contexts: either a fixed (N, W, C) array or a list of variable-
# length (L_i, C) arrays (embed()'s native variable-length input, Task 1.2).
Contexts = Union[np.ndarray, Sequence[np.ndarray]]


class Embedder(Protocol):
    def embed_windows(self, contexts: Contexts) -> tuple[np.ndarray, np.ndarray]: ...
    def describe(self) -> dict: ...


# ---------------------------------------------------------------------------
# Pooling (numpy reference; kept for tests + CPU paths)
#
# Pooling is TWO stages so it is common across every backbone (IMPLEMENTATION_PLAN
# §4.1, the semantic-not-index-based pooling contract):
#   1. pool the PATCH axis of one window ``(n_variates, patches, d_model)`` down to a
#      per-variate vector ``(n_variates, K)``, honoring the four pooling NAMES, and
#   2. collapse the variate axis by ``channel_aggregation`` -> the 1-D head feature.
#
# ``n_special_tokens`` is the layout knob that makes the same four names mean the
# same thing across layouts: Chronos-2 appends 2 trailing special tokens (REG at -2,
# forecast at -1), so ``n_special_tokens=2`` and content patches are ``[:-2]``; the
# plain-patch backbones (MOMENT/TimesFM/Moirai/TTM as wrapped here) append none, so
# ``n_special_tokens=0`` and every position is a content patch. The four names map:
#   * forecast_token -> the last position (the forecast special token for Chronos-2;
#     the last patch -- the closest "predict-next" summary -- when there are none).
#   * last_content   -> the last CONTENT patch (== forecast_token when n_special=0).
#   * mean           -> mean over content patches.
#   * flatten        -> all content patches concatenated (fixed-context only).
# The defaults (``channel_aggregation="concat"``, ``n_special_tokens=2``) reproduce
# the original Chronos-2 pooling byte-for-byte, so the recorded FD001 cache is
# unchanged.
# ---------------------------------------------------------------------------
def _content_count(n_patches: int, n_special_tokens: int) -> int:
    """Number of content patches, failing loud when there are none."""
    n_content = n_patches - n_special_tokens
    if n_content < 1:
        raise ValueError(
            f"need >=1 content patch after excluding {n_special_tokens} special "
            f"token(s), got {n_patches} patch position(s)")
    return n_content


def pool_patches(emb: np.ndarray, strategy: str, n_special_tokens: int = 2) -> np.ndarray:
    """Pool the patch axis of one window ``(n_variates, patches, d_model)`` down to a
    per-variate ``(n_variates, K)`` array (K = d_model, or content_patches*d_model
    for ``flatten``). Stage 1 of the pooling contract (see module header)."""
    if emb.ndim != 3:
        raise ValueError(f"expected (n_variates, patches, d_model), got {emb.shape}")
    n_content = _content_count(emb.shape[1], n_special_tokens)
    if strategy == "forecast_token":
        return emb[:, -1, :]                       # last position (special or last patch)
    if strategy == "last_content":
        return emb[:, n_content - 1, :]            # last real content patch
    content = emb[:, :n_content, :]
    if strategy == "mean":
        return content.mean(axis=1)               # content patches only
    if strategy == "flatten":
        return content.reshape(emb.shape[0], -1)  # (n_variates, content*d_model)
    raise ValueError(f"unknown pooling strategy: {strategy!r}")


def aggregate_variates(per_variate: np.ndarray, channel_aggregation: str) -> np.ndarray:
    """Collapse the variate axis of ``(n_variates, K)`` into a 1-D feature (stage 2):
    ``concat`` flattens it (F = n_variates*K, the practitioner default), ``mean``
    averages the variates (F = K, the RQ-M common-representation control)."""
    if channel_aggregation == "concat":
        return per_variate.reshape(-1)
    if channel_aggregation == "mean":
        return per_variate.mean(axis=0)
    raise ValueError(f"unknown channel_aggregation: {channel_aggregation!r}")


def pool_window_embedding(emb: np.ndarray, strategy: str,
                          channel_aggregation: str = "concat",
                          n_special_tokens: int = 2) -> np.ndarray:
    """Reduce one window embedding ``(n_variates, patches, d_model)`` to a 1D feature
    vector: pool the patch axis (``strategy``) then collapse the variate axis
    (``channel_aggregation``). See the module header for the layout contract."""
    per_variate = pool_patches(emb, strategy, n_special_tokens)
    return aggregate_variates(per_variate, channel_aggregation).astype(np.float32)


def _pool_one_torch(emb_t, strategy: str, channel_aggregation: str = "concat",
                    n_special_tokens: int = 2):
    """On-device twin of ``pool_window_embedding`` for one window tensor
    ``(n_variates, P, d_model)`` -> 1D tensor, so only the small pooled vector is
    transferred to host (Task 2 vectorized pooling)."""
    n_content = _content_count(emb_t.shape[1], n_special_tokens)
    if strategy == "forecast_token":
        per_variate = emb_t[:, -1, :]
    elif strategy == "last_content":
        per_variate = emb_t[:, n_content - 1, :]
    elif strategy == "mean":
        per_variate = emb_t[:, :n_content, :].mean(dim=1)
    elif strategy == "flatten":
        per_variate = emb_t[:, :n_content, :].reshape(emb_t.shape[0], -1)
    else:
        raise ValueError(f"unknown pooling strategy: {strategy!r}")
    if channel_aggregation == "concat":
        return per_variate.reshape(-1)
    if channel_aggregation == "mean":
        return per_variate.mean(dim=0)
    raise ValueError(f"unknown channel_aggregation: {channel_aggregation!r}")


# ---------------------------------------------------------------------------
# loc/scale extraction (defensive: normalize embed()'s return to (batch, V, 2))
# ---------------------------------------------------------------------------
def _to_numpy(x):
    try:
        return x.detach().to("cpu").float().numpy()
    except AttributeError:
        return np.asarray(x, dtype=np.float32)


def extract_loc_scale(loc_scale, batch: int, n_variates: int) -> np.ndarray:
    """Normalize embed()'s per-window loc/scale return into ``(batch, n_variates, 2)``.

    Chronos-2 (2.x) returns instance-norm loc/scale alongside the embeddings. Its
    exact container varies by version, so this accepts: a single stacked tensor, a
    ``(loc, scale)`` pair, or a per-item sequence, and validates the final shape so a
    mismatch fails loudly on the first Stage A run rather than caching garbage.
    """
    is_seq = isinstance(loc_scale, (tuple, list))
    # (loc, scale) pair -- prefer this reading unless the batch itself is 2 (then a
    # length-2 container is ambiguous and handled as a per-item sequence below).
    if is_seq and len(loc_scale) == 2 and batch != 2:
        loc, scale = _to_numpy(loc_scale[0]), _to_numpy(loc_scale[1])
        arr = np.stack([loc.reshape(batch, n_variates),
                        scale.reshape(batch, n_variates)], axis=-1)
    # Per-item sequence: one (n_variates, 2) / (n_variates,)x2 entry per window.
    elif is_seq and len(loc_scale) == batch:
        arr = np.stack([_to_numpy(x).reshape(n_variates, 2) for x in loc_scale], axis=0)
    else:
        arr = _to_numpy(loc_scale)
        if arr.shape[-1] == 2:                       # (..., 2)
            arr = arr.reshape(batch, n_variates, 2)
        elif arr.shape[0] == 2:                      # (2, batch, n_variates)
            arr = np.moveaxis(arr, 0, -1).reshape(batch, n_variates, 2)
        else:
            raise ValueError(
                f"cannot interpret embed() loc/scale of shape {arr.shape} as "
                f"(batch={batch}, n_variates={n_variates}, 2); adjust "
                f"extract_loc_scale for this chronos version."
            )
    if arr.shape != (batch, n_variates, 2):
        raise ValueError(f"loc/scale normalized to {arr.shape}, expected {(batch, n_variates, 2)}")
    return arr.astype(np.float32)


# ---------------------------------------------------------------------------
# Concrete embedders live in src/models/ (one module per TSFM), selected by
# ``config.model_name`` through ``models.make_embedder``. ChronosEmbedder is the
# anchor backbone; see src/models/chronos.py.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Disk cache (idempotent)
# ---------------------------------------------------------------------------
def build_embedding_cache(
    config: Config,
    embedder: Optional[Embedder] = None,
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """Stage A: build FIXED baseline windows + variable-length TSFM contexts, embed
    the contexts once, and cache to ``config.cache_path()``. Idempotent.

    The fixed windows (float32) feed the baselines and the raw-fusion last-cycle
    sensors; the variable-length contexts feed embed() and are NOT cached (large,
    transient). Cached embeddings are stored ``embedding_storage_dtype`` (default
    float16) to cut Drive I/O; loc/scale stays float32 (Task 2). A JSON sidecar
    records the resolved key fields + embedder info + measured throughput.
    """
    from . import data as data_mod  # local import keeps config/embeddings light

    cache_path = config.cache_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not overwrite:
        return cache_path

    # ONE loading path for every stage: labels + (resolved) condition-wise
    # normalization applied identically everywhere (CHANGES.md §21).
    df_train, df_test = data_mod.load_prepared(config)

    ws, tsfm_ctx = config.window_size, config.effective_tsfm_context()
    cols = config.sensor_columns

    # Fixed windows (baselines + raw-fusion last cycle). Train label = clipped RUL;
    # test label = UNCLIPPED actual RUL (evaluate reports both protocols, Task 1.4).
    tr_w, tr_y, tr_u = data_mod.make_windows(df_train, cols, ws, target_col="clipped_rul")
    te_w, te_y, te_u = data_mod.make_test_last_windows(
        df_test, cols, ws, target_col="actual_rul", pad_short=config.pad_short_test_units,
    )
    # Variable-length TSFM contexts, aligned 1:1 to the fixed windows (Task 1.2).
    tr_ctx, tr_y2, tr_u2 = data_mod.make_windows_varlen(
        df_train, cols, ws, tsfm_ctx, target_col="clipped_rul")
    te_ctx, te_y2, te_u2 = data_mod.make_test_last_contexts(
        df_test, cols, tsfm_ctx, target_col="actual_rul")
    # Alignment guarantee (also asserted in tests): same labels + units, same order.
    assert np.array_equal(tr_u, tr_u2) and np.allclose(tr_y, tr_y2), "train varlen misaligned"
    assert np.array_equal(te_u, te_u2) and np.allclose(te_y, te_y2), "test varlen misaligned"
    assert len(tr_ctx) == len(tr_w) and len(te_ctx) == len(te_w)

    if embedder is None:
        from .models import make_embedder  # local import: avoids models<->embeddings cycle
        embedder = make_embedder(config)
    tr_emb, tr_ls = embedder.embed_windows(tr_ctx)
    if verbose and getattr(embedder, "last_throughput", None):
        print(f"[Stage A] train embed throughput: {embedder.last_throughput:.1f} windows/s")
    te_emb, te_ls = embedder.embed_windows(te_ctx)
    if verbose and getattr(embedder, "last_throughput", None):
        print(f"[Stage A] test  embed throughput: {embedder.last_throughput:.1f} windows/s")

    store_dtype = np.dtype(config.embedding_storage_dtype)
    arrays = dict(
        train_windows=tr_w.astype(np.float32),
        train_labels=tr_y.astype(np.float32),
        train_units=tr_u,
        train_emb=tr_emb.astype(store_dtype),
        train_locscale=tr_ls.astype(np.float32),
        test_windows=te_w.astype(np.float32),
        test_labels=te_y.astype(np.float32),
        test_units=te_u,
        test_emb=te_emb.astype(store_dtype),
        test_locscale=te_ls.astype(np.float32),
    )
    saver = np.savez_compressed if config.cache_compressed else np.savez
    saver(cache_path, **arrays)

    sidecar = cache_path.with_suffix(".json")
    sidecar.write_text(json.dumps(
        {"embedding_key_fields": config._embedding_key_fields(),
         "embedder": embedder.describe(),
         "embedding_storage_dtype": str(store_dtype),
         "feature_dim": int(tr_emb.shape[1]) if tr_emb.size else 0,
         "n_variates": int(tr_ls.shape[1]) if tr_ls.size else 0,
         "n_train_windows": int(tr_w.shape[0]),
         "n_test_windows": int(te_w.shape[0]),
         "train_throughput_windows_per_s": getattr(embedder, "last_throughput", None)},
        indent=2, sort_keys=True,
    ))
    return cache_path


def load_embedding_cache(config: Config) -> dict:
    """Load the Stage A cache as a dict of numpy arrays; embeddings upcast to
    float32 for training. Raises if missing."""
    cache_path = config.cache_path()
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Embedding cache {cache_path} not found. Run Stage A "
            f"(build_embedding_cache) first."
        )
    with np.load(cache_path) as npz:
        out = {k: npz[k] for k in npz.files}
    for k in ("train_emb", "test_emb"):
        if k in out:
            out[k] = out[k].astype(np.float32)  # upcast fp16 storage -> fp32 compute
    return out
