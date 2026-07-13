"""Frozen-TSFM embedding wrapper + disk cache.

Wraps the OFFICIAL ``Chronos2Pipeline.embed()`` (chronos-forecasting 2.x) -- we do
not reimplement embedding (Task 2.1). ``embed()`` returns, per window, a tensor of
shape ``(n_variates, num_patches + 2, d_model)`` AND per-window loc/scale from its
internal instance normalization; ``pool_window_embedding`` reduces the embedding to
one feature vector.

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
import time
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
# ---------------------------------------------------------------------------
def pool_window_embedding(emb: np.ndarray, strategy: str) -> np.ndarray:
    """Reduce one window embedding ``(n_variates, num_patches+2, d_model)`` to a 1D
    feature vector. The variate axis is always flattened into the output (Chronos-2
    mixes variates via group attention, so per-variate embeddings carry signal).

    Special tokens (REG at -2, forecast at -1) are EXCLUDED from content-based
    poolings (``mean``, ``flatten``); ``forecast_token``/``last_content`` index them
    explicitly (Task 1.3)."""
    if emb.ndim != 3:
        raise ValueError(f"expected (n_variates, patches, d_model), got {emb.shape}")
    if emb.shape[1] < 3:
        raise ValueError(
            f"need >=3 patch positions (>=1 content + REG + forecast), got {emb.shape[1]}"
        )
    if strategy == "forecast_token":
        pooled = emb[:, -1, :]            # masked output/forecast patch (CLS-like)
    elif strategy == "last_content":
        pooled = emb[:, -3, :]            # last real content patch (before REG, forecast)
    elif strategy == "mean":
        pooled = emb[:, :-2, :].mean(axis=1)   # content patches only
    elif strategy == "flatten":
        return emb[:, :-2, :].reshape(-1).astype(np.float32)  # content patches, all
    else:
        raise ValueError(f"unknown pooling strategy: {strategy!r}")
    return pooled.reshape(-1).astype(np.float32)


def _pool_one_torch(emb_t, strategy: str):
    """On-device pooling of one window tensor ``(n_variates, P, d_model)`` -> 1D
    tensor. Same semantics as ``pool_window_embedding`` but stays on the GPU so only
    the small pooled vector is transferred to host (Task 2 vectorized pooling)."""
    if emb_t.shape[1] < 3:
        raise ValueError(
            f"need >=3 patch positions (>=1 content + REG + forecast), got {emb_t.shape[1]}"
        )
    if strategy == "forecast_token":
        return emb_t[:, -1, :].reshape(-1)
    if strategy == "last_content":
        return emb_t[:, -3, :].reshape(-1)
    if strategy == "mean":
        return emb_t[:, :-2, :].mean(dim=1).reshape(-1)
    if strategy == "flatten":
        return emb_t[:, :-2, :].reshape(-1)
    raise ValueError(f"unknown pooling strategy: {strategy!r}")


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
# Chronos-2 embedder
# ---------------------------------------------------------------------------
class ChronosEmbedder:
    """Lazy wrapper around ``Chronos2Pipeline`` from chronos-forecasting 2.x."""

    def __init__(self, config: Config, device: Optional[str] = None):
        self.config = config
        self.model_name = config.model_name
        self.pooling = config.pooling
        self.batch_size = config.embed_batch_size
        self.dtype = config.embed_dtype
        self.context_length = config.effective_tsfm_context()
        self._device = device
        self._pipeline = None
        self.last_throughput = None  # windows/s of the most recent embed_windows call

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        import torch  # local import: never required by CPU smoke tests
        from chronos import Chronos2Pipeline  # official embed() lives here (Task 2.1)

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}.get(self.dtype, torch.bfloat16)
        if device == "cpu":
            dtype = torch.float32  # bf16/fp16 slow/unsupported on CPU for some ops
        # Embedding is inference-only and cached once, so cuDNN benchmark autotuning
        # is safe here (Task 2) -- it never touches the seeded head-training path.
        if device == "cuda":
            torch.backends.cudnn.benchmark = True
        self._pipeline = Chronos2Pipeline.from_pretrained(
            self.model_name, device_map=device, torch_dtype=dtype
        )
        self._torch = torch
        self._device_resolved = device
        return self._pipeline

    @staticmethod
    def _as_context_list(contexts: Contexts) -> list[np.ndarray]:
        if isinstance(contexts, np.ndarray) and contexts.ndim == 3:
            return [contexts[i] for i in range(contexts.shape[0])]
        return list(contexts)

    def embed_windows(self, contexts: Contexts) -> tuple[np.ndarray, np.ndarray]:
        """(list of (L_i, C) or (N, W, C)) -> (pooled embeddings (N, F) float32,
        loc/scale (N, n_variates, 2) float32).

        Feeds embed() its native variable-length list input; short contexts are
        left-pad-MASKED internally. Pooling is done on-device per batch and only the
        pooled vectors are transferred to host (Task 2)."""
        pipeline = self._load_pipeline()
        torch = self._torch
        ctx_list = self._as_context_list(contexts)
        n = len(ctx_list)
        if n == 0:
            return (np.empty((0, 0), np.float32), np.empty((0, 0, 2), np.float32))

        flatten = self.pooling == "flatten"
        feats: list[np.ndarray] = []
        ls_batches: list[np.ndarray] = []
        t0 = time.perf_counter()
        for start in range(0, n, self.batch_size):
            chunk = ctx_list[start : start + self.batch_size]
            # embed() wants each item as (n_variates, history_length).
            inp = [np.transpose(w, (1, 0)) for w in chunk]
            n_variates = inp[0].shape[0]
            with torch.inference_mode():
                embeddings, loc_scale = pipeline.embed(
                    inp, batch_size=len(inp), context_length=self.context_length,
                )
            if flatten:
                for emb in embeddings:
                    feats.append(_pool_one_torch(emb, self.pooling).to(torch.float32).cpu().numpy())
            else:
                pooled = torch.stack([_pool_one_torch(e, self.pooling) for e in embeddings])
                feats.append(pooled.to(torch.float32).cpu().numpy())  # one transfer / batch
            ls_batches.append(extract_loc_scale(loc_scale, len(inp), n_variates))
        dt = time.perf_counter() - t0
        self.last_throughput = n / dt if dt > 0 else float("inf")

        # flatten: feats is a list of N per-window 1D vectors (must be equal length
        # => fixed context); otherwise a list of per-batch (b, F) arrays.
        emb_arr = np.stack(feats, axis=0) if flatten else np.concatenate(feats, axis=0)
        ls_arr = np.concatenate(ls_batches, axis=0)
        return emb_arr.astype(np.float32), ls_arr.astype(np.float32)

    def describe(self) -> dict:
        return {
            "embedder": "ChronosEmbedder",
            "model_name": self.model_name,
            "pooling": self.pooling,
            "dtype": self.dtype,
            "tsfm_context_length": self.context_length,
        }


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

    df_train, df_test, rul_truth = data_mod.load_cmapss(config)
    df_train = data_mod.add_train_rul(df_train, config)
    df_test = data_mod.add_test_rul(df_test, rul_truth, config)

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
        embedder = ChronosEmbedder(config)
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
