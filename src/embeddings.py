"""Frozen-TSFM embedding wrapper + disk cache.

Wraps the OFFICIAL ``Chronos2Pipeline.embed()`` (chronos-forecasting 2.x) -- we do
not reimplement embedding (Task 2.1). ``embed()`` returns, per window, a tensor of
shape ``(n_variates, num_patches + 2, d_model)``; ``pool_window_embedding`` reduces
that to one feature vector using the configured strategy (last_patch | mean |
flatten -- Task 2.2, RESEARCH_PLAN sec.1).

The backbone is frozen, so every window is embedded exactly once and the result is
cached to disk keyed by ``config.embedding_cache_key()`` (window size, pooling,
model name, sensor set...). Stage A builds the cache; Stage B/sweep only ever
loads it. Any code path that re-embeds during a sweep is a bug (Task 3).

``model_name`` is a plain string so MOMENT/TimesFM/TTM slot in later behind the
same ``Embedder`` protocol (Task 2.6) -- only ``_load_pipeline``/``embed_windows``
would gain a branch.

The embedder is injectable: ``build_embedding_cache(config, embedder=...)`` accepts
any object exposing ``embed_windows(windows) -> (N, D)`` and ``describe() -> dict``,
so CPU-only smoke tests pass a mock and never import chronos or hit a GPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

from .config import Config


class Embedder(Protocol):
    def embed_windows(self, windows: np.ndarray) -> np.ndarray: ...
    def describe(self) -> dict: ...


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------
def pool_window_embedding(emb: np.ndarray, strategy: str) -> np.ndarray:
    """Reduce one window embedding ``(n_variates, num_patches+2, d_model)`` to a 1D
    feature vector. The variate axis is always flattened into the output (Chronos-2
    mixes variates via group attention, so per-variate embeddings carry signal)."""
    if emb.ndim != 3:
        raise ValueError(f"expected (n_variates, patches, d_model), got {emb.shape}")
    if strategy == "last_patch":
        # DECISION (uncited): take the final position along the patch axis as the
        # window summary. embed() appends 2 register/boundary tokens after the
        # content patches, so this is the last such token.
        pooled = emb[:, -1, :]           # (n_variates, d_model)
    elif strategy == "mean":
        pooled = emb.mean(axis=1)        # (n_variates, d_model)
    elif strategy == "flatten":
        return emb.reshape(-1).astype(np.float32)  # the PHM paper flattens
    else:
        raise ValueError(f"unknown pooling strategy: {strategy!r}")
    return pooled.reshape(-1).astype(np.float32)


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
        self.context_length = config.context_length
        self._device = device
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        import torch  # local import: never required by CPU smoke tests
        from chronos import Chronos2Pipeline  # official embed() lives here (Task 2.1)

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}.get(self.dtype, torch.bfloat16)
        # On CPU, bf16/fp16 are slow/unsupported for some ops; fall back to fp32.
        if device == "cpu":
            dtype = torch.float32
        self._pipeline = Chronos2Pipeline.from_pretrained(
            self.model_name, device_map=device, torch_dtype=dtype
        )
        self._torch = torch
        return self._pipeline

    def embed_windows(self, windows: np.ndarray) -> np.ndarray:
        """(N, window_size, n_channels) -> (N, feature_dim) pooled embeddings."""
        pipeline = self._load_pipeline()
        torch = self._torch
        # embed() wants (batch, n_variates, history_length); windows are
        # (N, window_size, n_channels) => transpose the last two axes.
        n = windows.shape[0]
        feats: list[np.ndarray] = []
        for start in range(0, n, self.batch_size):
            chunk = windows[start : start + self.batch_size]
            inp = np.transpose(chunk, (0, 2, 1))  # (b, n_variates, window_size)
            with torch.inference_mode():
                embeddings, _loc_scale = pipeline.embed(
                    inp, batch_size=self.batch_size,
                    context_length=self.context_length,
                )
            for emb in embeddings:  # each (n_variates, num_patches+2, d_model)
                arr = emb.detach().to(torch.float32).cpu().numpy()
                feats.append(pool_window_embedding(arr, self.pooling))
        return np.asarray(feats, dtype=np.float32)

    def describe(self) -> dict:
        return {
            "embedder": "ChronosEmbedder",
            "model_name": self.model_name,
            "pooling": self.pooling,
            "dtype": self.dtype,
            "context_length": self.context_length,
        }


# ---------------------------------------------------------------------------
# Disk cache (idempotent)
# ---------------------------------------------------------------------------
def build_embedding_cache(
    config: Config,
    embedder: Optional[Embedder] = None,
    overwrite: bool = False,
) -> Path:
    """Stage A: window all FD001 train units + test last-cycle windows, embed once,
    and cache to ``config.cache_path()``. Idempotent: if the cache key already
    exists and ``overwrite`` is False, does nothing and returns the path.

    Cache contents (npz): train/test raw windows, labels, unit_ids, and pooled
    embeddings, plus a JSON sidecar with the resolved key fields + embedder info.
    Baselines consume the raw windows from this same cache (no re-embedding).
    """
    from . import data as data_mod  # local import keeps config/embeddings light

    cache_path = config.cache_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not overwrite:
        return cache_path

    df_train, df_test, rul_truth = data_mod.load_cmapss(config)
    df_train = data_mod.add_train_rul(df_train, config)
    df_test = data_mod.add_test_rul(df_test, rul_truth, config)

    tr_w, tr_y, tr_u = data_mod.make_windows(
        df_train, config.sensor_columns, config.window_size, target_col="clipped_rul"
    )
    te_w, te_y, te_u = data_mod.make_test_last_windows(
        df_test, config.sensor_columns, config.window_size,
        target_col="clipped_rul", pad_short=config.pad_short_test_units,
    )

    if embedder is None:
        embedder = ChronosEmbedder(config)
    tr_emb = embedder.embed_windows(tr_w)
    te_emb = embedder.embed_windows(te_w)

    np.savez_compressed(
        cache_path,
        train_windows=tr_w, train_labels=tr_y, train_units=tr_u, train_emb=tr_emb,
        test_windows=te_w, test_labels=te_y, test_units=te_u, test_emb=te_emb,
    )
    sidecar = cache_path.with_suffix(".json")
    sidecar.write_text(json.dumps(
        {"embedding_key_fields": config._embedding_key_fields(),
         "embedder": embedder.describe(),
         "feature_dim": int(tr_emb.shape[1]) if tr_emb.size else 0,
         "n_train_windows": int(tr_w.shape[0]),
         "n_test_windows": int(te_w.shape[0])},
        indent=2, sort_keys=True,
    ))
    return cache_path


def load_embedding_cache(config: Config) -> dict:
    """Load the Stage A cache as a dict of numpy arrays. Raises if missing."""
    cache_path = config.cache_path()
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Embedding cache {cache_path} not found. Run Stage A "
            f"(build_embedding_cache) first."
        )
    with np.load(cache_path) as npz:
        return {k: npz[k] for k in npz.files}
