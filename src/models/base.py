"""Shared base for the four v2 TSFM embedders (Moirai-2, MOMENT, TimesFM, TTM).

The generic embedding infra -- pooling, channel aggregation, loc/scale shaping,
the fp16 disk cache -- already lives in ``src/embeddings.py`` and is model-agnostic
(IMPLEMENTATION_PLAN §4.1). This base captures the ONE thing those four backbones
share beyond Chronos-2: they expose plain per-patch encoder hidden states with **no
Chronos-style special tokens** (``n_special_tokens = 0``), and each returns, per
window, a canonical ``(n_variates, patches, d_model)`` tensor that pools EXACTLY like
Chronos-2's once the special-token count is accounted for. So the semantic pooling
contract (``forecast_token``/``last_content``/``mean``/``flatten``) and the RQ-M
``channel_aggregation`` knob apply uniformly -- the only per-backbone code is how the
raw backbone output is mapped into that canonical tensor (``_encode_batch``) and how
the pipeline is loaded (``_load_pipeline``).

Those two backbone-touching methods are the sanctioned ``# pragma: no cover``
boundary (CHANGES.md §32/§34): they lazily import a heavy GPU library and call it, so
CPU tests never reach them. EVERYTHING ELSE here -- batching, pooling, the loc/scale
fallback, throughput bookkeeping, ``describe`` -- is covered through a fake subclass
whose ``_encode_batch`` returns synthetic canonical tensors (``tests/test_models.py``),
so no test imports a backbone. Each concrete embedder is a thin subclass that sets
``embedder_name`` / ``layout`` and implements the two pragma'd methods.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from ..config import Config
from ..embeddings import Contexts, pool_window_embedding


class TSFMEmbedderBase:
    """Base embedder for plain-patch (no special-token) frozen TSFMs.

    Concrete subclasses override the two backbone methods and the identity class
    attributes; the pooling/aggregation/assembly path is shared and CPU-tested.
    """

    embedder_name: str = "TSFMEmbedderBase"
    # Plain per-patch encoders append no REG/forecast tokens, so every position is a
    # content patch (contrast Chronos-2's 2 trailing special tokens). See the pooling
    # contract in ``embeddings.pool_patches``.
    n_special_tokens: int = 0
    # Descriptive only ({"univariate","multivariate"}): univariate backbones embed
    # each channel independently then stack; multivariate-native ones embed jointly.
    # Both yield the canonical (n_variates, patches, d_model) tensor, so pooling is
    # identical -- the label just records how a practitioner uses the model (RQ-M).
    layout: str = "univariate"

    def __init__(self, config: Config, device: Optional[str] = None):
        self.config = config
        self.model_name = config.model_name
        self.pooling = config.pooling
        self.channel_aggregation = config.channel_aggregation
        self.batch_size = config.embed_batch_size
        self.dtype = config.embed_dtype
        self.context_length = config.effective_tsfm_context()
        self._device = device
        self._pipeline = None
        self.last_throughput = None  # windows/s of the most recent embed_windows call

    # ---- backbone seam (the ONLY sanctioned pragma boundary) ---------------
    def _load_pipeline(self):  # pragma: no cover -- GPU-only heavy backbone load
        raise NotImplementedError("concrete embedder must implement _load_pipeline")

    def _encode_batch(self, batch):  # pragma: no cover -- GPU-only backbone call
        """``batch``: list of ``(L_i, n_channels)`` context arrays. Returns
        ``(canonical, loc_scale)`` where ``canonical`` is a length-``len(batch)`` list
        of ``(n_variates, patches, d_model)`` float arrays and ``loc_scale`` is
        ``(len(batch), n_variates, 2)``. Concrete embedders map the backbone's raw
        output into this shape (and use ``loc_scale_from_contexts`` when the backbone
        does not expose instance-norm statistics)."""
        raise NotImplementedError("concrete embedder must implement _encode_batch")

    # ---- shared, CPU-covered machinery -------------------------------------
    @staticmethod
    def _as_context_list(contexts: Contexts) -> list[np.ndarray]:
        if isinstance(contexts, np.ndarray) and contexts.ndim == 3:
            return [contexts[i] for i in range(contexts.shape[0])]
        return list(contexts)

    @staticmethod
    def loc_scale_from_contexts(batch) -> np.ndarray:
        """Per-channel instance-norm loc/scale computed from the INPUT series, for
        backbones that do not expose their own (IMPLEMENTATION_PLAN §4.1 fallback):
        ``(len(batch), n_channels, 2)`` = per-window [mean, std] of each channel.
        DECISION (uncited): input mean/std is the natural RevIN-equivalent when the
        backbone hides its normalization."""
        out = []
        for w in batch:
            w = np.asarray(w, np.float32)
            out.append(np.stack([w.mean(axis=0), w.std(axis=0)], axis=-1))  # (C, 2)
        return np.asarray(out, np.float32)

    def _pool_canonical(self, canonical: np.ndarray) -> np.ndarray:
        """Pool one canonical ``(n_variates, patches, d_model)`` window into the head
        feature vector, honoring ``pooling`` + ``channel_aggregation`` at this
        backbone's ``n_special_tokens`` layout."""
        return pool_window_embedding(
            np.asarray(canonical, np.float32), self.pooling,
            self.channel_aggregation, self.n_special_tokens)

    def embed_windows(self, contexts: Contexts) -> tuple[np.ndarray, np.ndarray]:
        """(list of ``(L_i, C)`` or ``(N, W, C)``) -> (pooled embeddings ``(N, F)``
        float32, loc/scale ``(N, n_variates, 2)`` float32). Batches, encodes each
        batch through the backbone, and pools every window with the shared contract.
        """
        ctx_list = self._as_context_list(contexts)
        n = len(ctx_list)
        if n == 0:
            return (np.empty((0, 0), np.float32), np.empty((0, 0, 2), np.float32))

        feats: list[np.ndarray] = []
        ls_batches: list[np.ndarray] = []
        t0 = time.perf_counter()
        for start in range(0, n, self.batch_size):
            batch = ctx_list[start : start + self.batch_size]
            canonical, loc_scale = self._encode_batch(batch)
            for canon in canonical:
                feats.append(self._pool_canonical(canon))
            ls_batches.append(np.asarray(loc_scale, np.float32))
        dt = time.perf_counter() - t0
        self.last_throughput = n / dt if dt > 0 else float("inf")

        # np.stack requires equal feature length -> holds for every pooling except
        # variable-length `flatten`, which is fixed-context-only (raises loudly, as
        # documented) rather than silently ragged.
        emb_arr = np.stack(feats, axis=0)
        ls_arr = np.concatenate(ls_batches, axis=0)
        return emb_arr.astype(np.float32), ls_arr.astype(np.float32)

    def describe(self) -> dict:
        return {
            "embedder": self.embedder_name,
            "model_name": self.model_name,
            "pooling": self.pooling,
            "channel_aggregation": self.channel_aggregation,
            "layout": self.layout,
            "n_special_tokens": self.n_special_tokens,
            "dtype": self.dtype,
            "tsfm_context_length": self.context_length,
        }
