"""Chronos-2 embedder: the anchor frozen TSFM (amazon/chronos-2, arXiv:2510.15821).

Lazy wrapper around the OFFICIAL ``Chronos2Pipeline.embed()`` (chronos-forecasting
2.x) -- we do not reimplement embedding (Task 2.1). The generic embedding cache,
pooling, and loc/scale extraction live in ``src/embeddings.py``; this module is just
the concrete backbone, registered under its ``model_name`` in ``src/models``. Adding
another TSFM (MOMENT/TimesFM/TTM/Moirai) is one new module here exposing the same
``embed_windows`` / ``describe`` interface + one registry entry (RESEARCH_PLAN sec.4).
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from ..config import Config
from ..embeddings import Contexts, _pool_one_torch, extract_loc_scale


class ChronosEmbedder:
    """Lazy wrapper around ``Chronos2Pipeline`` from chronos-forecasting 2.x."""

    def __init__(self, config: Config, device: Optional[str] = None):
        self.config = config
        self.model_name = config.model_name
        self.pooling = config.pooling
        self.channel_aggregation = config.channel_aggregation  # RQ-M knob (§34)
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
            # Chronos-2's embed() appends 2 special tokens (REG, forecast), so pool
            # with n_special_tokens=2; channel_aggregation is the RQ-M knob (§34).
            if flatten:
                for emb in embeddings:
                    feats.append(_pool_one_torch(emb, self.pooling, self.channel_aggregation)
                                 .to(torch.float32).cpu().numpy())
            else:
                pooled = torch.stack([_pool_one_torch(e, self.pooling, self.channel_aggregation)
                                      for e in embeddings])
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
            "channel_aggregation": self.channel_aggregation,
            "dtype": self.dtype,
            "tsfm_context_length": self.context_length,
        }
