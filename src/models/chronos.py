"""Chronos-2 embedder: the anchor frozen TSFM (amazon/chronos-2, arXiv:2510.15821).

Lazy wrapper around the OFFICIAL ``Chronos2Pipeline.embed()`` (chronos-forecasting
2.x) -- we do not reimplement embedding (Task 2.1). Like the other four v2 backbones
it extends ``TSFMEmbedderBase`` (``models/base.py``): the shared base owns batching,
the semantic pooling contract, ``channel_aggregation``, loc/scale shaping and the
throughput bookkeeping, so ONLY the two backbone-touching methods differ and they are
the sole ``# pragma: no cover`` boundary (everything else is CPU-tested through a fake
``_encode_batch``). Chronos-2's ``embed()`` appends 2 trailing special tokens (REG,
forecast), so ``n_special_tokens = 2`` and the four pooling names map onto its layout
exactly as documented in ``embeddings.pool_patches``.

Chronos-2 surfaces its own per-window instance-norm loc/scale, so (unlike the
univariate backbones) it does NOT use the input mean/std fallback: ``_encode_batch``
returns the real loc/scale via ``extract_loc_scale``.

CHANGES.md §40: this replaces the earlier bespoke ``embed_windows`` (which pooled
on-device, §13) with the shared base path (host pooling) so the anchor model meets the
same 100%-coverage / single-pragma bar as the four v2 backbones. Stage A is one-time
and cached, so pooling on host instead of on-device is immaterial.
"""

from __future__ import annotations

import numpy as np

from ..embeddings import extract_loc_scale
from .base import TSFMEmbedderBase


class ChronosEmbedder(TSFMEmbedderBase):
    """Chronos-2 (multivariate-native). Extends the shared plain-patch base; only the
    backbone load/call differ (both pragma'd). ``n_special_tokens = 2`` accounts for
    embed()'s trailing REG + forecast tokens."""

    embedder_name = "ChronosEmbedder"
    layout = "multivariate"
    n_special_tokens = 2

    def _load_pipeline(self):  # pragma: no cover -- GPU-only heavy backbone load
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
            self.model_name, device_map=device, torch_dtype=dtype)
        self._torch = torch
        self._device_resolved = device
        return self._pipeline

    def _encode_batch(self, batch):  # pragma: no cover -- GPU-only backbone call
        """One batch of ``(L_i, C)`` contexts -> (canonical per-window
        ``(n_variates, num_patches+2, d_model)`` arrays, loc/scale ``(b, n_variates,
        2)``). Feeds embed() its native variable-length list input (short contexts are
        left-pad-MASKED internally); the shared base then pools every window with
        ``n_special_tokens = 2``."""
        pipeline = self._load_pipeline()
        torch = self._torch
        # embed() wants each item as (n_variates, history_length).
        inp = [np.transpose(w, (1, 0)) for w in batch]
        n_variates = inp[0].shape[0]
        with torch.inference_mode():
            embeddings, loc_scale = pipeline.embed(
                inp, batch_size=len(inp), context_length=self.context_length)
        canonical = [np.asarray(e.detach().to("cpu").float().numpy(), np.float32)
                     for e in embeddings]
        return canonical, extract_loc_scale(loc_scale, len(inp), n_variates)
