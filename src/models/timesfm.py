"""TimesFM 2.5 embedder (google/timesfm-2.5): a univariate decoder-only TSFM.

Channel-independent like MOMENT (RESEARCH_PLAN §6): each 1-D channel is embedded on
its own, the per-channel patch hidden states are stacked into the canonical
``(n_variates, patches, d_model)`` tensor, and the shared pooling + RQ-M
``channel_aggregation`` apply (``F = n_variates * d_model`` under ``concat``).
TimesFM emits plain per-patch decoder states with no special tokens, so
``n_special_tokens = 0`` and the four pooling names map as in ``models/base.py``:

  * forecast_token -> the last patch (the decoder's next-step summary).  DECISION (uncited): a documented judgment call.
  * last_content   -> the last patch (== forecast_token; no special tokens).
  * mean / flatten -> over all patches (flatten fixed-context only).

TimesFM's per-series normalization is internal; loc/scale is taken from the input
series (``loc_scale_from_contexts``).  DECISION (uncited): a documented judgment call.

The backbone import + call live only in ``_load_pipeline`` / ``_encode_batch`` (the
``# pragma: no cover`` boundary). Documented fallback if per-patch states are not
exposed: the model's penultimate decoder hidden states (RESEARCH_PLAN §11).
"""

from __future__ import annotations

import numpy as np

from .base import TSFMEmbedderBase


class TimesFMEmbedder(TSFMEmbedderBase):
    embedder_name = "TimesFMEmbedder"
    layout = "univariate"
    n_special_tokens = 0

    def _load_pipeline(self):  # pragma: no cover -- GPU-only heavy backbone load
        if self._pipeline is not None:
            return self._pipeline
        import torch
        import timesfm  # lazy heavy import (GPU-only)

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.model_name)
        model = model.to(device).eval() if hasattr(model, "to") else model
        self._torch = torch
        self._device_resolved = device
        self._pipeline = model
        return self._pipeline

    def _encode_batch(self, batch):  # pragma: no cover -- GPU-only backbone call
        model = self._load_pipeline()
        torch = self._torch
        canonical = []
        for w in batch:
            w = np.asarray(w, np.float32)                 # (L, C)
            per_channel = []
            for c in range(w.shape[1]):
                series = torch.as_tensor(w[:, c], dtype=torch.float32,
                                         device=self._device_resolved)[None, :]
                with torch.inference_mode():
                    hidden = model.embed(series)          # (1, patches, d_model)
                per_channel.append(np.asarray(hidden, np.float32).reshape(
                    -1, hidden.shape[-1]))
            canonical.append(np.stack(per_channel, axis=0))  # (C, patches, d_model)
        loc_scale = self.loc_scale_from_contexts(batch)
        return canonical, loc_scale
