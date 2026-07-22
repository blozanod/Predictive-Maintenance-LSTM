"""MOMENT embedder (AutonLab/MOMENT-1-large): a univariate masked-reconstruction TSFM.

MOMENT is channel-independent (RESEARCH_PLAN §6): it embeds each 1-D series on its
own, so we loop channels, embed each, stack the per-channel patch embeddings into the
canonical ``(n_variates, patches, d_model)`` tensor, and reuse the shared pooling +
``channel_aggregation`` (``F = n_variates * d_model`` under the default ``concat``).
It exposes no Chronos-style special tokens, so ``n_special_tokens = 0`` and the four
pooling names map onto its patch layout as documented in ``models/base.py``:

  * forecast_token -> MOMENT's summary/CLS reconstruction embedding if present, else
    the last patch (the closest "predict-next" summary).  DECISION (uncited): a documented judgment call.
  * last_content   -> the last patch (== forecast_token here, no special tokens).
  * mean           -> mean over patches.
  * flatten        -> all patches concatenated (fixed-context only).

MOMENT does not surface its RevIN loc/scale, so loc/scale is the per-channel input
mean/std (``TSFMEmbedderBase.loc_scale_from_contexts``).  DECISION (uncited): a documented judgment call.

The backbone is imported and called ONLY inside ``_load_pipeline`` / ``_encode_batch``
(the sanctioned ``# pragma: no cover`` boundary); everything testable is in the shared
base. Documented fallback if ``.embed()`` will not surface per-patch states: the
encoder's penultimate hidden states (RESEARCH_PLAN §11).  DECISION (uncited): a documented judgment call.
"""

from __future__ import annotations

import numpy as np

from .base import TSFMEmbedderBase


class MomentEmbedder(TSFMEmbedderBase):
    embedder_name = "MomentEmbedder"
    layout = "univariate"
    n_special_tokens = 0

    def _load_pipeline(self):  # pragma: no cover -- GPU-only heavy backbone load
        if self._pipeline is not None:
            return self._pipeline
        import torch
        from momentfm import MOMENTPipeline  # lazy heavy import (GPU-only)

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = MOMENTPipeline.from_pretrained(
            self.model_name, model_kwargs={"task_name": "embedding"})
        model.init()
        model = model.to(device).eval()
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
                                         device=self._device_resolved)[None, None, :]
                with torch.inference_mode():
                    out = model(x_enc=series)
                # (1, patches, d_model) encoder states -> (patches, d_model)
                hidden = out.embeddings if hasattr(out, "embeddings") else out
                per_channel.append(np.asarray(hidden, np.float32).reshape(
                    -1, hidden.shape[-1]))
            canonical.append(np.stack(per_channel, axis=0))  # (C, patches, d_model)
        loc_scale = self.loc_scale_from_contexts(batch)
        return canonical, loc_scale
