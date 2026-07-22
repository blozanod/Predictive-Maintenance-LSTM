"""Moirai-2 embedder (Salesforce/moirai-2): a multivariate-native "any-variate" TSFM.

Moirai feeds all channels JOINTLY, flattening the variates with variate-ids (uni2ts
native), so it is the second member of the multivariate-native family alongside
Chronos-2 -- the category is not a category-of-one (RESEARCH_PLAN §6, RQ-M). We map
its any-variate output back to the canonical ``(n_variates, patches, d_model)`` tensor
so the shared pooling + ``channel_aggregation`` apply exactly as for every other model
(``F = n_variates * d_model`` under ``concat``). It appends no Chronos-style special
tokens, so ``n_special_tokens = 0`` and the four pooling names map as in
``models/base.py``:

  * forecast_token -> the last per-variate patch (Moirai's next-step summary).  DECISION (uncited): a documented judgment call.
  * last_content   -> the last patch (== forecast_token; no special tokens).
  * mean / flatten -> over all patches (flatten fixed-context only).

Moirai instance-normalizes internally; when its loc/scale is not surfaced we fall
back to the per-channel input mean/std (``loc_scale_from_contexts``).  DECISION (uncited): a documented judgment call.

Backbone import + call live only in ``_load_pipeline`` / ``_encode_batch`` (the
``# pragma: no cover`` boundary). Documented fallback if the any-variate encoder
states are hard to recover cleanly: the module's encoder hidden states
(RESEARCH_PLAN §11).
"""

from __future__ import annotations

import numpy as np

from .base import TSFMEmbedderBase


class MoiraiEmbedder(TSFMEmbedderBase):
    embedder_name = "MoiraiEmbedder"
    layout = "multivariate"
    n_special_tokens = 0

    def _load_pipeline(self):  # pragma: no cover -- GPU-only heavy backbone load
        if self._pipeline is not None:
            return self._pipeline
        import torch
        from uni2ts.model.moirai2 import Moirai2Module  # lazy heavy import (GPU-only)

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        module = Moirai2Module.from_pretrained(self.model_name).to(device).eval()
        self._torch = torch
        self._device_resolved = device
        self._pipeline = module
        return self._pipeline

    def _encode_batch(self, batch):  # pragma: no cover -- GPU-only backbone call
        module = self._load_pipeline()
        torch = self._torch
        canonical = []
        for w in batch:
            w = np.asarray(w, np.float32)                 # (L, C)
            series = torch.as_tensor(w, dtype=torch.float32,
                                     device=self._device_resolved).transpose(0, 1)[None]
            with torch.inference_mode():
                hidden = module.encode(series)            # (1, n_variates, patches, d_model)
            arr = np.asarray(hidden, np.float32)
            canonical.append(arr.reshape(arr.shape[-3], arr.shape[-2], arr.shape[-1]))
        loc_scale = self.loc_scale_from_contexts(batch)
        return canonical, loc_scale
