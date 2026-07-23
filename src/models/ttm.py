"""TTM embedder (ibm-granite/granite-timeseries-ttm-r2): a tiny channel-mixing TSFM.

Tiny Time Mixers (~1-5M params) is the "does scale matter?" foil (RESEARCH_PLAN §6).
It is multivariate-native -- its mixer blocks mix channels internally -- but per-variate
embeddings still exist after the backbone, so we read them into the canonical
``(n_variates, patches, d_model)`` tensor and reuse the shared pooling +
``channel_aggregation`` (``F = n_variates * d_model`` under ``concat``). No
Chronos-style special tokens, so ``n_special_tokens = 0`` and the four pooling names
map as in ``models/base.py``:

  * forecast_token -> the last per-variate patch (the next-step summary).  DECISION (uncited): a documented judgment call.
  * last_content   -> the last patch (== forecast_token; no special tokens).
  * mean / flatten -> over all patches (flatten fixed-context only).

TTM's per-channel RevIN loc/scale is not always surfaced; the fallback is the
per-channel input mean/std (``loc_scale_from_contexts``).  DECISION (uncited): a documented judgment call.

Backbone import + call live only in ``_load_pipeline`` / ``_encode_batch`` (the
``# pragma: no cover`` boundary). Documented fallback if the backbone exposes only a
pooled output: its encoder/backbone hidden states (RESEARCH_PLAN §11).
"""

from __future__ import annotations

import numpy as np

from .base import TSFMEmbedderBase


class TTMEmbedder(TSFMEmbedderBase):
    embedder_name = "TTMEmbedder"
    layout = "multivariate"
    n_special_tokens = 0

    def _load_pipeline(self):  # pragma: no cover -- GPU-only heavy backbone load
        if self._pipeline is not None:
            return self._pipeline
        import torch
        from tsfm_public import get_model  # lazy heavy import (GPU-only; granite-tsfm)

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        # get_model (verified against granite-tsfm) REQUIRES prediction_length and, for a
        # requested context below the shortest pretrained TTM (512 for r2), raises unless
        # force_return="zeropad" -- which returns a >=512-context model we zero-pad into.
        # We only need the backbone hidden states, so prediction_length is a placeholder.
        model = get_model(self.model_name, context_length=self.context_length,
                          prediction_length=1, force_return="zeropad")
        model = model.to(device).eval()
        self._torch = torch
        self._device_resolved = device
        self._pipeline = model
        return self._pipeline

    def _encode_batch(self, batch):  # pragma: no cover -- GPU-only backbone call
        # TTM's patching is fixed to model.config.context_length; place each window's
        # most-recent `take` cycles at the END of a context-length buffer (front zero-pad,
        # the get_model "zeropad" contract). forward(past_values=...) returns
        # backbone_hidden_state (batch, n_variates, num_patches, d_model) -- verified
        # against granite-tsfm's TinyTimeMixerForPredictionOutput. TTM is multivariate-
        # native (one forward per window, all channels jointly); every window buffers to
        # the same (ctx, C) shape, so whole WINDOWS stack into ONE forward via
        # _grouped_forward instead of one call per window (CHANGES.md §46).
        model = self._load_pipeline()
        torch = self._torch
        dev = self._device_resolved
        ctx = int(model.config.context_length)

        bufs = []
        for w in batch:
            w = np.asarray(w, np.float32)                 # (L, C)
            take = min(w.shape[0], ctx)
            buf = np.zeros((ctx, w.shape[1]), np.float32)
            buf[-take:] = w[-take:]                        # right-aligned; front zero-pad
            bufs.append(buf)                               # (ctx, C), uniform within a dataset

        def _fwd(group):
            xb = torch.as_tensor(np.stack(group), device=dev)              # (b, ctx, C)
            # The r2.1 model card serves frequency-prefix-tuned ("-ft-") revisions whose
            # forward REQUIRES a freq_token (batch,) or raises "Expecting freq_token".
            # DECISION (uncited): freq index 0 (base/unknown frequency) -- we extract
            # representations, not forecast a specific-cadence series; harmless (unused)
            # for non-ft variants. ft models prepend a freq patch -> patches+1, which the
            # shared pooling absorbs.
            freq_token = torch.zeros(xb.shape[0], dtype=torch.long, device=dev)
            with torch.inference_mode():
                out = model(past_values=xb, freq_token=freq_token)
            hidden = np.asarray(out.backbone_hidden_state.detach().to("cpu").float().numpy(),
                                np.float32)                # (b, n_variates, patches, d_model)
            return [hidden[k] for k in range(hidden.shape[0])]   # each (n_variates, patches, d)

        # multivariate-native: one canonical (n_variates, patches, d) per WINDOW already,
        # so no per-channel regroup -- group the whole window buffers by shape.
        canonical = self._grouped_forward(bufs, lambda b: b.shape, _fwd)
        return canonical, self.loc_scale_from_contexts(batch)
