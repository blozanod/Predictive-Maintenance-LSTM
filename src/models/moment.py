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
        # MOMENT-1 has a FIXED input length (config.seq_len, 512 for -large) with no
        # auto-padding: the positional embedding is sized for it, so a raw variable-
        # length series raises. We place each channel's most-recent min(L, seq_len)
        # cycles into a seq_len buffer + an input_mask marking the valid positions (the
        # standard MOMENT contract; verified against momentfm 0.1.4). Every (window,
        # channel) series shares seq_len, so they all stack into ONE embed() call rather
        # than one call per series (batch-1 left the GPU ~95% idle -- CHANGES.md §46);
        # embed(reduction="mean") returns one summary vector per series -> canonical
        # (C, 1, d_model): a single "patch" the shared pooling collapses to (per
        # RESEARCH_PLAN §6, MOMENT is used as a per-channel summary).
        model = self._load_pipeline()
        torch = self._torch
        dev = self._device_resolved
        seq_len = int(model.config.seq_len)

        items, chans = [], []
        for w in batch:
            w = np.asarray(w, np.float32)                 # (L, C)
            chans.append(w.shape[1])
            take = min(w.shape[0], seq_len)
            mask = np.zeros(seq_len, np.float32); mask[:take] = 1.0
            for c in range(w.shape[1]):
                buf = np.zeros(seq_len, np.float32)
                buf[:take] = w[-take:, c]                 # most-recent `take` cycles
                items.append((buf, mask))

        def _fwd(group):
            xb = torch.as_tensor(np.stack([g[0] for g in group]),
                                 device=dev)[:, None, :]        # (b, 1, seq_len)
            mb = torch.as_tensor(np.stack([g[1] for g in group]), device=dev)  # (b, seq_len)
            with torch.inference_mode():
                out = model.embed(x_enc=xb, input_mask=mb, reduction="mean")
            emb = np.asarray(out.embeddings.detach().to("cpu").float().numpy(), np.float32)
            return [emb[k][None, :] for k in range(emb.shape[0])]   # each (1, d_model)

        flat = self._grouped_forward(items, lambda it: it[0].shape[0], _fwd)  # key=seq_len
        return self._regroup_channels(flat, chans), self.loc_scale_from_contexts(batch)
