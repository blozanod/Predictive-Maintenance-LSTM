"""TimesFM 2.5 embedder (google/timesfm-2.5-200m-pytorch): a univariate decoder-only TSFM.

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

        # model_name is the real HF weights id "google/timesfm-2.5-200m-pytorch" (the
        # registry key; the earlier "google/timesfm-2.5" 404'd on Colab). torch_compile=
        # False keeps the eager module.forward we read hidden states from.
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            self.model_name, torch_compile=False)
        self._torch = torch
        self._device_resolved = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipeline = model
        return self._pipeline

    def _encode_batch(self, batch):  # pragma: no cover -- GPU-only backbone call
        # TimesFM 2.5 exposes NO .embed(); the transformer stack's per-patch hidden
        # states are output_embeddings (index 1) of the underlying module's forward
        # (verified against timesfm_2p5_torch). We patch each channel to the input patch
        # length p (front-pad + mask), instance-normalize (RevIN, the model's own
        # convention), and read output_embeddings (1, num_patches, model_dims).
        # DECISION (uncited): per-series (not running-stat) normalization for embedding.
        model = self._load_pipeline()
        torch = self._torch
        core = model.model.to(self._device_resolved).eval()   # the nn.Module
        p = int(core.p)                                        # input patch length (32)
        canonical = []
        for w in batch:
            w = np.asarray(w, np.float32)                     # (L, C)
            per_channel = []
            for c in range(w.shape[1]):
                s = w[:, c]
                pad = (-len(s)) % p
                padded = np.concatenate([np.zeros(pad, np.float32), s])
                is_pad = np.concatenate([np.ones(pad, bool), np.zeros(len(s), bool)])
                mu = float(s.mean()); sigma = float(s.std()) or 1.0
                normed = np.where(is_pad, 0.0, (padded - mu) / sigma).astype(np.float32)
                x = torch.as_tensor(normed.reshape(-1, p),
                                    device=self._device_resolved)[None]     # (1, np, p)
                m = torch.as_tensor(is_pad.reshape(-1, p),
                                    device=self._device_resolved)[None]     # (1, np, p)
                with torch.inference_mode():
                    (_, output_emb, _, _), _ = core(x, m)     # output_emb (1, np, d_model)
                per_channel.append(np.asarray(output_emb[0].detach().to("cpu")
                                              .float().numpy(), np.float32))
            canonical.append(np.stack(per_channel, axis=0))   # (C, num_patches, d_model)
        loc_scale = self.loc_scale_from_contexts(batch)
        return canonical, loc_scale
