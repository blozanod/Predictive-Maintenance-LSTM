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
        # convention), and read output_embeddings (b, num_patches, model_dims).
        # DECISION (uncited): per-series (not running-stat) normalization for embedding.
        # Series that share a patch count (few distinct values: ceil(L/p)) stack into ONE
        # forward via _grouped_forward instead of one call per (window, channel) -- the
        # batch-1 loop left the GPU ~95% idle (CHANGES.md §46).
        model = self._load_pipeline()
        torch = self._torch
        dev = self._device_resolved
        core = model.model.to(dev).eval()                     # the nn.Module
        p = int(core.p)                                        # input patch length (32)

        items, chans = [], []
        for w in batch:
            w = np.asarray(w, np.float32)                     # (L, C)
            chans.append(w.shape[1])
            for c in range(w.shape[1]):
                s = w[:, c]
                pad = (-len(s)) % p
                padded = np.concatenate([np.zeros(pad, np.float32), s])
                is_pad = np.concatenate([np.ones(pad, bool), np.zeros(len(s), bool)])
                mu = float(s.mean()); sigma = float(s.std()) or 1.0
                normed = np.where(is_pad, 0.0, (padded - mu) / sigma).astype(np.float32)
                items.append((normed.reshape(-1, p), is_pad.reshape(-1, p)))  # (np,p),(np,p)

        def _fwd(group):
            xb = torch.as_tensor(np.stack([g[0] for g in group]), device=dev)  # (b, np, p)
            mb = torch.as_tensor(np.stack([g[1] for g in group]), device=dev)  # (b, np, p)
            with torch.inference_mode():
                (_, output_emb, _, _), _ = core(xb, mb)       # (b, np, d_model)
            arr = np.asarray(output_emb.detach().to("cpu").float().numpy(), np.float32)
            return [arr[k] for k in range(arr.shape[0])]      # each (np, d_model)

        flat = self._grouped_forward(items, lambda it: it[0].shape[0], _fwd)  # key=num_patches
        return self._regroup_channels(flat, chans), self.loc_scale_from_contexts(batch)
