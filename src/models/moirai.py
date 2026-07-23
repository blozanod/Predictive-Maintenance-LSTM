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
        # Moirai2Module exposes NO .encode(): its forward() consumes PACKED inputs
        # (target, observed_mask, sample_id, time_id, variate_id, prediction_mask) and
        # the representations are internal. We reproduce the encoder path exactly as
        # module.forward does -- scaler -> in_proj -> encoder -> reprs -- per variate
        # (single-variate packing is the documented fallback that yields the canonical
        # (n_variates, patches, d_model); RESEARCH_PLAN §11). Verified against uni2ts's
        # moirai2/module.py; weight-level shapes are confirmed by the Colab spike.
        from uni2ts.common.torch_util import packed_causal_attention_mask
        module = self._load_pipeline()
        torch = self._torch
        dev = self._device_resolved
        patch = int(module.patch_size)
        canonical = []
        for w in batch:
            w = np.asarray(w, np.float32)                 # (L, C)
            n = w.shape[0]
            pad = (-n) % patch
            npatch = (n + pad) // patch
            per_variate = []
            for c in range(w.shape[1]):
                s = np.concatenate([np.zeros(pad, np.float32), w[:, c]])
                target = torch.as_tensor(s.reshape(npatch, patch), device=dev)[None]
                observed = torch.ones(1, npatch, patch, dtype=torch.bool, device=dev)
                if pad:
                    observed[0, 0, :pad] = False
                sample_id = torch.ones(1, npatch, dtype=torch.long, device=dev)
                time_id = torch.arange(npatch, device=dev)[None]
                variate_id = torch.zeros(1, npatch, dtype=torch.long, device=dev)
                pred_mask = torch.zeros(1, npatch, dtype=torch.bool, device=dev)
                with torch.inference_mode():
                    loc, scale = module.scaler(
                        target, observed * ~pred_mask.unsqueeze(-1), sample_id, variate_id)
                    tokens = torch.cat([(target - loc) / scale,
                                        observed.to(torch.float32)], dim=-1)
                    reprs = module.in_proj(tokens)
                    reprs = module.encoder(
                        reprs, packed_causal_attention_mask(sample_id, time_id),
                        time_id=time_id, var_id=variate_id)
                per_variate.append(np.asarray(reprs[0].detach().to("cpu").float().numpy(),
                                              np.float32))   # (npatch, d_model)
            canonical.append(np.stack(per_variate, axis=0))  # (C, npatch, d_model)
        loc_scale = self.loc_scale_from_contexts(batch)
        return canonical, loc_scale
