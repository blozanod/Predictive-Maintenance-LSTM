"""Moirai-2 embedder (Salesforce/moirai-2.0-R-small): a multivariate-native "any-variate" TSFM.

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
        # Series that share (num_patches, front-pad) stack into ONE encoder call via
        # _grouped_forward -- the batch-1 per-(window, channel) loop left the GPU ~95%
        # idle (CHANGES.md §46); each batched element is still an independent
        # single-variate sequence (sample_id constant within an element), so the packed
        # attention is unchanged.
        from uni2ts.common.torch_util import packed_causal_attention_mask
        module = self._load_pipeline()
        torch = self._torch
        dev = self._device_resolved
        patch = int(module.patch_size)

        items, chans = [], []
        for w in batch:
            w = np.asarray(w, np.float32)                 # (L, C)
            chans.append(w.shape[1])
            n = w.shape[0]
            pad = (-n) % patch
            npatch = (n + pad) // patch
            for c in range(w.shape[1]):
                s = np.concatenate([np.zeros(pad, np.float32), w[:, c]])
                items.append((s.reshape(npatch, patch), npatch, pad))   # (npatch, patch)

        def _fwd(group):
            b = len(group)
            npatch, pad = group[0][1], group[0][2]
            target = torch.as_tensor(np.stack([g[0] for g in group]), device=dev)  # (b,np,patch)
            observed = torch.ones(b, npatch, patch, dtype=torch.bool, device=dev)
            if pad:
                observed[:, 0, :pad] = False
            sample_id = torch.ones(b, npatch, dtype=torch.long, device=dev)
            time_id = torch.arange(npatch, device=dev)[None].expand(b, npatch)
            variate_id = torch.zeros(b, npatch, dtype=torch.long, device=dev)
            pred_mask = torch.zeros(b, npatch, dtype=torch.bool, device=dev)
            with torch.inference_mode():
                loc, scale = module.scaler(
                    target, observed * ~pred_mask.unsqueeze(-1), sample_id, variate_id)
                tokens = torch.cat([(target - loc) / scale,
                                    observed.to(torch.float32)], dim=-1)
                reprs = module.in_proj(tokens)
                reprs = module.encoder(
                    reprs, packed_causal_attention_mask(sample_id, time_id),
                    time_id=time_id, var_id=variate_id)
            arr = np.asarray(reprs.detach().to("cpu").float().numpy(), np.float32)  # (b,np,d)
            return [arr[k] for k in range(arr.shape[0])]

        flat = self._grouped_forward(items, lambda it: (it[1], it[2]), _fwd)  # key=(npatch,pad)
        return self._regroup_channels(flat, chans), self.loc_scale_from_contexts(batch)
