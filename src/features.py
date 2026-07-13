"""Head-feature assembly (Task 1.1): compose the MLP head's input from cached
signals, with a leakage-safe standardizer.

``head_features`` selects which cached signals feed the head:
  * ``emb``               -- pooled embedding only (baseline).
  * ``emb+locscale``      -- + the per-window Chronos-2 instance-norm loc/scale,
                             flattened and standardized. This is the degradation-
                             level signal the internal normalization discards
                             (the dominant fix for the 17.4-RMSE regression).
  * ``emb+locscale+raw``  -- + the window's last-cycle raw sensors, standardized
                             (Wide & Deep-lite, mirrors the PHM 10.32 paper).

Leakage rule (Task 2.4): the standardizer for the APPENDED columns (loc/scale, raw)
is fit on the CURRENT data fraction's TRAIN split rows only. The embedding block is
passed through unstandardized (the MLP handles it, as before). All ops run on the
tensors' device, so the sweep can keep the whole cache on the GPU (Task 2 Stage B).
"""

from __future__ import annotations

import torch

from .config import Config


def _flatten_locscale(locscale: torch.Tensor) -> torch.Tensor:
    """(n, n_variates, 2) -> (n, 2*n_variates), interleaving [loc, scale] per variate."""
    return locscale.reshape(locscale.shape[0], -1)


class HeadFeatureBuilder:
    """Fit-on-train / transform-anywhere builder for the head input matrix.

    Usage (per data-fraction cell):
        b = HeadFeatureBuilder(config).fit(locscale[tr], raw_last[tr])
        Xtr = b.transform(emb[tr], locscale[tr], raw_last[tr])
        Xva = b.transform(emb[va], locscale[va], raw_last[va])
        Xte = b.transform(emb_te, locscale_te, raw_te)
    """

    def __init__(self, config: Config):
        self.mode = config.head_features
        self.max_rul = float(config.max_rul)
        self._use_ls = self.mode in ("emb+locscale", "emb+locscale+raw")
        self._use_raw = self.mode == "emb+locscale+raw"
        self._ls_mean = self._ls_std = None
        self._raw_mean = self._raw_std = None

    @staticmethod
    def _fit_stats(x: torch.Tensor):
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        std = torch.where(std < 1e-8, torch.ones_like(std), std)  # guard flat columns
        return mean, std

    def fit(self, locscale_train: torch.Tensor, raw_train: torch.Tensor) -> "HeadFeatureBuilder":
        if self._use_ls:
            self._ls_mean, self._ls_std = self._fit_stats(_flatten_locscale(locscale_train))
        if self._use_raw:
            self._raw_mean, self._raw_std = self._fit_stats(raw_train)
        return self

    def transform(self, emb: torch.Tensor, locscale: torch.Tensor,
                  raw_last: torch.Tensor) -> torch.Tensor:
        blocks = [emb]  # embeddings pass through unstandardized
        if self._use_ls:
            if self._ls_mean is None:
                raise RuntimeError("HeadFeatureBuilder.transform called before fit()")
            ls = (_flatten_locscale(locscale) - self._ls_mean) / self._ls_std
            blocks.append(ls)
        if self._use_raw:
            raw = (raw_last - self._raw_mean) / self._raw_std
            blocks.append(raw)
        return torch.cat(blocks, dim=1)

    def output_dim(self, emb_dim: int, n_variates: int, n_channels: int) -> int:
        dim = emb_dim
        if self._use_ls:
            dim += 2 * n_variates
        if self._use_raw:
            dim += n_channels
        return dim


def raw_last_cycle(windows):
    """Last-cycle raw sensors from fixed windows ``(n, W, C)`` -> ``(n, C)``. Works on
    numpy arrays or torch tensors. The fixed window ends at the prediction cycle
    (front-padded if short), so ``[:, -1]`` is always the true last-cycle reading --
    aligned to the TSFM context's last cycle.
    """
    return windows[:, -1, :]
