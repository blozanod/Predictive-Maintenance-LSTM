"""Regression head + loss variants for frozen-TSFM embeddings.

Head: 2-layer MLP (hidden 256, dropout) mirroring arXiv:2606.11990 (Task 2.2).
``head_num_layers=1`` gives the linear-head ablation.

Losses (Task 2.6, RESEARCH_PLAN sec.5):
  * ``mse``      -- point regression, the comparability anchor.
  * ``corn``     -- CORN ordinal loss over K RUL bins. We REUSE ``coral_pytorch``
                    (``corn_loss``, ``corn_label_from_logits``); we do NOT
                    reimplement it (Task 2.1). CORN: Shi, Cao & Raschka
                    arXiv:2111.08851.
  * ``quantile`` -- pinball loss over configured quantile levels (optional arm).

Ordinal decoding supports expected-value (default) and argmax (ablation,
RESEARCH_PLAN sec.11). ``decode`` always returns RUL in ``[0, max_rul]``.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .config import Config


# ---------------------------------------------------------------------------
# RUL <-> ordinal bins
# ---------------------------------------------------------------------------
def bin_width(config: Config) -> float:
    return config.max_rul / config.num_bins  # e.g. 125/25 = 5 cycles (RESEARCH_PLAN sec.5)


def rul_to_bin(rul: np.ndarray, config: Config) -> np.ndarray:
    """Continuous RUL -> integer bin index in [0, num_bins-1] (labels start at 0,
    as CORN requires)."""
    idx = np.floor(np.asarray(rul, dtype=np.float64) / bin_width(config)).astype(np.int64)
    return np.clip(idx, 0, config.num_bins - 1)


def bin_to_rul(bin_idx: np.ndarray, config: Config) -> np.ndarray:
    """Bin index (possibly fractional expected rank) -> RUL at the bin center."""
    return (np.asarray(bin_idx, dtype=np.float64) + 0.5) * bin_width(config)


# The censored/alarm arm (RESEARCH_PLAN §4; CHANGES.md §54). Unlike every other arm it
# predicts a PROBABILITY, not a RUL in cycles, so its outputs are scored by
# evaluate.alarm_metrics and are NEVER tabled against NASA/RMSE (non-comparability).
ALARM_LOSS = "failure_within_horizon"


def is_alarm_loss(loss_type: str) -> bool:
    """True for the binary censoring-aware arm, whose predictions are probabilities
    rather than RUL -- the one arm decoded and scored on a different scale."""
    return loss_type == ALARM_LOSS


def alarm_targets(targets_rul, config: Config):
    """Binary alarm target read off the RUL label: 1 iff the intervention falls within
    ``config.alarm_horizon`` cycles. Deriving it here (rather than plumbing a second
    label through the cache) keeps ONE target array end to end; the censored rows whose
    label is unknowable were already dropped upstream (``data.add_alarm_label``), and
    ``Config`` guarantees ``alarm_horizon < max_rul`` so RUL clipping cannot erase the
    distinction."""
    if config.alarm_horizon is None:
        raise ValueError(
            f"loss {ALARM_LOSS!r} requires config.alarm_horizon to be set (it defines "
            f"'within how many cycles?'). Set it on the censored dataset's config.")
    return (targets_rul <= float(config.alarm_horizon)).to(targets_rul.dtype)


def head_output_dim(loss_type: str, config: Config) -> int:
    if loss_type == "mse":
        return 1
    if loss_type == "corn":
        return config.num_bins - 1  # CORN emits num_classes-1 logits (coral_pytorch)
    if loss_type == "quantile":
        return len(config.quantile_levels)
    if loss_type == ALARM_LOSS:
        return 1  # one logit -> P(intervention within alarm_horizon)
    raise ValueError(f"unknown loss_type: {loss_type!r}")


# ---------------------------------------------------------------------------
# Head
# ---------------------------------------------------------------------------
class MLPHead(nn.Module):
    """MLP regression/ordinal head over a frozen embedding vector."""

    def __init__(self, input_dim: int, output_dim: int, config: Config):
        super().__init__()
        h = config.head_hidden_dim
        layers: list[nn.Module] = []
        if config.head_num_layers <= 1:
            layers.append(nn.Linear(input_dim, output_dim))  # linear-head ablation
        else:
            layers += [nn.Linear(input_dim, h), nn.ReLU(), nn.Dropout(config.head_dropout)]
            for _ in range(config.head_num_layers - 2):
                layers += [nn.Linear(h, h), nn.ReLU(), nn.Dropout(config.head_dropout)]
            layers.append(nn.Linear(h, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_head(input_dim: int, loss_type: str, config: Config) -> MLPHead:
    return MLPHead(input_dim, head_output_dim(loss_type, config), config)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def pinball_loss(
    preds: torch.Tensor, target: torch.Tensor, quantiles: list[float]
) -> torch.Tensor:
    """Mean pinball/quantile loss. ``preds`` (N, Q), ``target`` (N,), scaled to
    [0,1] to match the MSE arm when scale_targets is on."""
    q = torch.as_tensor(quantiles, dtype=preds.dtype, device=preds.device)  # (Q,)
    err = target.unsqueeze(1) - preds                                       # (N, Q)
    return torch.maximum(q * err, (q - 1.0) * err).mean()


def compute_loss(
    outputs: torch.Tensor, targets_rul: torch.Tensor, loss_type: str, config: Config
) -> torch.Tensor:
    """Loss from raw head outputs and CONTINUOUS RUL targets (unscaled cycles)."""
    if loss_type == "mse":
        target = targets_rul / config.max_rul if config.scale_targets else targets_rul
        return nn.functional.mse_loss(outputs.squeeze(-1), target)
    if loss_type == "corn":
        from coral_pytorch.losses import corn_loss  # reuse reference impl (Task 2.1)
        bins = rul_to_bin(targets_rul.detach().cpu().numpy(), config)
        y = torch.as_tensor(bins, dtype=torch.long, device=outputs.device)
        return corn_loss(outputs, y, num_classes=config.num_bins)
    if loss_type == "quantile":
        target = targets_rul / config.max_rul if config.scale_targets else targets_rul
        return pinball_loss(outputs, target, config.quantile_levels)
    if loss_type == ALARM_LOSS:
        # Binary cross-entropy on one logit (numerically-stable fused form).
        return nn.functional.binary_cross_entropy_with_logits(
            outputs.squeeze(-1), alarm_targets(targets_rul, config))
    raise ValueError(f"unknown loss_type: {loss_type!r}")


# ---------------------------------------------------------------------------
# Decoding -> RUL in [0, max_rul]
# ---------------------------------------------------------------------------
def corn_expected_rank(logits: torch.Tensor) -> torch.Tensor:
    """Expected ordinal rank E[y] from CORN logits.

    CORN outputs conditional probabilities f_k = sigmoid(logit_k) = P(y>k | y>k-1),
    so the cumulative P(y>k) = prod_{j<=k} f_j. For a non-negative integer y in
    {0..K-1}, E[y] = sum_{k>=0} P(y>k) = sum over the cumulative-product columns.
    (Standard CORN construction, arXiv:2111.08851 -- coral_pytorch exposes only
    argmax labels, so the expectation is computed here from its probabilities.)
    """
    cum = torch.sigmoid(logits).cumprod(dim=1)  # (N, K-1): column k is P(y>k)
    return cum.sum(dim=1)                        # (N,)


def decode(outputs: torch.Tensor, loss_type: str, config: Config) -> np.ndarray:
    """Raw head outputs -> predicted RUL (numpy, clipped to [0, max_rul]).

    EXCEPTION -- the ``failure_within_horizon`` arm returns a PROBABILITY in [0, 1],
    not a RUL: it answers a different question on a different scale, and its outputs
    are scored by ``evaluate.alarm_metrics`` and never tabled against RUL metrics
    (§54). Callers distinguish the two with ``heads.is_alarm_loss``."""
    if is_alarm_loss(loss_type):
        with torch.no_grad():
            prob = torch.sigmoid(outputs.squeeze(-1))
        return prob.detach().cpu().numpy().astype(np.float64)
    with torch.no_grad():
        if loss_type == "mse":
            pred = outputs.squeeze(-1)
            if config.scale_targets:
                pred = pred * config.max_rul
            rul = pred.detach().cpu().numpy()
        elif loss_type == "corn":
            if config.corn_decoding == "expected_value":
                rank = corn_expected_rank(outputs).detach().cpu().numpy()
                rul = bin_to_rul(rank, config)
            elif config.corn_decoding == "argmax":
                from coral_pytorch.dataset import corn_label_from_logits  # reuse impl
                label = corn_label_from_logits(outputs).detach().cpu().numpy()
                rul = bin_to_rul(label, config)
            else:
                raise ValueError(f"unknown corn_decoding: {config.corn_decoding!r}")
        elif loss_type == "quantile":
            # Point estimate = median quantile if present, else the middle column.
            levels = config.quantile_levels
            j = levels.index(0.5) if 0.5 in levels else len(levels) // 2
            pred = outputs[:, j]
            if config.scale_targets:
                pred = pred * config.max_rul
            rul = pred.detach().cpu().numpy()
        else:
            raise ValueError(f"unknown loss_type: {loss_type!r}")
    return np.clip(rul, 0.0, config.max_rul).astype(np.float64)
