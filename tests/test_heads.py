"""Head + loss correctness: output dims, bin mapping, CORN decode range, losses."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.config import Config
from src import heads as H


def test_head_output_dims():
    cfg = Config(num_bins=25, quantile_levels=[0.1, 0.5, 0.9])
    assert H.head_output_dim("mse", cfg) == 1
    assert H.head_output_dim("corn", cfg) == cfg.num_bins - 1  # CORN emits K-1 logits
    assert H.head_output_dim("quantile", cfg) == 3


def test_bin_mapping_range_and_roundtrip():
    cfg = Config(max_rul=125, num_bins=25)  # width 5
    rul = np.array([0, 2.4, 5, 7.5, 124, 125, 130])
    bins = H.rul_to_bin(rul, cfg)
    assert bins.min() >= 0 and bins.max() <= cfg.num_bins - 1
    assert bins[0] == 0 and bins[2] == 1
    # decode of a bin index lands at that bin's center, within [0, max_rul]
    centers = H.bin_to_rul(np.arange(cfg.num_bins), cfg)
    assert centers.min() >= 0 and centers.max() <= cfg.max_rul
    assert centers[0] == pytest.approx(2.5)
    assert centers[-1] == pytest.approx(122.5)


@pytest.mark.parametrize("decoding", ["expected_value", "argmax"])
def test_corn_decode_in_range(decoding):
    cfg = Config(max_rul=125, num_bins=25, corn_decoding=decoding)
    torch.manual_seed(0)
    logits = torch.randn(64, cfg.num_bins - 1) * 5  # extreme logits too
    rul = H.decode(logits, "corn", cfg)
    assert rul.shape == (64,)
    assert rul.min() >= 0.0 and rul.max() <= cfg.max_rul  # Task 2.7 requirement


def test_corn_expected_rank_matches_probability_definition():
    cfg = Config(max_rul=50, num_bins=10)
    logits = torch.randn(16, cfg.num_bins - 1)
    rank = H.corn_expected_rank(logits)
    # E[y] = sum_k P(y>k) must lie in [0, K-1].
    assert torch.all(rank >= 0) and torch.all(rank <= cfg.num_bins - 1)


def test_mse_and_quantile_decode_range_and_scaling():
    cfg = Config(max_rul=125, scale_targets=True, quantile_levels=[0.1, 0.5, 0.9])
    # MSE head outputs are in scaled [0,1] space; decode multiplies back by max_rul.
    out = torch.tensor([[0.0], [0.5], [1.0], [2.0]])
    rul = H.decode(out, "mse", cfg)
    assert rul.tolist() == [0.0, 62.5, 125.0, 125.0]  # clipped at max_rul
    q = torch.tensor([[0.1, 0.4, 0.9]])
    rq = H.decode(q, "quantile", cfg)  # median column (0.5) used as point estimate
    assert rq[0] == pytest.approx(50.0)


@pytest.mark.parametrize("loss", ["mse", "corn", "quantile"])
def test_compute_loss_finite_and_backprop(loss):
    cfg = Config(max_rul=125, num_bins=25, quantile_levels=[0.1, 0.5, 0.9])
    head = H.build_head(16, loss, cfg)
    x = torch.randn(20, 16)
    targets = torch.rand(20) * cfg.max_rul
    out = head(x)
    val = H.compute_loss(out, targets, loss, cfg)
    assert torch.isfinite(val)
    val.backward()  # gradients flow
    assert any(p.grad is not None for p in head.parameters())


def test_linear_head_ablation():
    cfg = Config(head_num_layers=1)
    head = H.build_head(8, "mse", cfg)
    # single Linear layer only
    linears = [m for m in head.net if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 1
