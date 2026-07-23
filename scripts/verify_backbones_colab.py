"""Colab GPU spike: validate every TSFM embedder end-to-end on REAL weights.

The four v2 backbone `_encode_batch` / `_load_pipeline` bodies are the sole
`# pragma: no cover` boundary -- CPU tests mock them, and this repo's CI container has
no GPU and no HuggingFace egress, so their bodies are verified against each library's
SOURCE but cannot be run there. This script is the missing weight-level check: run it on
Colab (GPU + `pip install -r requirements.txt`) and it loads each real model and runs the
actual embedding path on a tiny synthetic context, reporting shapes, finiteness, and
timing -- exactly the Phase-1 integration spike RESEARCH_PLAN §9/§11 calls for.

    !pip install -r requirements.txt
    !python scripts/verify_backbones_colab.py                 # all five
    !python scripts/verify_backbones_colab.py --models amazon/chronos-2 ibm-granite/granite-timeseries-ttm-r2

A backbone that fails here is "reported as such, not forced" (RESEARCH_PLAN §11): the
verified fallback is the module's encoder/penultimate hidden states, already how the
univariate + Moirai bodies extract representations. If a HuggingFace repo id differs from
the registry key (e.g. TimesFM's weights are at `google/timesfm-2.5-200m-pytorch`), pass
the real id via --models and update `src/models/__init__.EMBEDDERS` once confirmed.

Exit code is non-zero if ANY requested backbone fails, so it doubles as a CI gate on a
GPU runner.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# Make the repo root importable however the script is launched (scripts/ is on
# sys.path when run directly, the repo root is not).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import Config, DEFAULT_SENSOR_COLUMNS
from src.models import EMBEDDERS, make_embedder


def _synthetic_contexts(n_windows: int, n_channels: int, lengths=(30, 96, 200)):
    """A few variable-length `(L_i, C)` degradation-ish contexts -- mirrors what
    `data.make_windows_varlen` feeds `embed_windows` (variable history per window)."""
    rng = np.random.default_rng(0)
    out = []
    for i in range(n_windows):
        L = lengths[i % len(lengths)]
        trend = np.linspace(0.0, 1.0, L)[:, None] * rng.normal(0, 2, (1, n_channels))
        out.append((rng.normal(500, 5, (L, n_channels)) + trend).astype(np.float32))
    return out


def verify_one(model_name: str, device: str, n_channels: int = 8) -> dict:
    """Load the real backbone and run its embedding path on synthetic contexts.
    Returns a result dict; never raises (captures the traceback)."""
    cfg = Config(dataset="FD001", model_name=model_name,
                 sensor_columns=list(DEFAULT_SENSOR_COLUMNS["cmapss"])[:n_channels],
                 tsfm_context_length=200)
    # >1 window at MIXED lengths so the batched _encode_batch (CHANGES.md §46) forms
    # several shape-groups and the batch-invariance check below is meaningful.
    contexts = _synthetic_contexts(n_windows=12, n_channels=n_channels)
    res = {"model": model_name, "ok": False, "detail": ""}
    try:
        embedder = make_embedder(cfg, device=device)
        t0 = time.perf_counter()
        emb, loc_scale = embedder.embed_windows(contexts)      # default (batched) path
        dt = time.perf_counter() - t0
        # Batch-invariance: re-embed the SAME contexts one-series-at-a-time (batch_size=1,
        # the pre-§46 behaviour) and require the result to match. This is the weight-level
        # guard that the grouping/sub-chunk/scatter did not reorder or corrupt anything --
        # a mis-scatter would grossly change the vectors, far beyond fp16 kernel noise.
        embedder.batch_size = 1
        emb1, _ = embedder.embed_windows(contexts)
        n = len(contexts)
        checks = {
            "emb.ndim == 2": emb.ndim == 2,
            "emb rows == n_windows": emb.shape[0] == n,
            "loc_scale (N, V, 2)": loc_scale.ndim == 3 and loc_scale.shape[0] == n
                                   and loc_scale.shape[2] == 2,
            "emb finite": bool(np.isfinite(emb).all()),
            "loc_scale finite": bool(np.isfinite(loc_scale).all()),
            "emb non-degenerate (std>0)": float(np.std(emb)) > 0.0,
            "batch-invariant (bs=1)": bool(
                np.allclose(emb, emb1, rtol=1e-2, atol=1e-2)),
        }
        res["ok"] = all(checks.values())
        res["detail"] = (f"emb={emb.shape} loc_scale={loc_scale.shape} "
                         f"F={emb.shape[1]} {dt:.1f}s | "
                         + ", ".join(f"{k}={'Y' if v else 'N'}" for k, v in checks.items()))
    except Exception:
        res["detail"] = "EXCEPTION\n" + traceback.format_exc()
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="*", default=sorted(EMBEDDERS),
                    help="model_name keys to verify (default: all registered)")
    ap.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    args = ap.parse_args()

    try:
        import torch
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        device = args.device or "cpu"
    print(f"Device: {device}\nModels: {args.models}\n" + "=" * 72)

    results = [verify_one(m, device) for m in args.models]
    print("\n" + "=" * 72 + "\nSUMMARY")
    for r in results:
        print(f"  [{'PASS' if r['ok'] else 'FAIL'}] {r['model']}: {r['detail']}")
    n_fail = sum(not r["ok"] for r in results)
    print(f"\n{len(results) - n_fail}/{len(results)} backbones passed.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
