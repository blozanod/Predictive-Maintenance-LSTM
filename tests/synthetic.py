"""Tiny synthetic C-MAPSS generator + a mock embedder.

Lets the CPU smoke tests exercise the real loading -> windowing -> cache -> sweep
path WITHOUT a GPU and WITHOUT downloading C-MAPSS (Task 2.7). The generated files
match the 26-column C-MAPSS text schema so ``src.data.load_cmapss`` reads them
unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import ALL_COLUMNS


def write_synthetic_cmapss(
    data_dir: Path,
    dataset: str = "FD001",
    n_train_units: int = 8,
    n_test_units: int = 6,
    min_cycles: int = 20,
    max_cycles: int = 45,
    seed: int = 0,
) -> None:
    """Write train_{ds}.txt, test_{ds}.txt, RUL_{ds}.txt into ``data_dir``.

    Sensors are a degradation trend + noise so there is learnable signal; a couple
    of test units are deliberately short to exercise left-padding.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    def unit_rows(unit_id, n_cycles):
        rows = np.zeros((n_cycles, len(ALL_COLUMNS)), dtype=np.float64)
        rows[:, 0] = unit_id
        rows[:, 1] = np.arange(1, n_cycles + 1)
        rows[:, 2:5] = rng.normal(0, 0.001, size=(n_cycles, 3))  # settings ~const
        frac = np.linspace(0, 1, n_cycles)[:, None]
        base = rng.normal(500, 5, size=(1, 21))
        trend = rng.normal(0, 20, size=(1, 21)) * frac  # degradation trend
        noise = rng.normal(0, 1, size=(n_cycles, 21))
        rows[:, 5:26] = base + trend + noise
        return rows

    # ---- train: full run-to-failure ----
    train_blocks = []
    for u in range(1, n_train_units + 1):
        n = int(rng.integers(min_cycles, max_cycles))
        train_blocks.append(unit_rows(u, n))
    _write(data_dir / f"train_{dataset}.txt", np.vstack(train_blocks))

    # ---- test: truncated before failure + provided RUL ----
    test_blocks, rul = [], []
    for u in range(1, n_test_units + 1):
        # a couple of short units (< typical window) to test padding
        n = 5 if u <= 2 else int(rng.integers(min_cycles, max_cycles))
        test_blocks.append(unit_rows(u, n))
        rul.append(int(rng.integers(5, 90)))
    _write(data_dir / f"test_{dataset}.txt", np.vstack(test_blocks))
    (data_dir / f"RUL_{dataset}.txt").write_text("\n".join(str(r) for r in rul) + "\n")


def _write(path: Path, arr: np.ndarray) -> None:
    lines = []
    for row in arr:
        vals = [str(int(row[0])), str(int(row[1]))] + [f"{v:.4f}" for v in row[2:]]
        lines.append(" ".join(vals))
    path.write_text("\n".join(lines) + "\n")


class MockEmbedder:
    """Deterministic random-projection stand-in for ``ChronosEmbedder``.

    Same interface (``embed_windows(contexts) -> (emb, loc_scale)`` / ``describe``),
    CPU-only, no downloads. Accepts variable-length contexts (a list of ``(L_i, C)``
    arrays) or a fixed ``(N, W, C)`` array, mirroring embed()'s native input. The
    ``loc_scale`` return is the per-channel mean/std of each context (shape
    ``(N, C, 2)``), standing in for Chronos-2's instance-norm loc/scale. Counts calls
    so tests can assert Stage A idempotency and that sweeps never re-embed.
    """

    def __init__(self, feature_dim: int = 32, seed: int = 0):
        self.feature_dim = feature_dim
        self.seed = seed
        self.n_calls = 0
        self._proj = None
        self.last_throughput = None

    def embed_windows(self, contexts) -> tuple[np.ndarray, np.ndarray]:
        self.n_calls += 1
        if isinstance(contexts, np.ndarray) and contexts.ndim == 3:
            ctx = [contexts[i] for i in range(contexts.shape[0])]
        else:
            ctx = list(contexts)
        if not ctx:
            return np.empty((0, self.feature_dim), np.float32), np.empty((0, 0, 2), np.float32)
        n_channels = ctx[0].shape[1]
        if self._proj is None or self._proj.shape[0] != n_channels:
            rng = np.random.default_rng(self.seed)
            self._proj = rng.normal(0, 1, size=(n_channels, self.feature_dim)).astype(np.float32)
        feats, loc_scale = [], []
        for w in ctx:                        # w: (L_i, C), variable L_i
            w = np.asarray(w, np.float32)
            mean_t = w.mean(axis=0)          # (C,)
            std_t = w.std(axis=0)            # (C,)
            feats.append(np.tanh(mean_t @ self._proj))            # (feature_dim,)
            loc_scale.append(np.stack([mean_t, std_t], axis=-1))  # (C, 2)
        return (np.asarray(feats, np.float32), np.asarray(loc_scale, np.float32))

    def describe(self) -> dict:
        return {"embedder": "MockEmbedder", "feature_dim": self.feature_dim, "seed": self.seed}
