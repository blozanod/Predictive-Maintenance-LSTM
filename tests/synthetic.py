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
    """Deterministic fixed random-projection stand-in for ``ChronosEmbedder``.

    Same interface (``embed_windows`` / ``describe``), CPU-only, no downloads.
    Counts calls so tests can assert Stage A idempotency and that sweeps never
    re-embed.
    """

    def __init__(self, feature_dim: int = 32, seed: int = 0):
        self.feature_dim = feature_dim
        self.seed = seed
        self.n_calls = 0
        self._proj = None

    def embed_windows(self, windows: np.ndarray) -> np.ndarray:
        self.n_calls += 1
        N = windows.shape[0]
        flat = windows.reshape(N, -1).astype(np.float32)
        if self._proj is None or self._proj.shape[0] != flat.shape[1]:
            rng = np.random.default_rng(self.seed)
            self._proj = rng.normal(0, 1, size=(flat.shape[1], self.feature_dim)).astype(np.float32)
        return np.tanh(flat @ self._proj).astype(np.float32)

    def describe(self) -> dict:
        return {"embedder": "MockEmbedder", "feature_dim": self.feature_dim, "seed": self.seed}
