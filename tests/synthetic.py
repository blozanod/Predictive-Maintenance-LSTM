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


# Discrete operating points used when n_conditions > 1 (recoverable by the
# per-column rounding in ``data.condition_keys``: 0 / 2 / 0 decimals).
_COND_SETTINGS = np.array([[0.0, 0.25, 100.0], [10.0, 0.70, 60.0],
                           [20.0, 0.84, 40.0], [35.0, 0.62, 80.0]])
# Large per-condition sensor offsets: regime switching dominates raw variance,
# burying the degradation trend until condition-wise normalization removes it.
_COND_OFFSET_SCALE = 60.0


def write_synthetic_cmapss(
    data_dir: Path,
    dataset: str = "FD001",
    n_train_units: int = 8,
    n_test_units: int = 6,
    min_cycles: int = 20,
    max_cycles: int = 45,
    seed: int = 0,
    n_conditions: int = 1,
) -> None:
    """Write train_{ds}.txt, test_{ds}.txt, RUL_{ds}.txt into ``data_dir``.

    Sensors are a degradation trend + noise so there is learnable signal; a couple
    of test units are deliberately short to exercise left-padding. With
    ``n_conditions > 1`` each cycle randomly draws one of ``_COND_SETTINGS`` and a
    large condition-dependent sensor offset (the FD002/FD004 shape).
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cond_offsets = rng.normal(0, _COND_OFFSET_SCALE, size=(max(n_conditions, 1), 21))

    def unit_rows(unit_id, n_cycles):
        rows = np.zeros((n_cycles, len(ALL_COLUMNS)), dtype=np.float64)
        rows[:, 0] = unit_id
        rows[:, 1] = np.arange(1, n_cycles + 1)
        frac = np.linspace(0, 1, n_cycles)[:, None]
        base = rng.normal(500, 5, size=(1, 21))
        trend = rng.normal(0, 20, size=(1, 21)) * frac  # degradation trend
        noise = rng.normal(0, 1, size=(n_cycles, 21))
        rows[:, 5:26] = base + trend + noise
        if n_conditions <= 1:
            rows[:, 2:5] = rng.normal(0, 0.001, size=(n_cycles, 3))  # settings ~const
        else:
            cond = rng.integers(0, n_conditions, size=n_cycles)
            # settings wobble stays below condition_keys' rounding resolution
            rows[:, 2:5] = (_COND_SETTINGS[cond]
                            + rng.normal(0, [0.05, 0.001, 0.05], size=(n_cycles, 3)))
            rows[:, 5:26] += cond_offsets[cond]
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
    """Deterministic random-projection stand-in for a frozen-TSFM embedder.

    Same interface as ``ChronosEmbedder``
    (``embed_windows(contexts) -> (emb (N, F) float32, loc_scale (N, C, 2) float32)`` /
    ``describe``), CPU-only, no downloads. Accepts variable-length contexts (a list of
    ``(L_i, C)`` arrays) or a fixed ``(N, W, C)`` array, mirroring embed()'s native
    input. ``loc_scale`` is the per-channel mean/std of each context (shape ``(N, C, 2)``),
    standing in for Chronos-2's instance-norm loc/scale. Call count is tracked so tests
    can assert Stage A idempotency and that sweeps never re-embed.

    Parametrized (IMPLEMENTATION_PLAN M0.3) so ONE mock can stand in for either backbone
    FAMILY the v2 build adds, distinguished by the shape of the pooled feature vector:

    * ``layout="multivariate"`` (default -- Chronos-2 / Moirai / TTM-like): the backbone
      embeds all channels JOINTLY (and appends REG + forecast special tokens); one pooled
      summary of width ``feature_dim`` (= the mock's ``d_model``) represents the window, so
      ``F`` does NOT grow with the channel count.
    * ``layout="univariate"`` (MOMENT / TimesFM-like): the backbone embeds EACH channel
      independently through a shared 1-D projection (no special tokens); the per-variate
      vectors are combined by ``channel_aggregation``.

    ``channel_aggregation`` is the RQ-M fairness knob (M1's ``config.channel_aggregation``):
    ``"concat"`` keeps per-variate detail, ``"mean"`` collapses the variate axis to a common
    representation. It applies to BOTH layouts, exactly as the real knob applies uniformly
    across every model. Resulting feature width ``F``:

    | layout        | concat            | mean        |
    |---------------|-------------------|-------------|
    | multivariate  | ``feature_dim``   | ``feature_dim`` |
    | univariate    | ``C * feature_dim`` | ``feature_dim`` |

    (The multivariate mock's joint summary is already channel-collapsed, so its two
    aggregation modes coincide -- a deliberate mock simplification; the univariate mock is
    where ``concat`` vs ``mean`` genuinely changes the output dim, which is the property M1
    tests exercise.)

    Backwards-compatibility: with the defaults (``layout="multivariate"``,
    ``channel_aggregation="concat"``) ``MockEmbedder(feature_dim=F)`` reproduces the
    original fixture byte-for-byte -- ``F == feature_dim`` independent of channel count --
    so every pre-M0 test stays green.
    """

    def __init__(self, feature_dim: int = 32, seed: int = 0,
                 layout: str = "multivariate", channel_aggregation: str = "concat"):
        if layout not in ("multivariate", "univariate"):
            raise ValueError(f"layout must be 'multivariate' or 'univariate', got {layout!r}")
        if channel_aggregation not in ("concat", "mean"):
            raise ValueError("channel_aggregation must be 'concat' or 'mean', got "
                             f"{channel_aggregation!r}")
        self.feature_dim = feature_dim
        self.seed = seed
        self.layout = layout
        self.channel_aggregation = channel_aggregation
        self.n_calls = 0
        self._proj = None
        self.last_throughput = None

    def _ensure_proj(self, n_channels: int) -> None:
        # multivariate: one (C, d_model) projection of the joint channel-mean vector.
        # univariate:   one (1, d_model) projection applied to each channel independently.
        want = n_channels if self.layout == "multivariate" else 1
        if self._proj is None or self._proj.shape[0] != want:
            rng = np.random.default_rng(self.seed)
            self._proj = rng.normal(0, 1, size=(want, self.feature_dim)).astype(np.float32)

    def _pool(self, mean_t: np.ndarray) -> np.ndarray:
        """Pooled feature vector for one window from its per-channel mean ``(C,)``."""
        if self.layout == "multivariate":
            # Joint summary already fuses every channel -> width feature_dim; the two
            # channel_aggregation modes coincide (see class docstring). The concat/default
            # path is byte-identical to the original fixture.
            return np.tanh(mean_t @ self._proj)                       # (feature_dim,)
        # univariate: embed each channel through the shared (1, feature_dim) projection.
        per_variate = np.tanh(mean_t[:, None] @ self._proj)           # (C, feature_dim)
        if self.channel_aggregation == "concat":
            return per_variate.reshape(-1)                            # (C * feature_dim,)
        return per_variate.mean(axis=0)                               # (feature_dim,)

    def embed_windows(self, contexts) -> tuple[np.ndarray, np.ndarray]:
        self.n_calls += 1
        if isinstance(contexts, np.ndarray) and contexts.ndim == 3:
            ctx = [contexts[i] for i in range(contexts.shape[0])]
        else:
            ctx = list(contexts)
        if not ctx:
            return np.empty((0, self.feature_dim), np.float32), np.empty((0, 0, 2), np.float32)
        n_channels = ctx[0].shape[1]
        self._ensure_proj(n_channels)
        feats, loc_scale = [], []
        for w in ctx:                        # w: (L_i, C), variable L_i
            w = np.asarray(w, np.float32)
            mean_t = w.mean(axis=0)          # (C,)
            std_t = w.std(axis=0)            # (C,)
            feats.append(self._pool(mean_t))                      # (F,)
            loc_scale.append(np.stack([mean_t, std_t], axis=-1))  # (C, 2)
        return (np.asarray(feats, np.float32), np.asarray(loc_scale, np.float32))

    def describe(self) -> dict:
        return {"embedder": "MockEmbedder", "feature_dim": self.feature_dim,
                "seed": self.seed, "layout": self.layout,
                "channel_aggregation": self.channel_aggregation}


def write_synthetic_ncmapss(
    data_dir: Path,
    dataset: str = "DS02",
    n_dev_units: int = 3,
    n_test_units: int = 2,
    min_cycles: int = 10,
    max_cycles: int = 16,
    min_rows: int = 15,
    max_rows: int = 25,
    seed: int = 0,
    suffix: str = "-000",
    rename_sensor: str | None = None,
) -> Path:
    """Write a miniature ``N-CMAPSS_<dataset><suffix>.h5`` matching the real key set.

    Structure mirrors src/datasets/ncmapss.py's expectations: ``W_*`` (4 flight-
    condition channels), ``X_s_*`` (14 measured sensors), ``A_*`` (unit, cycle, Fc, hs),
    and byte-string ``W_var/X_s_var/A_var``. ``X_v_*``/``T_*``/``Y_*`` are written
    full-length (random) so the loader is proven to IGNORE the oracle channels even
    though they are present. Each unit's channels drift toward failure so heads and
    baselines have learnable signal. Dev/test unit ids are disjoint. ``rename_sensor``
    swaps one X_s var name to exercise the fail-loud schema check. Returns the path.
    """
    import h5py

    from src.config import NCMAPSS_W_VARS, NCMAPSS_XS_VARS

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    w_vars = list(NCMAPSS_W_VARS)
    xs_vars = list(NCMAPSS_XS_VARS)
    if rename_sensor is not None:
        xs_vars[0] = rename_sensor

    def _split(unit_ids, flight_classes):
        W, Xs, A = [], [], []
        for u, fc in zip(unit_ids, flight_classes):
            n_cycles = int(rng.integers(min_cycles, max_cycles + 1))
            for c in range(1, n_cycles + 1):
                rows = int(rng.integers(min_rows, max_rows + 1))
                frac = c / n_cycles                       # 0..1 degradation progress
                w = rng.normal(0.0, 1.0, size=(rows, len(w_vars)))
                base = rng.normal(500.0, 5.0, size=(1, len(xs_vars)))
                trend = rng.normal(0.0, 20.0, size=(1, len(xs_vars))) * frac
                xs = base + trend + rng.normal(0.0, 1.0, size=(rows, len(xs_vars)))
                W.append(w.astype(np.float32))
                Xs.append(xs.astype(np.float32))
                a = np.zeros((rows, 4), np.float32)
                a[:, 0] = u          # unit
                a[:, 1] = c          # cycle
                a[:, 2] = fc         # Fc (flight class, constant per unit)
                a[:, 3] = 1.0 - frac  # hs (health state, decreasing)
                A.append(a)
        return (np.concatenate(W), np.concatenate(Xs), np.concatenate(A))

    dev_units = list(range(1, n_dev_units + 1))
    test_units = list(range(100, 100 + n_test_units))   # disjoint from dev
    dev_fc = [1 + (i % 3) for i in range(n_dev_units)]
    test_fc = [1 + (i % 3) for i in range(n_test_units)]
    W_dev, Xs_dev, A_dev = _split(dev_units, dev_fc)
    W_test, Xs_test, A_test = _split(test_units, test_fc)

    def _bytes(names):
        return np.array([n.encode() for n in names]).reshape(-1, 1)

    path = data_dir / f"N-CMAPSS_{dataset}{suffix}.h5"
    with h5py.File(path, "w") as h:
        for tag, (Wd, Xd, Ad) in (("dev", (W_dev, Xs_dev, A_dev)),
                                  ("test", (W_test, Xs_test, A_test))):
            h.create_dataset(f"W_{tag}", data=Wd)
            h.create_dataset(f"X_s_{tag}", data=Xd)
            h.create_dataset(f"A_{tag}", data=Ad)
            # Oracle channels the loader must NOT read (present, full-length, random).
            h.create_dataset(f"X_v_{tag}", data=rng.normal(size=(Wd.shape[0], 5)).astype(np.float32))
            h.create_dataset(f"T_{tag}", data=rng.normal(size=(Wd.shape[0], 10)).astype(np.float32))
            h.create_dataset(f"Y_{tag}", data=rng.normal(size=(Wd.shape[0], 1)).astype(np.float32))
        h.create_dataset("W_var", data=_bytes(w_vars))
        h.create_dataset("X_s_var", data=_bytes(xs_vars))
        h.create_dataset("A_var", data=_bytes(["unit", "cycle", "Fc", "hs"]))
    return path


def write_synthetic_xjtu(
    root: Path,
    bearings_per_condition: int = 5,
    min_snapshots: int = 18,
    max_snapshots: int = 40,
    samples_per_snapshot: int = 256,
    seed: int = 0,
) -> None:
    """Write a miniature XJTU-SY directory tree (src/datasets/xjtu.py layout): 3
    condition folders x N ``BearingC_B`` folders x one 2-column CSV per minute.
    Vibration amplitude grows toward failure so the indicators carry RUL signal."""
    from src.datasets.xjtu import XJTU_CONDITIONS

    root = Path(root)
    rng = np.random.default_rng(seed)
    for cond_name, (cond_idx, _, _) in XJTU_CONDITIONS.items():
        for b in range(1, bearings_per_condition + 1):
            bdir = root / cond_name / f"Bearing{cond_idx + 1}_{b}"
            bdir.mkdir(parents=True, exist_ok=True)
            n_snap = int(rng.integers(min_snapshots, max_snapshots + 1))
            for i in range(1, n_snap + 1):
                frac = i / n_snap
                amp = 0.5 + 3.0 * frac ** 2          # degradation: growing energy
                x = rng.normal(0, amp, size=(samples_per_snapshot, 2))
                if frac > 0.7:                        # late-life impulsiveness
                    spikes = rng.integers(0, samples_per_snapshot, size=5)
                    x[spikes] += rng.normal(0, 6 * amp, size=(5, 2))
                header = "Horizontal_vibration_signals,Vertical_vibration_signals"
                np.savetxt(bdir / f"{i}.csv", x, delimiter=",", header=header,
                           comments="", fmt="%.5f")
