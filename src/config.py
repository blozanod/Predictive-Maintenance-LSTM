"""Single resolved configuration for the whole pipeline.

RESEARCH-RIGOR CONTRACT (Task 2):
  * Every design decision that affects results is a field here, never a hardcoded
    constant buried in a module.
  * Each such field carries a comment citing its source, OR is marked
    ``# DECISION (uncited):`` so every judgment call is grep-able:
        grep -rn "DECISION (uncited)" src/ tests/ notebooks/
  * ``embedding_cache_key`` hashes exactly the fields that change the cached
    embeddings, so Stage A (the GPU pass) is idempotent and Stage B never
    re-embeds.

Nothing here reads data or imports heavy libraries, so it is safe to import
anywhere (including CPU-only smoke tests).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# C-MAPSS column schema (Saxena et al. 2008; see CMAPSSData/readme.txt).
# 26 whitespace-separated columns: unit, cycle, 3 operating settings, 21 sensors.
# ---------------------------------------------------------------------------
INDEX_COLUMNS = ["unit_number", "time_cycles"]
SETTING_COLUMNS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLUMNS = [f"s_{i}" for i in range(1, 22)]
ALL_COLUMNS = INDEX_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS

# Bump this whenever the on-disk cache LAYOUT or its semantics change, so stale
# caches (e.g. pre-loc/scale, per-window-normalized embeddings) are invalidated
# by a new ``embedding_cache_key``. v2 = stores per-window loc/scale + variable-
# length TSFM context embeddings + fp16 embedding storage (Task 1/2 fixes).
CACHE_SCHEMA_VERSION = 2

# Head-feature composition options (Task 1.1). ``emb`` is the pooled embedding
# only; ``emb+locscale`` appends the per-window Chronos-2 instance-norm loc/scale
# (the degradation-level signal the internal normalization otherwise discards);
# ``emb+locscale+raw`` additionally appends the last-cycle raw sensors (Wide &
# Deep-lite, mirrors the PHM 10.32 paper, RESEARCH_PLAN sec.1).
HEAD_FEATURE_CHOICES = ("emb", "emb+locscale", "emb+locscale+raw")

# Pooling of embed()'s (n_variates, num_patches+2, d_model) output. The last two
# positions are special tokens appended by embed(): index -1 is the masked
# output/forecast patch, index -2 is the REG token; content patches are [:-2].
POOLING_CHOICES = ("forecast_token", "last_content", "mean", "flatten")

# The 14 non-constant FD001 sensors. Sensors 1,5,6,10,16,18,19 are flat (zero
# variance) under FD001's single operating condition and are dropped by
# convention. This is a fixed, dataset-level, a-priori list (a property of the
# sensor set, NOT fit on any data split), so using it introduces no train/val/
# test leakage and keeps embeddings cacheable in one pass.
# Convention: Li et al. 2018 (arXiv:1806.09347), Heimes 2008.
FD001_NONCONSTANT_SENSORS = [
    "s_2", "s_3", "s_4", "s_7", "s_8", "s_9",
    "s_11", "s_12", "s_13", "s_14", "s_15", "s_17", "s_20", "s_21",
]


@dataclass
class Config:
    """Resolved configuration. Override fields via ``dataclasses.replace`` or the
    ``override`` helper; never mutate module-level constants."""

    # ---- reproducibility ---------------------------------------------------
    seed: int = 42  # base seed; threaded through numpy/torch/dataloaders (Task 2.3)
    deterministic: bool = True  # torch deterministic algorithms where feasible (Task 2.3)

    # ---- dataset -----------------------------------------------------------
    dataset: str = "FD001"  # Phase 1 scope is FD001; FD002-004 slot in later (Task 2.6)
    data_dir: str = "CMAPSSData"

    # ---- RUL labels --------------------------------------------------------
    # Piecewise-linear RUL: clip at a constant beyond which degradation is not yet
    # observable. 125 is community convention (Heimes 2008; Li et al. 2018).
    max_rul: int = 125
    # NOTE: test-label clipping is no longer a toggle. evaluate.py ALWAYS reports
    # BOTH protocols: labels clipped at max_rul (the literature-comparable numbers)
    # and unclipped (the raw RUL_FDxxx.txt PHM08 target). See CHANGES.md sec.5.
    # Training labels are always clipped at max_rul.

    # ---- windowing ---------------------------------------------------------
    window_size: int = 30  # baseline sliding-window length in cycles; community convention (Li et al. 2018)
    # Sensor channels fed to every model. Fixed a-priori list => no leakage (see
    # FD001_NONCONSTANT_SENSORS above). Use ALL_COLUMNS[2:] for the full 24.
    sensor_columns: list = field(default_factory=lambda: list(FD001_NONCONSTANT_SENSORS))
    # Left-pad FIXED test windows (baselines) shorter than window_size by repeating
    # the first cycle. The TSFM path does NOT use this: it feeds embed()'s native
    # variable-length input so short test histories are left-pad-MASKED internally
    # (Task 1.2), avoiding fabricated cycles that corrupt instance-norm statistics.
    pad_short_test_units: bool = True

    # ---- splits ------------------------------------------------------------
    # Unit-level validation split fraction (splits are BY UNIT, never by row, so
    # no unit's windows cross a split -- Task 2.4).
    val_fraction: float = 0.2
    # Data-efficiency sweep grid expressed as ENGINE-UNIT COUNTS, not row
    # fractions (RESEARCH_PLAN.md sec.6). FD001 has 100 train units.
    data_unit_counts: list = field(default_factory=lambda: [2, 5, 10, 25, 50, 100])
    # Seeds per sweep cell (>=5 recommended, RESEARCH_PLAN.md sec.6).
    sweep_seeds: list = field(default_factory=lambda: [0, 1, 2, 3, 4])

    # ---- embeddings (frozen TSFM) -----------------------------------------
    # Model-name string so other TSFMs (MOMENT/TimesFM/TTM) slot in later (Task 2.6).
    model_name: str = "amazon/chronos-2"  # anchor TSFM (Chronos-2, arXiv:2510.15821)
    # Pooling of embed()'s (n_variates, num_patches+2, d_model) output into one
    # window feature vector (POOLING_CHOICES). forecast_token (index -1, the masked
    # output patch) is a defensible CLS-like default (Task 1.3). Part of the cache
    # key -- each pooling is cached independently.
    pooling: str = "forecast_token"  # DECISION (uncited): CLS-like default; ablated in run_ablation
    embed_batch_size: int = 256  # embed() batch size; lower for a T4 (Stage A note)
    embed_dtype: str = "bfloat16"  # fp16/bf16 for GPU embed compute; degrades to a T4 (Stage A note)
    # How much history (in cycles) the TSFM sees, INDEPENDENT of the baseline
    # window_size (Task 1.2). None => use window_size. The TSFM path feeds variable-
    # length contexts capped at this length; short test units are shorter, not padded.
    tsfm_context_length: Optional[int] = None
    # On-disk storage dtype for POOLED embeddings. float16 halves Drive I/O; the raw
    # windows and loc/scale stay float32. DECISION (uncited): measure & record the
    # full-data RMSE effect (expected negligible; revert to float32 if not) -- see
    # CHANGES.md. Compute dtype during embed() is embed_dtype (bf16), independent.
    embedding_storage_dtype: str = "float16"
    cache_compressed: bool = False  # uncompressed .npz: much faster save on ~GB float16 (Task 2)
    # Head-feature composition (HEAD_FEATURE_CHOICES). Selects, at Stage B, which
    # cached signals feed the head; does NOT change the embedding cache (Task 1.1).
    head_features: str = "emb"

    # ---- MLP regression head ----------------------------------------------
    # 2-layer MLP, hidden 256, dropout -- mirrors arXiv:2606.11990 (their ablation:
    # linear < 2-layer ~ 4-layer). Set num_layers=1 for the linear-head ablation.
    head_hidden_dim: int = 256
    head_dropout: float = 0.1  # DECISION (uncited): standard light regularization for the head
    head_num_layers: int = 2

    # ---- losses ------------------------------------------------------------
    # Phase-1 loss arms. "quantile" is the optional third arm (RESEARCH_PLAN sec.5).
    losses: list = field(default_factory=lambda: ["mse", "corn"])
    # Ordinal binning for CORN: K ordered bins over [0, max_rul]. K=25 => width 5
    # cycles after clipping at 125 (RESEARCH_PLAN sec.5). CORN: Shi, Cao & Raschka
    # arXiv:2111.08851, impl coral-pytorch.
    num_bins: int = 25
    # CORN decoding: expected value over bin probabilities (vs. argmax). Ablate
    # per RESEARCH_PLAN sec.11 (risks). "expected_value" | "argmax".
    corn_decoding: str = "expected_value"
    # Pinball/quantile levels for the optional quantile arm (RESEARCH_PLAN sec.5).
    quantile_levels: list = field(default_factory=lambda: [0.1, 0.5, 0.9])

    # ---- head training -----------------------------------------------------
    head_lr: float = 1e-3  # DECISION (uncited): Adam default-ish LR for the small head
    head_weight_decay: float = 1e-4  # DECISION (uncited): light L2 on the head
    head_batch_size: int = 256
    head_max_epochs: int = 100
    head_early_stopping_patience: int = 10  # early stop on val (Task 1 train.py)
    # Scale regression targets to [0,1] by dividing by max_rul during training;
    # decode back on predict. DECISION (uncited): standard target scaling for
    # stable MLP regression. Does not affect CORN (which uses integer bins).
    scale_targets: bool = True

    # ---- from-scratch baseline training -----------------------------------
    baseline_max_epochs: int = 100
    baseline_early_stopping_patience: int = 10
    baseline_lr: float = 1e-3  # DECISION (uncited): Adam LR for CNN/LSTM baselines
    baseline_batch_size: int = 256
    # Per-baseline window length override (name -> cycles). Empty => every baseline
    # uses window_size. Equal-tuning-budget fairness (RESEARCH_PLAN sec.6): if a
    # longer window (e.g. 120) improves GBM/LSTM at full data, set it here so the
    # sweep windows the raw series for that baseline (Task 1.5). Other baselines and
    # the cached fixed windows are unaffected.
    baseline_windows: dict = field(default_factory=dict)

    # ---- paths -------------------------------------------------------------
    cache_dir: str = "cache"      # embedding + window caches (Stage A output)
    results_dir: str = "results"  # metrics CSVs, run metadata, sampled unit IDs

    # -- validation ----------------------------------------------------------
    def __post_init__(self):
        if self.pooling not in POOLING_CHOICES:
            raise ValueError(f"pooling must be one of {POOLING_CHOICES}, got {self.pooling!r}")
        if self.head_features not in HEAD_FEATURE_CHOICES:
            raise ValueError(
                f"head_features must be one of {HEAD_FEATURE_CHOICES}, got {self.head_features!r}"
            )

    # -- helpers -------------------------------------------------------------
    def replace(self, **kwargs) -> "Config":
        """Return a copy with fields overridden (validates unknown keys)."""
        known = {f.name for f in dataclasses.fields(self)}
        unknown = set(kwargs) - known
        if unknown:
            raise KeyError(f"Unknown config field(s): {sorted(unknown)}")
        return dataclasses.replace(self, **kwargs)

    def to_dict(self) -> dict:
        return asdict(self)

    def num_channels(self) -> int:
        return len(self.sensor_columns)

    def effective_tsfm_context(self) -> int:
        """History length (cycles) the TSFM sees. Defaults to the baseline window."""
        return self.tsfm_context_length if self.tsfm_context_length is not None else self.window_size

    # ---- cache keys --------------------------------------------------------
    def _window_key_fields(self) -> dict:
        """Fields that determine the RAW cached FIXED windows (model-independent;
        baselines + raw-fusion last-cycle sensors read these)."""
        return {
            "dataset": self.dataset,
            "window_size": self.window_size,
            "sensor_columns": list(self.sensor_columns),
            "max_rul": self.max_rul,
            "pad_short_test_units": self.pad_short_test_units,
        }

    def _embedding_key_fields(self) -> dict:
        """Fields that determine the cached EMBEDDINGS (Stage A key): the fixed-
        window fields (for the co-cached raw windows), plus the TSFM axes that
        change the embeddings -- model, pooling, and the variable-length context
        length -- and the cache SCHEMA VERSION so old caches invalidate (Task 1.1).

        NOTE: head_features is NOT here -- loc/scale and raw-last are always cached;
        head_features only selects which are USED at Stage B (Task 1.1)."""
        d = self._window_key_fields()
        d.update({
            "model_name": self.model_name,
            "pooling": self.pooling,
            "tsfm_context_length": self.effective_tsfm_context(),
            "cache_schema_version": CACHE_SCHEMA_VERSION,
        })
        return d

    @staticmethod
    def _hash(d: dict) -> str:
        blob = json.dumps(d, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def window_cache_key(self) -> str:
        return f"windows_{self.dataset}_{self._hash(self._window_key_fields())}"

    def embedding_cache_key(self) -> str:
        model_tag = self.model_name.split("/")[-1]
        return (
            f"emb_{self.dataset}_{model_tag}_{self.pooling}"
            f"_w{self.window_size}_c{self.effective_tsfm_context()}"
            f"_v{CACHE_SCHEMA_VERSION}_{self._hash(self._embedding_key_fields())}"
        )

    def cache_path(self) -> Path:
        return Path(self.cache_dir) / f"{self.embedding_cache_key()}.npz"


# Default resolved configuration. Import and override; do not mutate in place.
CONFIG = Config()
