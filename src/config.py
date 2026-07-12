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
    # Evaluate against the UNCLIPPED ground-truth test RUL (the original PHM08
    # competition target). Set True to also clip test labels at max_rul, which
    # some piecewise-linear papers do. DECISION (uncited): default False for
    # comparability with the raw RUL_FDxxx.txt targets.
    clip_test_labels: bool = False

    # ---- windowing ---------------------------------------------------------
    window_size: int = 30  # sliding-window length in cycles; community convention (Li et al. 2018)
    # Sensor channels fed to every model. Fixed a-priori list => no leakage (see
    # FD001_NONCONSTANT_SENSORS above). Use ALL_COLUMNS[2:] for the full 24.
    sensor_columns: list = field(default_factory=lambda: list(FD001_NONCONSTANT_SENSORS))
    # Left-pad test units shorter than window_size by repeating the first cycle,
    # so every test unit yields exactly one last-cycle window. Community practice
    # for the C-MAPSS last-cycle test protocol (Li et al. 2018).
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
    # window feature vector. Ablate: last_patch | mean | flatten (RESEARCH_PLAN
    # sec.1; the PHM paper flattens). Materially affects results, hence a config
    # option and part of the cache key.
    pooling: str = "last_patch"
    embed_batch_size: int = 256  # embed() batch size; lower for a T4 (Stage A note)
    embed_dtype: str = "bfloat16"  # fp16/bf16 for GPU embed; degrades to a T4 (Stage A note)
    context_length: Optional[int] = None  # None => use full window as context (embed default)

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

    # ---- paths -------------------------------------------------------------
    cache_dir: str = "cache"      # embedding + window caches (Stage A output)
    results_dir: str = "results"  # metrics CSVs, run metadata, sampled unit IDs

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

    # ---- cache keys --------------------------------------------------------
    def _window_key_fields(self) -> dict:
        """Fields that determine the RAW cached windows (model-independent)."""
        return {
            "dataset": self.dataset,
            "window_size": self.window_size,
            "sensor_columns": list(self.sensor_columns),
            "max_rul": self.max_rul,
            "clip_test_labels": self.clip_test_labels,
            "pad_short_test_units": self.pad_short_test_units,
        }

    def _embedding_key_fields(self) -> dict:
        """Fields that determine the cached EMBEDDINGS (Task 3 / Stage A key:
        window size, pooling, model name -- plus the raw-window fields, since
        different windows produce different embeddings)."""
        d = self._window_key_fields()
        d.update({"model_name": self.model_name, "pooling": self.pooling})
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
            f"_w{self.window_size}_{self._hash(self._embedding_key_fields())}"
        )

    def cache_path(self) -> Path:
        return Path(self.cache_dir) / f"{self.embedding_cache_key()}.npz"


# Default resolved configuration. Import and override; do not mutate in place.
CONFIG = Config()
