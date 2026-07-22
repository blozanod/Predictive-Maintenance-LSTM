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
import re
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

# How the pooled PER-VARIATE embeddings collapse into the head feature vector
# (RQ-M fairness knob, IMPLEMENTATION_PLAN §4.1). "concat" preserves per-channel
# detail (F = n_variates * d_model) -- how a practitioner uses each model and the
# byte-identical historical Chronos-2 behavior; "mean" collapses the variate axis
# (F = d_model) for the cross-TSFM common-representation fairness control. Applied
# uniformly to ALL five models so the control is genuinely common.
CHANNEL_AGGREGATION_CHOICES = ("concat", "mean")

# Perturbative noise-injection kinds (sim-only, RQ-H; IMPLEMENTATION_PLAN §4.4).
# gaussian = additive white noise at a target SNR; drift = slow per-channel bias
# ramp; dropout = random per-row channel blanking. Applied only to SIMULATED
# datasets (C-MAPSS/N-CMAPSS) -- real readings are never perturbed (guarded loud).
NOISE_INJECTION_KINDS = ("gaussian", "drift", "dropout")

# Dataset families whose sensor readings are SIMULATED and may therefore be
# perturbed by noise_injection (RESEARCH_PLAN §1: controlled noise makes an
# unrealistically-clean simulated signal MORE lifelike). Every other family is a
# REAL measurement and perturbation is out of scope by design (fail loud).
SIMULATED_DATASET_KINDS = ("cmapss", "ncmapss")

# The 14 non-constant FD001 sensors. Sensors 1,5,6,10,16,18,19 are flat (zero
# variance) under FD001's single operating condition and are dropped by
# convention. This is a fixed, dataset-level, a-priori list (a property of the
# sensor set, NOT fit on any data split), so using it introduces no train/val/
# test leakage and keeps embeddings cacheable in one pass.
# Convention: Li et al. 2018 (arXiv:1806.09347), Heimes 2008.
# The SAME list is retained for FD002/FD004 under condition-wise normalization:
# those 7 sensors are flat WITHIN each operating condition too (they only move
# with the condition itself), so after per-condition normalization they carry no
# signal -- an a-priori property of the sensor suite, not a fitted selection
# (CHANGES.md §21).
FD001_NONCONSTANT_SENSORS = [
    "s_2", "s_3", "s_4", "s_7", "s_8", "s_9",
    "s_11", "s_12", "s_13", "s_14", "s_15", "s_17", "s_20", "s_21",
]

# C-MAPSS datasets with multiple discrete operating conditions (6 combinations of
# altitude/Mach/TRA). These REQUIRE condition-wise normalization (plan §6):
# without it, regime switching dominates the sensor variance and buries the
# degradation trend. FD001/FD003 are single-condition.
MULTI_CONDITION_DATASETS = ("FD002", "FD004")

# Datasets served by the XJTU-SY bearing loader (src/xjtu.py). Run-to-failure
# vibration, 15 bearings under 3 operating conditions -- the natural extreme-
# low-data domain (plan §3). "Cycles" are 1-minute snapshots.
XJTU_DATASETS = ("XJTU-SY",)

# N-CMAPSS (NASA "Turbofan Engine Degradation Simulation Data Set 2"; Arias Chao et
# al. 2021). One .h5 per sub-dataset, 1 Hz WITHIN flights; the loader aggregates each
# flight cycle to per-cycle summary statistics so the canonical frame stays cycle-
# level like C-MAPSS (CHANGES.md §27). DSALL is the combined all-files fleet -- the
# RQ1 high-data arm (§28). Per-file names carry the DS0x id; "DS08a/c/d" are separate.
NCMAPSS_DATASETS = ("DS01", "DS02", "DS03", "DS04", "DS05", "DS06", "DS07",
                    "DS08a", "DS08c", "DSALL")

# N-CMAPSS channel schema. W = flight-condition scenario descriptors (4), X_s =
# measured sensors (14); virtual sensors X_v, health params T, and per-row RUL Y are
# simulation ORACLES and are never read (CHANGES.md §27). Names/order are asserted
# against the file's decoded *_var arrays at load time (fail loud on drift).
# DECISION (uncited): per-cycle features = mean+std of each of the 18 raw channels plus
# cycle_len_s (flight duration) -- the cycle-level indicator-trend formulation (no
# community-standard cycle-level N-CMAPSS protocol exists; CHANGES.md §27).
NCMAPSS_W_VARS = ("alt", "Mach", "TRA", "T2")
NCMAPSS_XS_VARS = ("T24", "T30", "T48", "T50", "P15", "P2", "P21", "P24",
                   "Ps30", "P40", "P50", "Nf", "Nc", "Wf")
NCMAPSS_FEATURE_COLUMNS = (
    [f"{v}_{s}" for v in NCMAPSS_W_VARS + NCMAPSS_XS_VARS for s in ("mean", "std")]
    + ["cycle_len_s"]
)

# Rounding (decimals per setting column) used to snap the 3 operational settings
# onto their discrete condition grid before grouping: altitude wobbles ~0.008
# around {0,10,20,25,35,42}K ft, Mach ~0.001 around {0..0.84}, TRA is {20..100}.
# Convention: standard condition-clustering preprocessing for FD002/FD004.
CONDITION_SETTING_DECIMALS = (0, 2, 0)

# XJTU-SY per-snapshot condition-indicator channels (h_ = horizontal axis,
# v_ = vertical). Defined here (not in datasets/xjtu.py) so the per-dataset
# sensor-column DEFAULTS below can live in config without an import cycle;
# datasets/xjtu.py re-exports it and computes the features.
XJTU_BASE_FEATURES = ("rms", "kurtosis", "skewness", "peak", "p2p",
                      "crest", "impulse", "shape")
XJTU_FEATURE_COLUMNS = [f"{ax}_{f}" for ax in ("h", "v") for f in XJTU_BASE_FEATURES]

# Default sensor channels per dataset KIND, applied when config.sensor_columns is
# left None -- switching datasets is one knob, no cryptic KeyError deep in
# preprocessing (CHANGES.md §24). Values match the previously-required explicit
# lists exactly, so resolved configs hash to the SAME cache keys as before.
DEFAULT_SENSOR_COLUMNS = {
    "cmapss": list(FD001_NONCONSTANT_SENSORS),
    "xjtu": list(XJTU_FEATURE_COLUMNS),
    "ncmapss": list(NCMAPSS_FEATURE_COLUMNS),
}


@dataclass
class Config:
    """Resolved configuration. Override fields via ``dataclasses.replace`` or the
    ``override`` helper; never mutate module-level constants."""

    # ---- reproducibility ---------------------------------------------------
    seed: int = 42  # base seed; threaded through numpy/torch/dataloaders (Task 2.3)
    deterministic: bool = True  # torch deterministic algorithms where feasible (Task 2.3)

    # ---- dataset -----------------------------------------------------------
    # C-MAPSS "FD001".."FD004", "XJTU-SY" bearings, or N-CMAPSS "DS01".."DS08d" /
    # "DSALL" (the combined fleet). The raw files live under ONE ``data_root`` folder,
    # one subdirectory per dataset family, resolved by the loader registry
    # (src/datasets/): FD00x -> ``data_root/CMAPSSData``, XJTU-SY -> ``data_root/XJTU-SY``
    # (3 condition folders; src/datasets/xjtu.py), DS0x/DSALL -> ``data_root/N-CMAPSS``
    # (.h5 per sub-dataset; src/datasets/ncmapss.py).
    dataset: str = "FD001"
    # One root housing every dataset (config.data_root/<subdir>). ``data_dir``
    # overrides this with an explicit, dataset-specific path when set (tests point it
    # straight at a synthetic folder); leave it None to use the data_root layout.
    data_root: str = "Data"
    data_dir: Optional[str] = None
    # Condition-wise normalization (plan §6): per-condition z-normalization of the
    # sensor channels, statistics fit on the TRAIN split (all units, once -- the
    # cache-economics deviation is documented in CHANGES.md §21). None => auto:
    # ON for multi-condition datasets (FD002/FD004, XJTU-SY), OFF for FD001/FD003
    # (which keeps every earlier FD001 result byte-identical). Part of the cache
    # key -- toggling it re-embeds.
    condition_norm: Optional[bool] = None

    # ---- XJTU-SY split protocol (ignored for C-MAPSS; CHANGES.md §22) --------
    # Held-out test bearings (2 of 5 per condition) and the life fraction at
    # which each test bearing's series is truncated to mimic the C-MAPSS
    # "predict at last observed cycle" protocol. DECISION (uncited): no
    # community-standard split exists for XJTU-SY; this fixed, documented choice
    # keeps the protocol deterministic and unit-disjoint.
    xjtu_test_bearings: list = field(default_factory=lambda: [
        "Bearing1_4", "Bearing1_5", "Bearing2_4", "Bearing2_5",
        "Bearing3_4", "Bearing3_5"])
    xjtu_test_truncation: float = 0.6

    # ---- N-CMAPSS split protocol (ignored for C-MAPSS/XJTU; CHANGES.md §27-28) --
    # The file's own *_test units are run-to-failure (RUL hits 0 at the last row); to
    # match the pipeline's predict-at-last-observed-cycle protocol each test unit is
    # truncated at this life fraction (same device as XJTU, §22). ncmapss-only cache-
    # key field. DECISION (uncited): 0.6 mirrors the XJTU default; no community standard.
    ncmapss_test_truncation: float = 0.6
    # DSALL member list (§28): which per-file DS0x datasets the combined fleet unions.
    # None => whatever N-CMAPSS_DS*.h5 is on disk at load time (keyed "auto" -- for
    # exploration only). Set an explicit list for reproducible runs (the campaign does,
    # §30); the loader then REQUIRES every named member and raises if one is missing.
    dsall_datasets: Optional[list] = None

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
    # FD001_NONCONSTANT_SENSORS above). None => the dataset kind's default
    # (DEFAULT_SENSOR_COLUMNS), resolved in __post_init__ so switching datasets is
    # one knob. C-MAPSS: use ALL_COLUMNS[2:] for the full 24 channels.
    sensor_columns: Optional[list] = None
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
    # Cross-model fairness knob (RQ-M, CHANGES.md §34). How the pooled per-variate
    # embeddings collapse into the head feature vector (CHANNEL_AGGREGATION_CHOICES):
    # "concat" (default; F = n_variates*d_model, the byte-identical historical path)
    # or "mean" (F = d_model, the common-representation control). Part of the
    # embedding key ONLY when != "concat", so every existing FD001 key is unchanged.
    channel_aggregation: str = "concat"

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

    # ---- scoring & the win-rule (RESEARCH_PLAN §8; CHANGES.md §36) ----------
    # These score EXISTING result CSVs (src/scoring.py); none is a cache key.
    # DECISION (uncited): min seed-mean improvement (in the primary-metric's units)
    # a TSFM must beat the strongest per-cell baseline by to be called a "win".
    win_margin: float = 0.0
    # Paired-seed significance threshold for the win test. Descriptive only: at 5
    # seeds the paired test is low-powered, so read p alongside the seed-means.
    win_alpha: float = 0.05
    # The metric the absolute-floor guard reads: a "win" where even the winner's
    # error is worse than the predict-mean floor is HOLLOW and not a success
    # condition (RESEARCH_PLAN §8). One of evaluate.METRIC_FIELDS.
    usability_floor_metric: str = "nasa_clipped"

    # ---- earliness: "too early is also bad" (RESEARCH_PLAN §8; CHANGES.md §37) --
    # Neither is a cache key -- both drive the earliness histogram / cost curve over
    # existing predictions (src/evaluate.py). Edges bin d = pred - true (the horizon
    # `bias` sign convention, CHANGES.md §16): d >= 0 is the penalized "late" side
    # (predicted more life than remains), d < 0 is "wastefully early".
    earliness_bin_edges: list = field(default_factory=lambda: [
        -50.0, -25.0, -10.0, 0.0, 10.0, 25.0, 50.0])
    # early-cost : late-cost sweep for the cost curve (values = late_cost / early_cost,
    # early_cost fixed at 1). No single arbitrary ratio -- the curve is the result.
    cost_ratios: list = field(default_factory=lambda: [1.0, 2.0, 5.0, 10.0, 20.0,
                                                       50.0, 100.0])

    # ---- interventions: sim-only noise/drift injection (RQ-H; CHANGES.md §38) ---
    # Controlled degradation of SIMULATED sensor readings to map the noise-tolerance
    # frontier (RESEARCH_PLAN §1). {} = off. Applied in data.load_prepared AFTER
    # labels, BEFORE windowing. RAISES if config.dataset is a REAL dataset
    # (XJTU/MetroPT/Hydraulic/Backblaze) -- perturbing real readings is out of scope.
    # DECISION (uncited): kinds/params, e.g. {"kind":"gaussian","snr_db":20,"seed":0}.
    # Added to the window key ONLY when non-empty (existing keys unchanged).
    noise_injection: dict = field(default_factory=dict)

    # ---- paths -------------------------------------------------------------
    cache_dir: str = "cache"      # embedding + window caches (Stage A output)
    results_dir: str = "results"  # metrics CSVs, run metadata, sampled unit IDs
    # Names every result artifact this run writes: CSVs become
    # ``<experiment_name>_<name>.csv`` and figures ``<experiment_name>_<name>.png``,
    # and the per-run bookkeeping dirs are ``<experiment_name>_runs`` etc. Set it in
    # the notebook's Config cell so experiments never clobber each other's results.
    # "" => no prefix (the historical flat layout; keeps existing files untouched).
    # NOT part of any cache key -- it names outputs only, never affects embeddings.
    experiment_name: str = ""

    # -- validation ----------------------------------------------------------
    def __post_init__(self):
        if self.pooling not in POOLING_CHOICES:
            raise ValueError(f"pooling must be one of {POOLING_CHOICES}, got {self.pooling!r}")
        if self.head_features not in HEAD_FEATURE_CHOICES:
            raise ValueError(
                f"head_features must be one of {HEAD_FEATURE_CHOICES}, got {self.head_features!r}"
            )
        if self.channel_aggregation not in CHANNEL_AGGREGATION_CHOICES:
            raise ValueError(
                f"channel_aggregation must be one of {CHANNEL_AGGREGATION_CHOICES}, "
                f"got {self.channel_aggregation!r}")
        # Typo-guard the noise kind at construction; the sim-only (real-dataset)
        # guard fires where the perturbation is APPLIED (data.load_prepared), so a
        # real-dataset config can still be built to assert the key/guard behavior.
        if self.noise_injection:
            kind = self.noise_injection.get("kind")
            if kind not in NOISE_INJECTION_KINDS:
                raise ValueError(
                    f"noise_injection['kind'] must be one of {NOISE_INJECTION_KINDS}, "
                    f"got {kind!r}")
        # experiment_name lands in every result filename -- keep it path-safe.
        if self.experiment_name and not re.fullmatch(r"[A-Za-z0-9._-]+", self.experiment_name):
            raise ValueError(
                f"experiment_name {self.experiment_name!r} must contain only letters, "
                f"digits, '.', '_', '-' (it prefixes result filenames)")
        # Resolve the dataset kind's default sensor channels (one-knob dataset
        # switching, CHANGES.md §24). replace() re-runs this, so a dataset change
        # with sensor_columns=None re-resolves for the new dataset.
        if self.sensor_columns is None:
            self.sensor_columns = list(DEFAULT_SENSOR_COLUMNS[self.dataset_kind()])

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

    def dataset_kind(self) -> str:
        """'cmapss', 'xjtu', or 'ncmapss' -- selects the loader in
        ``data.load_prepared`` (via the src/datasets/ registry)."""
        if self.dataset in XJTU_DATASETS:
            return "xjtu"
        if self.dataset in NCMAPSS_DATASETS or self.dataset.startswith("DS"):
            return "ncmapss"
        if self.dataset.startswith("FD"):
            return "cmapss"
        raise ValueError(
            f"unknown dataset {self.dataset!r}; expected FD001-FD004, "
            f"one of {XJTU_DATASETS}, or one of {NCMAPSS_DATASETS}")

    def is_simulated_dataset(self) -> bool:
        """True iff ``config.dataset`` is a SIMULATED family (C-MAPSS/N-CMAPSS) and
        may therefore be perturbed by ``noise_injection`` (RQ-H, sim-only)."""
        return self.dataset_kind() in SIMULATED_DATASET_KINDS

    def effective_condition_norm(self) -> bool:
        """Resolved condition-normalization flag: explicit value, else auto by
        dataset (ON for FD002/FD004 and XJTU-SY, OFF for FD001/FD003)."""
        if self.condition_norm is not None:
            return bool(self.condition_norm)
        return self.dataset in MULTI_CONDITION_DATASETS or self.dataset in XJTU_DATASETS

    # ---- result-artifact paths (experiment-namespaced) ---------------------
    def result_prefix(self) -> str:
        """Filename prefix applied to every result artifact: ``<experiment_name>_``
        (or "" when ``experiment_name`` is unset, preserving the flat layout)."""
        return f"{self.experiment_name}_" if self.experiment_name else ""

    def results_path(self, name: str) -> Path:
        """Path under ``results_dir`` for a result CSV or per-run bookkeeping dir,
        prefixed with the experiment name so runs never clobber each other
        (e.g. ``results/<exp>_results_v2.csv``, ``results/<exp>_runs``)."""
        return Path(self.results_dir) / f"{self.result_prefix()}{name}"

    def figures_dir(self) -> Path:
        """Directory for Stage C figures (filenames carry the experiment prefix)."""
        return Path(self.results_dir) / "figures"

    # ---- cache keys --------------------------------------------------------
    def _window_key_fields(self) -> dict:
        """Fields that determine the RAW cached FIXED windows (model-independent;
        baselines + raw-fusion last-cycle sensors read these)."""
        d = {
            "dataset": self.dataset,
            "window_size": self.window_size,
            "sensor_columns": list(self.sensor_columns),
            "max_rul": self.max_rul,
            "pad_short_test_units": self.pad_short_test_units,
            # Changes every cached window/embedding when toggled (CHANGES.md §21).
            "condition_norm": self.effective_condition_norm(),
        }
        # Sim-only perturbation (RQ-H, §38) mutates the readings BEFORE windowing, so
        # it changes the cached windows/embeddings -- but only when set. Added
        # CONDITIONALLY so every existing (unperturbed) FD001 key stays byte-identical.
        if self.noise_injection:
            d["noise_injection"] = dict(self.noise_injection)
        if self.dataset_kind() == "xjtu":  # split protocol changes the data itself
            d["xjtu_test_bearings"] = sorted(self.xjtu_test_bearings)
            d["xjtu_test_truncation"] = self.xjtu_test_truncation
        if self.dataset_kind() == "ncmapss":  # truncation changes the test data
            d["ncmapss_test_truncation"] = self.ncmapss_test_truncation
            if self.dataset == "DSALL":  # which files were unioned defines the dataset
                d["dsall_datasets"] = (sorted(self.dsall_datasets)
                                       if self.dsall_datasets is not None else "auto")
        return d

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
        # Cross-model fairness knob (RQ-M, §34): "mean" collapses the variate axis, so
        # it changes the pooled embeddings. Added CONDITIONALLY (only when != "concat")
        # so every existing FD001 embedding key is byte-identical (stable-key test).
        if self.channel_aggregation != "concat":
            d["channel_aggregation"] = self.channel_aggregation
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
