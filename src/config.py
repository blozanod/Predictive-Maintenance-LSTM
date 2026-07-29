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

# Per-cycle statistic SETS for the N-CMAPSS aggregation knob (RQ-G, CHANGES.md §53).
# "mean_std" is the historical default (37 channels: mean+std of the 18 raw channels
# plus cycle_len_s) and MUST stay first-listed/byte-identical. "mean_std_minmax_slope"
# is the richer arm: does a finer per-cycle summary buy anything, or is mean+std all a
# TSFM needs from a 1 Hz flight? DECISION (uncited): the two stat sets and the linear
# least-squares slope over the flight's 1 Hz rows (per-second units).
NCMAPSS_AGG_STAT_SETS = {
    "mean_std": ("mean", "std"),
    "mean_std_minmax_slope": ("mean", "std", "min", "max", "slope"),
}


def ncmapss_feature_columns(agg_stats: str = "mean_std") -> list:
    """Per-cycle channel names for an N-CMAPSS aggregation stat set: every raw channel
    x every statistic, plus the observable flight duration ``cycle_len_s``."""
    stats = NCMAPSS_AGG_STAT_SETS[agg_stats]
    return ([f"{v}_{s}" for v in NCMAPSS_W_VARS + NCMAPSS_XS_VARS for s in stats]
            + ["cycle_len_s"])


NCMAPSS_FEATURE_COLUMNS = ncmapss_feature_columns("mean_std")

# ---------------------------------------------------------------------------
# MetroPT-3 (UCI 791; Veloso et al. 2022) -- REAL industrial, CENSORED (§54)
# Porto Metro train Air Production Unit. One flat CSV, 15 sensor signals logged by an
# onboard device (nominally 1 Hz, SHIPPED decimated to ~10 s and irregular), Feb-Sep
# 2020, with FOUR documented air-leak events supplied out-of-band -- the file itself
# carries no label column. Column names/order below are the file's verbatim header
# (including its misspelling ``DV_eletric``); a mismatch fails loud at load.
# ---------------------------------------------------------------------------
METROPT_DATASETS = ("MetroPT-3",)
METROPT_ANALOG_COLUMNS = ("TP2", "TP3", "H1", "DV_pressure", "Reservoirs",
                          "Oil_temperature", "Motor_current")
# NOTE: 'DV_eletric' is spelled that way IN THE FILE (missing the 'c') -- do not "fix".
METROPT_DIGITAL_COLUMNS = ("COMP", "DV_eletric", "Towers", "MPG", "LPS",
                           "Pressure_switch", "Oil_level", "Caudal_impulses")
METROPT_SIGNAL_COLUMNS = METROPT_ANALOG_COLUMNS + METROPT_DIGITAL_COLUMNS
METROPT_TIMESTAMP_COLUMN = "timestamp"

# The shipped file's NOMINAL cadence in seconds. The stream is acquired at 1 Hz but
# SHIPPED decimated ~10x and irregular (10 s x1.34M, 9 s x128k, 12 s x38k). It is the
# reference a bin's expected row count is computed from, so the coverage test
# (metropt_min_bin_coverage) is invariant under the metropt_cycle_minutes sweep.
METROPT_NOMINAL_CADENCE_S = 10.0


def metropt_feature_columns() -> list:
    """Per-cycle channels: mean+std of each ANALOG signal (its level and its within-bin
    variability) plus the duty fraction of each DIGITAL signal -- for a binary channel
    the mean IS the fraction of the bin it was active, and a std adds nothing beyond it.
    DECISION (uncited), CHANGES.md §54."""
    return ([f"{c}_{s}" for c in METROPT_ANALOG_COLUMNS for s in ("mean", "std")]
            + [f"{c}_duty" for c in METROPT_DIGITAL_COLUMNS])


METROPT_FEATURE_COLUMNS = metropt_feature_columns()

# The four documented air-leak events (UCI "Failure Information" table, normalized to
# ISO from its published US M/D/YYYY H:MM form). The table's own numbering is typo'd
# (#1, #1, #3, #4) and its row-2 report says "30Apr" for a 29-30 MAY failure; both are
# recorded here as they were published and corrected only in this comment.
# DECISION (uncited): we take the UCI 4-event table as ground truth rather than the
# finer 21-window list in Davari et al. (DSAA 2021), because the UCI table is what the
# dataset itself ships and is what the prediction target ("time to the next documented
# intervention", RESEARCH_PLAN §4) is defined against. CHANGES.md §54.
METROPT_FAILURE_EVENTS = (
    {"event": 1, "start": "2020-04-18 00:00:00", "end": "2020-04-18 23:59:59",
     "failure": "Air leak", "severity": "High stress"},
    {"event": 2, "start": "2020-05-29 23:30:00", "end": "2020-05-30 06:00:00",
     "failure": "Air leak", "severity": "High stress"},
    {"event": 3, "start": "2020-06-05 10:00:00", "end": "2020-06-07 14:30:00",
     "failure": "Air leak", "severity": "High stress"},
    {"event": 4, "start": "2020-07-15 14:30:00", "end": "2020-07-15 19:00:00",
     "failure": "Air leak", "severity": "High stress"},
)

# ---------------------------------------------------------------------------
# UCI "Condition monitoring of hydraulic systems" (UCI 447; Helwig et al. 2015)
# REAL test rig; the RQ-F adjust-vs-replace anchor (native GRADED severity per cycle).
# 2205 cycles x 17 tab-delimited, header-less sensor files; row i of every file is
# cycle i (positional alignment ONLY -- there is no join key).
# ---------------------------------------------------------------------------
HYDRAULIC_DATASETS = ("Hydraulic",)
# sensor name -> (samples per 60 s cycle, unit). Verified byte-level against the shipped
# files: 7 files at 100 Hz, 2 at 10 Hz, 8 at 1 Hz; 43680 attributes per cycle in total.
HYDRAULIC_SENSORS = {
    "PS1": (6000, "bar"), "PS2": (6000, "bar"), "PS3": (6000, "bar"),
    "PS4": (6000, "bar"), "PS5": (6000, "bar"), "PS6": (6000, "bar"),
    "EPS1": (6000, "W"),
    "FS1": (600, "l/min"), "FS2": (600, "l/min"),
    "TS1": (60, "degC"), "TS2": (60, "degC"), "TS3": (60, "degC"), "TS4": (60, "degC"),
    "VS1": (60, "mm/s"), "CE": (60, "%"), "CP": (60, "kW"), "SE": (60, "%"),
}
HYDRAULIC_SENSOR_NAMES = tuple(HYDRAULIC_SENSORS)
HYDRAULIC_N_CYCLES = 2205          # asserted at load (fail loud on a truncated download)
HYDRAULIC_PROFILE_FILE = "profile.txt"

# profile.txt's 5 integer columns, in file order, with each component's value set
# ordered HEALTHY -> WORST. The raw value sets run in different directions (cooler and
# valve and accumulator are "% / bar, higher = healthier"; pump leakage is an index
# where higher = worse), so the loader maps every component onto ONE polarity:
# severity 0 = healthy, higher = worse. Not assuming a global polarity is the whole
# point -- getting it backwards would invert the RQ-F labels.
HYDRAULIC_PROFILE_COLUMNS = ("cooler", "valve", "pump", "accumulator", "stable_flag")
HYDRAULIC_SEVERITY_ORDER = {
    "cooler": (100, 20, 3),            # % efficiency: full -> reduced -> near failure
    "valve": (100, 90, 80, 73),        # % switching: optimal -> lag -> severe -> near failure
    "pump": (0, 1, 2),                 # leakage index: none -> weak -> severe
    "accumulator": (130, 115, 100, 90),  # bar pre-charge: optimal -> ... -> near failure
}
HYDRAULIC_COMPONENTS = tuple(HYDRAULIC_SEVERITY_ORDER)
# The RQ-F action taxonomy (RESEARCH_PLAN §2 RQ-F / §4). DECISION (uncited): severity 0
# is "none" (healthy), the WORST severity level of a component is "replace" (a terminal
# fault: a part must be swapped), and every intermediate level is "adjust" (a minor or
# self-correcting fault a technician tunes out). This is the only mapping the dataset's
# own severity ladder supports without inventing thresholds, and it makes the classes
# comparable across components whose ladders have different lengths.
HYDRAULIC_ACTIONS = ("none", "adjust", "replace")


def hydraulic_feature_columns(agg_stats: str = "mean_std") -> list:
    """Per-cycle channels: the chosen statistics of each sensor's intra-cycle samples.
    One row of every sensor file IS one 60 s cycle, so this is the same cycle-aggregation
    device N-CMAPSS uses (§27) -- and it is what makes the three sampling rates
    (100/10/1 Hz) commensurable without resampling anything."""
    stats = NCMAPSS_AGG_STAT_SETS[agg_stats]
    return [f"{s}_{stat}" for s in HYDRAULIC_SENSOR_NAMES for stat in stats]


HYDRAULIC_FEATURE_COLUMNS = hydraulic_feature_columns("mean_std")

# ---------------------------------------------------------------------------
# Backblaze Drive Stats -- REAL industrial, fleet-scale, CENSORED (§56)
# Daily SMART snapshots per drive. ``failure=1`` marks a drive's LAST operational day;
# a drive that simply stops appearing (retired/migrated) is RIGHT-CENSORED, not failed.
# ---------------------------------------------------------------------------
BACKBLAZE_DATASETS = ("Backblaze",)
BACKBLAZE_META_COLUMNS = ("date", "serial_number", "model", "capacity_bytes", "failure")
# DECISION (uncited): the default SMART set is the five attributes Backblaze itself
# calls out as failure-predictive (5, 187, 188, 197, 198) plus power-on hours (9, the
# age covariate) and temperature (194) -- restricted to RAW values, which are the
# physically-meaningful counts. Normalized values are vendor-rescaled and not comparable
# across models. Which of these a model actually populates VARIES BY MODEL (187/188 are
# absent on several) -- that is itself the RQ-C "what can you even record?" question at
# fleet scale, so the set is a config field, not a constant.
BACKBLAZE_DEFAULT_SMART = ("smart_5_raw", "smart_9_raw", "smart_187_raw",
                           "smart_188_raw", "smart_194_raw", "smart_197_raw",
                           "smart_198_raw")
# DECISION (uncited): default scope is two high-volume, long-lived HDD models with good
# SMART coverage. Restricting the fleet controls the model-heterogeneity confound
# (RESEARCH_PLAN §11) and keeps the parse tractable; override per experiment.
BACKBLAZE_DEFAULT_MODELS = ("ST12000NM0008", "HGST HMS5C4040ALE640")

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

# XJTU-SY feature MODE (RQ-D, CHANGES.md §52): how one 25.6 kHz / 32768-sample
# snapshot becomes model channels. "indicators" = the 16 hand-crafted condition
# indicators above (the historical default, byte-identical keys); "raw" = the
# snapshot's own SAMPLES reduced to a fixed width (a collection choice: what you
# would have recorded at a lower rate / coarser aggregation -- never a mutation of a
# kept value); "raw+indicators" = both. This is the direct test of "do TSFMs make
# hand-crafted condition indicators obsolete?" (RESEARCH_PLAN §2 RQ-D).
XJTU_FEATURE_MODES = ("indicators", "raw", "raw+indicators")

# How the 32768 samples of one snapshot are reduced to ``xjtu_raw_channels`` values
# per axis (DECISION (uncited), CHANGES.md §52):
#   * "decimate"    -- keep that many evenly-spaced RAW SAMPLES (IMPLEMENTATION_PLAN
#     §6.1's "fixed decimation of the 32768 samples"). Subtractive in the strictest
#     sense: every emitted number is a reading that was actually taken. It is exactly
#     what a practitioner who sampled at the corresponding lower rate would hold.
#   * "segment_rms" -- RMS within each of that many contiguous, equal segments. Keeps
#     the FULL-RATE energy of the snapshot while coarsening time resolution, i.e. the
#     aggregation-coarsening intervention (RQ-G on XJTU, RESEARCH_PLAN §5).
# Running both separates "the TSFM lost because of the sampling RATE" from "...because
# of the REPRESENTATION" -- the two are confounded under decimation alone.
XJTU_RAW_REDUCTIONS = ("decimate", "segment_rms")


def xjtu_raw_columns(n_per_axis: int) -> list:
    """Raw-sample channel names for the XJTU ``raw`` modes: ``h_raw_0..`` /
    ``v_raw_0..``, ``n_per_axis`` per accelerometer axis (2 * n total)."""
    return [f"{ax}_raw_{i}" for ax in ("h", "v") for i in range(n_per_axis)]


# Default sensor channels per dataset KIND, applied when config.sensor_columns is
# left None -- switching datasets is one knob, no cryptic KeyError deep in
# preprocessing (CHANGES.md §24). Values match the previously-required explicit
# lists exactly, so resolved configs hash to the SAME cache keys as before.
DEFAULT_SENSOR_COLUMNS = {
    "cmapss": list(FD001_NONCONSTANT_SENSORS),
    "xjtu": list(XJTU_FEATURE_COLUMNS),
    "ncmapss": list(NCMAPSS_FEATURE_COLUMNS),
    "metropt": list(METROPT_FEATURE_COLUMNS),
    "hydraulic": list(HYDRAULIC_FEATURE_COLUMNS),
    "backblaze": list(BACKBLAZE_DEFAULT_SMART),
}

# Dataset families whose fleets are MOSTLY HEALTHY: rare observed failures plus many
# right-censored survivors (RESEARCH_PLAN §4). Their loaders emit the
# ``event_observed`` column, they are scored with the alarm/lead-time metric rather
# than the NASA score, and their numbers are never tabled against the run-to-failure
# datasets' (CHANGES.md §54).
CENSORED_DATASET_KINDS = ("metropt", "backblaze")

# Dataset families that contain NO failure events at all, so neither a RUL nor an alarm
# target is meaningful for them: the UCI hydraulic rig is a cyclic controlled-fault
# INJECTION experiment, where a "unit" (a constant-fault label block) ends because the
# experimenter changed the set-point, not because anything degraded or failed. Its
# time-to-event targets are therefore degenerate BY CONSTRUCTION -- on the real record
# every block is the same length, so `rul_truth` comes out a constant and the
# predict-the-mean floor scores a perfect 0.0 RMSE that no model can beat.
#
# Rather than table that meaningless number next to C-MAPSS's, the campaign runs these
# datasets for their ACTUAL deliverable: the RQ-F few-shot taxonomy probe
# (src/taxonomy.py). Their loaders emit `event_observed = 0` everywhere -- literally
# true: no block ends in an observed failure (CHANGES.md §55).
CLASSIFICATION_DATASET_KINDS = ("hydraulic",)


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

    # ---- XJTU-SY feature mode: raw-vs-indicators (RQ-D; CHANGES.md §52) ------
    # How one 25.6 kHz snapshot becomes model channels (XJTU_FEATURE_MODES):
    # "indicators" (default, the historical 16 hand-crafted channels -- keys unchanged),
    # "raw" (2 * xjtu_raw_channels reduced sample channels), or "raw+indicators".
    # The three fields join the window key ONLY when the mode is not "indicators",
    # so every recorded XJTU key is byte-identical. xjtu-only (like the split fields).
    xjtu_feature_mode: str = "indicators"
    # Raw channels emitted PER AXIS (total = 2x this). DECISION (uncited): 16 keeps the
    # raw arm's channel count (32) in the same order as the indicator arm's (16), so the
    # comparison is not confounded by a 1000x wider input; sweep it for RQ-G.
    xjtu_raw_channels: int = 16
    # How samples are reduced to those channels (XJTU_RAW_REDUCTIONS).
    xjtu_raw_reduce: str = "decimate"

    # ---- N-CMAPSS split protocol (ignored for C-MAPSS/XJTU; CHANGES.md §27-28) --
    # The file's own *_test units are run-to-failure (RUL hits 0 at the last row); to
    # match the pipeline's predict-at-last-observed-cycle protocol each test unit is
    # truncated at this life fraction (same device as XJTU, §22). ncmapss-only cache-
    # key field. DECISION (uncited): 0.6 mirrors the XJTU default; no community standard.
    ncmapss_test_truncation: float = 0.6
    # ---- N-CMAPSS aggregation granularity (RQ-G; CHANGES.md §53) ------------
    # How the 1 Hz WITHIN-flight rows are reduced to one cycle row. Both are
    # ncmapss-only cache-key fields, added ONLY when non-default (existing keys
    # byte-identical), and both also key the per-file AGGREGATE cache.
    # ``ncmapss_agg_stride`` sub-samples each flight's rows 1-in-N BEFORE aggregating
    # (the "how finely must you sample?" intervention: stride 10 means a 0.1 Hz
    # recorder). ``ncmapss_agg_stats`` selects the per-cycle statistic set
    # (NCMAPSS_AGG_STAT_SETS): "mean_std" (default, 37 channels) or the richer
    # "mean_std_minmax_slope" (91 channels).
    ncmapss_agg_stride: int = 1
    ncmapss_agg_stats: str = "mean_std"
    # ---- MetroPT-3 protocol (CHANGES.md §54) --------------------------------
    # The 1 Hz-ish irregular stream is binned into fixed-duration "cycles"; this is the
    # bin width in MINUTES and is the dataset's RQ-G sampling/aggregation lever.
    # DECISION (uncited): 60 min gives ~5100 cycles over the 7-month record -- enough
    # per-run history for a 30-cycle window while keeping each bin densely populated
    # (~360 raw samples at the shipped ~10 s cadence).
    metropt_cycle_minutes: int = 60
    # A bin with fewer than this many raw samples is DROPPED rather than aggregated:
    # ~17.6% of wall-clock time is missing from the shipped file with no gap marker, so
    # a sparsely-covered bin's mean/std would be a different quantity from a full bin's.
    # This is the ABSOLUTE floor; the scale-invariant test below usually binds first.
    metropt_min_samples_per_cycle: int = 10
    # Minimum fraction of a bin's wall-clock time that must actually be covered by rows,
    # at the nominal ~10 s shipped cadence (METROPT_NOMINAL_CADENCE_S). A bin is kept
    # only if it holds BOTH >= metropt_min_samples_per_cycle rows AND >= this fraction of
    # the rows a fully-covered bin would hold.
    #
    # Why a FRACTION and not just a count: metropt_cycle_minutes is this dataset's RQ-G
    # sweep lever, so an absolute floor makes the data-quality filter ~140x stricter at
    # 10-minute bins than at 1440-minute ones -- the aggregation-granularity comparison
    # would then be confounded by a coverage gradient instead of being apples-to-apples.
    # DECISION (uncited): 0.5 keeps any bin at least half covered, which at the 60-minute
    # default drops the badly-holed bins while retaining the typical ~82%-covered one.
    metropt_min_bin_coverage: float = 0.5
    # Which intervention runs are held out for test (1-based, in chronological order;
    # run k ends at documented event k). DECISION (uncited): run 4 (the July air leak)
    # is the last run that ends in an OBSERVED event, so it is the only choice that
    # gives the test split a real failure to predict; the censored tail run stays in
    # train, where it legitimately contributes alarm-negative rows.
    metropt_test_runs: list = field(default_factory=lambda: [4])
    metropt_test_truncation: float = 0.6   # same predict-at-last-observed-cycle device

    # ---- UCI Hydraulic protocol (CHANGES.md §55) ----------------------------
    # Drop the settling cycles the rig flags as not-yet-stable. NOTE the polarity: the
    # shipped column is 1 = NOT stable. DECISION (uncited): dropping them is the
    # standard preprocessing and leaves a near-perfect factorial design.
    hydraulic_drop_unstable: bool = True
    # Which component's action label the RQ-F probe targets (HYDRAULIC_COMPONENTS).
    hydraulic_taxonomy_component: str = "valve"
    # Per-cycle statistic set over each sensor's intra-cycle samples (NCMAPSS_AGG_STAT_SETS).
    hydraulic_agg_stats: str = "mean_std"
    hydraulic_test_fraction: float = 0.3   # unit(=label-block)-level held-out fraction

    # ---- Backblaze protocol (CHANGES.md §56) --------------------------------
    # Scope control (RESEARCH_PLAN §11): restrict the fleet to a few high-volume drive
    # models so SMART availability and failure physics are comparable across units.
    backblaze_models: list = field(
        default_factory=lambda: list(BACKBLAZE_DEFAULT_MODELS))
    # Which SMART columns become model channels. Selecting BY NAME (never by position)
    # is mandatory: the daily CSV's column count drifts across quarters and new SMART
    # columns are INSERTED in ascending attribute order, not appended.
    backblaze_smart_columns: list = field(
        default_factory=lambda: list(BACKBLAZE_DEFAULT_SMART))
    # Optional inclusive date bounds ("YYYY-MM-DD") over the daily snapshots.
    backblaze_start_date: Optional[str] = None
    backblaze_end_date: Optional[str] = None
    # Drives with fewer than this many observed days carry too little history to window.
    backblaze_min_days: int = 40
    # DECISION (uncited): cap the CENSORED survivors kept per model. The fleet is ~1 in
    # 23,500 drive-days a failure; keeping every survivor makes Stage A dominated by
    # drives that never fail while adding little signal. Every FAILED drive is always
    # kept -- only survivors are subsampled, seeded and recorded. None => keep all.
    backblaze_max_survivors_per_model: Optional[int] = 200
    backblaze_test_fraction: float = 0.3   # drive-level held-out fraction

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

    # ---- censoring & the alarm target (RESEARCH_PLAN §4; CHANGES.md §54) ----
    # Real fleets (MetroPT, Backblaze) are MOSTLY HEALTHY: rare failures plus many
    # right-censored survivors, so forcing a RUL regression on every unit invents a
    # failure date the data does not contain. ``alarm_horizon`` switches on the
    # censoring-aware target: "will this unit reach an intervention within H cycles?",
    # a binary label every censored survivor can still legitimately contribute a 0 to
    # (as long as it was observed for the whole horizon -- see data.add_alarm_label,
    # which leaves the genuinely unknown rows NaN and drops them rather than guessing).
    # None => pure RUL (every run-to-failure dataset; unchanged behaviour). Because it
    # changes the LABEL, it is a window-cache-key field -- added ONLY when set, so every
    # existing key is byte-identical. Must be < max_rul (a horizon at or beyond the
    # clip point is invisible after clipping); validated below.
    alarm_horizon: Optional[int] = None
    # Probability threshold at which an alarm is declared when scoring the binary arm
    # (evaluate.alarm_metrics). DECISION (uncited): 0.5 is the neutral default; the
    # threshold SWEEP (alarm_threshold_sweep) is the real deliverable, exactly as the
    # cost curve is for the RUL arms -- no single arbitrary operating point.
    alarm_threshold: float = 0.5

    # ---- losses ------------------------------------------------------------
    # Phase-1 loss arms. "quantile" is the optional third arm (RESEARCH_PLAN sec.5);
    # "failure_within_horizon" is the censored/alarm arm (requires alarm_horizon).
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
        if self.xjtu_feature_mode not in XJTU_FEATURE_MODES:
            raise ValueError(
                f"xjtu_feature_mode must be one of {XJTU_FEATURE_MODES}, "
                f"got {self.xjtu_feature_mode!r}")
        if self.xjtu_raw_reduce not in XJTU_RAW_REDUCTIONS:
            raise ValueError(
                f"xjtu_raw_reduce must be one of {XJTU_RAW_REDUCTIONS}, "
                f"got {self.xjtu_raw_reduce!r}")
        if self.xjtu_raw_channels < 1:
            raise ValueError(
                f"xjtu_raw_channels must be >= 1, got {self.xjtu_raw_channels}")
        if self.ncmapss_agg_stats not in NCMAPSS_AGG_STAT_SETS:
            raise ValueError(
                f"ncmapss_agg_stats must be one of {sorted(NCMAPSS_AGG_STAT_SETS)}, "
                f"got {self.ncmapss_agg_stats!r}")
        if self.ncmapss_agg_stride < 1:
            raise ValueError(
                f"ncmapss_agg_stride must be >= 1 (1 = keep every 1 Hz row), got "
                f"{self.ncmapss_agg_stride}")
        if self.alarm_horizon is not None:
            if self.alarm_horizon < 1:
                raise ValueError(
                    f"alarm_horizon must be >= 1 cycle, got {self.alarm_horizon}")
            if self.alarm_horizon >= self.max_rul:
                # The binary label is read off the (clipped) RUL, so a horizon at or
                # beyond the clip point would mark every clipped row as "no alarm".
                raise ValueError(
                    f"alarm_horizon ({self.alarm_horizon}) must be < max_rul "
                    f"({self.max_rul}): the alarm label is read off the RUL target, "
                    f"which is clipped at max_rul, so a horizon at/after the clip "
                    f"point is invisible. Raise max_rul or lower the horizon.")
        if not 0.0 < self.alarm_threshold < 1.0:
            raise ValueError(
                f"alarm_threshold must be in (0, 1), got {self.alarm_threshold}")
        if self.hydraulic_agg_stats not in NCMAPSS_AGG_STAT_SETS:
            raise ValueError(
                f"hydraulic_agg_stats must be one of {sorted(NCMAPSS_AGG_STAT_SETS)}, "
                f"got {self.hydraulic_agg_stats!r}")
        if self.hydraulic_taxonomy_component not in HYDRAULIC_COMPONENTS:
            raise ValueError(
                f"hydraulic_taxonomy_component must be one of {HYDRAULIC_COMPONENTS}, "
                f"got {self.hydraulic_taxonomy_component!r}")
        if self.metropt_cycle_minutes < 1:
            raise ValueError(
                f"metropt_cycle_minutes must be >= 1, got {self.metropt_cycle_minutes}")
        if not 0.0 <= self.metropt_min_bin_coverage <= 1.0:
            raise ValueError(
                f"metropt_min_bin_coverage is a FRACTION of a bin's wall-clock time and "
                f"must be in [0, 1], got {self.metropt_min_bin_coverage}")
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
        # NOTE: a mode/stat-set change that alters the CHANNEL SET (XJTU feature mode,
        # N-CMAPSS stat set) must be passed together with ``sensor_columns=None`` --
        # ``replace`` carries the already-resolved list forward otherwise. The factor
        # probes do exactly that (src/probes.py).
        if self.sensor_columns is None:
            self.sensor_columns = list(self.default_sensor_columns())

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

    def default_sensor_columns(self) -> list:
        """The channel set this dataset serves at the CURRENT feature/aggregation mode.

        Falls back to ``DEFAULT_SENSOR_COLUMNS[kind]`` for every family whose channel
        set is fixed. The two families with a mode knob resolve it here so switching
        the knob is one field, not a hand-copied channel list:
          * XJTU-SY -- ``xjtu_feature_mode`` (RQ-D, §52);
          * N-CMAPSS -- ``ncmapss_agg_stats`` (RQ-G, §53).
        At the default mode each returns EXACTLY the historical list, so every recorded
        cache key is byte-identical (asserted in tests/test_cache_keys.py)."""
        kind = self.dataset_kind()
        if kind == "xjtu":
            raw = xjtu_raw_columns(self.xjtu_raw_channels)
            if self.xjtu_feature_mode == "raw":
                return list(raw)
            if self.xjtu_feature_mode == "raw+indicators":
                return list(raw) + list(XJTU_FEATURE_COLUMNS)
            return list(XJTU_FEATURE_COLUMNS)
        if kind == "ncmapss":
            return ncmapss_feature_columns(self.ncmapss_agg_stats)
        if kind == "hydraulic":
            return hydraulic_feature_columns(self.hydraulic_agg_stats)
        if kind == "backblaze":
            # Fleet-scale RQ-C: WHICH SMART attributes you record is the config choice.
            return list(self.backblaze_smart_columns)
        return list(DEFAULT_SENSOR_COLUMNS[kind])

    def effective_tsfm_context(self) -> int:
        """History length (cycles) the TSFM sees. Defaults to the baseline window."""
        return self.tsfm_context_length if self.tsfm_context_length is not None else self.window_size

    def dataset_kind(self) -> str:
        """The loader family for ``config.dataset`` -- 'cmapss', 'xjtu', 'ncmapss',
        'metropt', 'hydraulic' or 'backblaze' -- which ``data.load_prepared`` dispatches
        on through the src/datasets/ registry."""
        if self.dataset in XJTU_DATASETS:
            return "xjtu"
        if self.dataset in METROPT_DATASETS:
            return "metropt"
        if self.dataset in HYDRAULIC_DATASETS:
            return "hydraulic"
        if self.dataset in BACKBLAZE_DATASETS:
            return "backblaze"
        if self.dataset in NCMAPSS_DATASETS or self.dataset.startswith("DS"):
            return "ncmapss"
        if self.dataset.startswith("FD"):
            return "cmapss"
        raise ValueError(
            f"unknown dataset {self.dataset!r}; expected FD001-FD004, "
            f"one of {XJTU_DATASETS}, one of {NCMAPSS_DATASETS}, "
            f"one of {METROPT_DATASETS}, one of {HYDRAULIC_DATASETS}, "
            f"or one of {BACKBLAZE_DATASETS}")

    def is_censored_dataset(self) -> bool:
        """True iff this dataset's fleet is mostly-healthy with right-censored
        survivors (MetroPT, Backblaze), so it is scored with the alarm/lead-time metric
        and never tabled against the run-to-failure NASA scores (§54)."""
        return self.dataset_kind() in CENSORED_DATASET_KINDS

    def is_classification_dataset(self) -> bool:
        """True iff this dataset contains NO failure events, so neither a RUL nor an
        alarm target is meaningful and its deliverable is the RQ-F taxonomy probe
        (UCI Hydraulic -- see CLASSIFICATION_DATASET_KINDS)."""
        return self.dataset_kind() in CLASSIFICATION_DATASET_KINDS

    def is_simulated_dataset(self) -> bool:
        """True iff ``config.dataset`` is a SIMULATED family (C-MAPSS/N-CMAPSS) and
        may therefore be perturbed by ``noise_injection`` (RQ-H, sim-only)."""
        return self.dataset_kind() in SIMULATED_DATASET_KINDS

    def effective_noise_seed(self) -> int:
        """Resolved seed for the sim-only ``noise_injection`` perturbation: the noise
        spec's own ``seed`` if given, else ``config.seed``. The single source of truth
        used both by ``data.apply_noise_injection`` (to draw the perturbation) and by
        the cache key (to capture it) -- so a ``config.seed`` change re-keys the
        perturbed cache instead of silently reusing a differently-perturbed one."""
        return int(self.noise_injection.get("seed", self.seed))

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
        # The censoring/alarm target adds a LABEL column and drops the rows whose alarm
        # label is unknowable (§54), so it changes the cached windows -- but only when
        # set, so every run-to-failure key stays byte-identical.
        if self.alarm_horizon is not None:
            d["alarm_horizon"] = int(self.alarm_horizon)
        # Sim-only perturbation (RQ-H, §38) mutates the readings BEFORE windowing, so
        # it changes the cached windows/embeddings -- but only when set. Added
        # CONDITIONALLY so every existing (unperturbed) FD001 key stays byte-identical.
        if self.noise_injection:
            d["noise_injection"] = dict(self.noise_injection)
            # The perturbation seed defaults to config.seed, which is otherwise in NO
            # cache key; fold the RESOLVED seed in so two configs differing only in
            # config.seed can't collide on one differently-perturbed cache (the
            # "cache keys are pure functions of Config" invariant, §1.2).
            d["noise_seed"] = self.effective_noise_seed()
        if self.dataset_kind() == "xjtu":  # split protocol changes the data itself
            d["xjtu_test_bearings"] = sorted(self.xjtu_test_bearings)
            d["xjtu_test_truncation"] = self.xjtu_test_truncation
            # RQ-D feature mode (§52): the raw arms emit DIFFERENT channel values from
            # the same snapshots, so they must key apart -- but only when engaged, so
            # every recorded indicator-mode XJTU key stays byte-identical.
            if self.xjtu_feature_mode != "indicators":
                d["xjtu_feature_mode"] = self.xjtu_feature_mode
                d["xjtu_raw_channels"] = self.xjtu_raw_channels
                d["xjtu_raw_reduce"] = self.xjtu_raw_reduce
        if self.dataset_kind() == "ncmapss":  # truncation changes the test data
            d["ncmapss_test_truncation"] = self.ncmapss_test_truncation
            # RQ-G aggregation knobs (§53): each changes the per-cycle values, so each
            # keys -- conditionally, so existing DS0x/DSALL keys are unchanged.
            if self.ncmapss_agg_stride != 1:
                d["ncmapss_agg_stride"] = self.ncmapss_agg_stride
            if self.ncmapss_agg_stats != "mean_std":
                d["ncmapss_agg_stats"] = self.ncmapss_agg_stats
            if self.dataset == "DSALL":  # which files were unioned defines the dataset
                d["dsall_datasets"] = (sorted(self.dsall_datasets)
                                       if self.dsall_datasets is not None else "auto")
        # The three Phase-B families are NEW, so every field that shapes their data goes
        # in unconditionally for them -- and, being family-scoped, can never re-key an
        # existing C-MAPSS / XJTU / N-CMAPSS cache.
        if self.dataset_kind() == "metropt":
            d["metropt_cycle_minutes"] = self.metropt_cycle_minutes
            d["metropt_min_samples_per_cycle"] = self.metropt_min_samples_per_cycle
            d["metropt_min_bin_coverage"] = self.metropt_min_bin_coverage
            d["metropt_test_runs"] = sorted(int(r) for r in self.metropt_test_runs)
            d["metropt_test_truncation"] = self.metropt_test_truncation
        if self.dataset_kind() == "hydraulic":
            d["hydraulic_drop_unstable"] = self.hydraulic_drop_unstable
            d["hydraulic_agg_stats"] = self.hydraulic_agg_stats
            d["hydraulic_test_fraction"] = self.hydraulic_test_fraction
        if self.dataset_kind() == "backblaze":
            d["backblaze_models"] = sorted(self.backblaze_models)
            d["backblaze_start_date"] = self.backblaze_start_date
            d["backblaze_end_date"] = self.backblaze_end_date
            d["backblaze_min_days"] = self.backblaze_min_days
            d["backblaze_max_survivors_per_model"] = self.backblaze_max_survivors_per_model
            d["backblaze_test_fraction"] = self.backblaze_test_fraction
            # Survivor subsampling is seeded from config.seed, which is in no other key.
            d["backblaze_seed"] = self.seed
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
