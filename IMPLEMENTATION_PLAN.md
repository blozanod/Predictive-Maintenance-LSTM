# Implementation Plan: The "When Do TSFMs Work" Playbook — build one dataset deep, then scale

**Status:** ready to implement · **Prepared:** July 2026 · **Target branch:** `claude/tsfm-research-plan-lc2zgc`

This is a self-contained instruction set for an implementing agent. It realizes `RESEARCH_PLAN.md` v2
(the repositioned "when/why do TSFMs win for RUL, and how should you collect data so they do" study).
**Read `RESEARCH_PLAN.md` (the research protocol) and `CHANGES.md` (§1–§31, the audit trail) first**, then
`DATASET_EXPANSION_PLAN.md` (the template for how a dataset gets added — this plan follows the same style).

The repo is **not** greenfield. A rigorous Chronos-2 pipeline already exists and is the right pipeline
(§1). This plan **extends** it; it never rewrites working, tested modules. Every instruction below either
adds a new module behind an existing registry or extends an existing function at a named seam.

---

## 0. Strategy (the two-phase build the user asked for)

> **Build the full experimental depth on ONE dataset (C-MAPSS), prove it answers every question it can,
> then aggressively replicate across the other datasets, each carrying its dataset-specific questions.**

Concretely:

- **Phase A — the reusable engine + the C-MAPSS vertical slice (Milestones 0–2).** Stand up all five
  TSFMs, the win-rule/scoring layer, the earliness metrics, the factor-probe harness, the zero-shot arm,
  and the reporting — then run **every RQ that C-MAPSS can answer end-to-end** (RQ-A history, RQ-B
  data-efficiency, RQ-C channels, RQ-E labeling/ordinal, RQ-H noise (C-MAPSS is simulated → perturbation
  is allowed here), RQ-M which-TSFM/does-multivariate-matter, RQ-Z zero-shot). This is the template
  experiment; it must be perfect and 100%-tested before scaling.
- **Phase B — aggressive horizontal scale (Milestones 3–7).** Replicate the template per dataset, each
  adding its loader plus the one or two chapters unique to it: XJTU→raw-vs-indicators (RQ-D), N-CMAPSS→
  sampling/aggregation (RQ-G) + high-data, MetroPT-3→alarms/lead-time + censoring, UCI Hydraulic→
  adjustment-vs-replacement (RQ-F), Backblaze→censored fleet-scale.

**What C-MAPSS cannot answer** (built as machinery in Phase A, exercised on real data in Phase B):
RQ-D (needs XJTU raw waveforms), RQ-F (needs Hydraulic/MetroPT action labels), censoring (needs
Backblaze/MetroPT), RQ-G (needs sub-cycle data). Phase A builds and unit-tests each of these against
**synthetic fixtures** so the interfaces are frozen before scale-up; Phase B wires the real data in.

---

## 1. The repo contract (invariants — violating any of these fails review)

1. **Every result-affecting choice is a `Config` field or a `# DECISION (uncited):` tag.** Grep-able:
   `grep -rn "DECISION (uncited):" src/`. No magic constants buried in modules.
2. **Cache keys are pure functions of `Config`** (`_window_key_fields`, `_embedding_key_fields`). Paths,
   `experiment_name`, and disk contents are **never** in a key. A new key field that changes cached data
   must be added to the key; a new field that defaults to current behavior must be added **conditionally
   (only when non-default)** so existing keys stay byte-identical — there is a test asserting FD001's keys
   (`windows_FD001_…`, the embedding key) never drift; keep it green. If a schema genuinely must change,
   bump `CACHE_SCHEMA_VERSION` and update that test's expected hash in the same commit, with a `CHANGES.md`
   note.
3. **Every stage is restartable** via completed-cell detection (`completed_cells`, per-cell keys). New
   sweeps follow the same pattern.
4. **One loading path:** all data comes through `data.load_prepared(config)`. New datasets are new loader
   modules behind `datasets/__init__.DATASET_LOADERS`; new models are new modules behind
   `models/__init__.EMBEDDERS`. No stage bypasses these registries.
5. **Reuse reference implementations; do not reimplement** (CORN→coral-pytorch, GBM→lightgbm,
   MiniRocket→sktime, and now MOMENT→momentfm, Moirai→uni2ts, TimesFM→timesfm, TTM→granite-tsfm,
   censoring→scikit-survival/lifelines, catch22→pycatch22). Wrap, don't fork.
6. **No numeric result or claim is ever written into the repo except from a completed run.** Recorded
   winners (`CHANGES.md` §12) come only from real runs; this plan writes **no numbers**.
7. **Fail loud:** channel-name/shape/registry drift raises with both the expected and observed values,
   never silently adapts (the N-CMAPSS `*_var` check, the registry-drift test — mirror this everywhere).
8. **CPU-only tests, no downloads, no GPU:** heavy backbones are lazily imported inside a `_load_*` method
   never hit by tests; every dataset/model is exercised through a synthetic fixture + a mock. 100% line +
   branch coverage on `src/` is the gate (§8).
9. **Append, never edit, `CHANGES.md`.** Next free section is **§32**.

---

## 2. Dependencies to add (`requirements.txt`)

Add under a clearly commented "v2 backbones + real datasets" block. Pin conservatively; keep heavy DL deps
in the GPU section, keep anything tests import in core.

| Package | For | Section |
|---|---|---|
| `momentfm` | MOMENT embedder | GPU |
| `uni2ts` | Moirai / Moirai-2 embedder | GPU |
| `timesfm` | TimesFM 2.5 embedder | GPU |
| `granite-tsfm` (or `tsfm_public`) | TTM embedder | GPU |
| `pycatch22` | catch22 features (cheap foil) | core (tests use it) |
| `scikit-survival` **or** `lifelines` | censored/time-to-event metrics & loss (Backblaze/MetroPT) | core |
| `pyarrow` | Backblaze SMART parquet/CSV at scale | core |
| `pytest-cov` | coverage gate (§8) | dev |

Each GPU backbone is imported **only** inside its embedder's `_load_pipeline`, so `pytest` never imports it
(exactly as `chronos` is handled today). `package_versions()` in `evaluate.py` — extend its module list to
include the new backbones so provenance JSON records them.

---

## 3. Milestone 0 — Foundations (do first; small but unblocks everything)

| # | Task | Files | Size |
|---|------|-------|------|
| 0.1 | Coverage gate: `pytest-cov`, `.coveragerc` (`fail_under=100` for `src/`, `# pragma: no cover` allowed only on lazy backbone imports), CI-ready `pytest -q --cov=src --cov-branch` | `.coveragerc`, `requirements.txt`, `README.md` | S |
| 0.2 | Extend `package_versions()` module list (momentfm, uni2ts, timesfm, tsfm_public, pycatch22, sksurv/lifelines) | `src/evaluate.py` | S |
| 0.3 | Extend `MockEmbedder` to parametrize **token layout** and **channel count**, so tests can stand in for BOTH multivariate-native (Chronos-2-like, appends REG+forecast tokens) and univariate (per-channel, no special tokens) embedders | `tests/synthetic.py` | S |

`# DECISION (uncited):` — the `# pragma: no cover` boundary is the single line where a lazy backbone/
dataset library is first imported (e.g. `from chronos import Chronos2Pipeline`). Everything above it (shape
handling, pooling, loc/scale, caching, scoring) is covered by mocks. Record this policy in `.coveragerc`
comments and `CHANGES.md` §32.

---

## 4. Milestone 1 — The reusable experiment engine

Model-agnostic and dataset-agnostic. This is what makes Phase B "aggressive."

### 4.1 The four new TSFM embedders + cross-model fairness (RQ-M)

**One new module per backbone in `src/models/`**, each exposing the exact `embeddings.Embedder`
protocol already in use:
`embed_windows(contexts) -> (emb (N, F) float32, loc_scale (N, n_variates, 2) float32)` and
`describe() -> dict`. Register each in `EMBEDDERS` (`src/models/__init__.py`).

| File | model_name key | Family | d_model (approx) | Native handling |
|---|---|---|---|---|
| `models/moirai.py` | `Salesforce/moirai-2` | multivariate-native (any-variate) | ~ per release | flatten variates with variate-ids (uni2ts native) |
| `models/moment.py` | `AutonLab/MOMENT-1-large` | univariate | 1024 | per-channel embed → concat |
| `models/timesfm.py` | `google/timesfm-2.5` | univariate | ~1280 | per-channel embed → concat |
| `models/ttm.py` | `ibm-granite/granite-timeseries-ttm-r2` | tiny (channel-mixing) | ~small | native multivariate (TTM mixes channels) |

**The pooling contract is semantic, not index-based (critical).** Chronos-2's four poolings
(`forecast_token`, `last_content`, `mean`, `flatten`) assume its token layout (content patches + REG +
forecast). Other backbones have different layouts (MOMENT: a CLS-like reconstruction embedding and/or
per-patch; TimesFM/Moirai: per-patch hidden states, no special tokens). **Each embedder maps the four
pooling names onto its own layout** and documents the mapping in its module docstring + a
`# DECISION (uncited):` tag:
- `forecast_token` → the model's CLS/summary token if it has one, else its **last patch** (the closest
  "predict-next" summary).
- `last_content` → last real content patch.
- `mean` → mean over content patches.
- `flatten` → all content patches concatenated (fixed-context only).

The generic `pool_window_embedding` / `_pool_one_torch` in `embeddings.py` stay the Chronos-2 reference;
each new embedder implements its own `_pool` honoring the same four names. This keeps `config.pooling`
meaningful (and cache-key-valid) across all five models.

**Univariate models (MOMENT, TimesFM):** internally loop channels, embed each 1-D series, stack per-channel
patch embeddings into the canonical `(n_variates, patches, d_model)` tensor, then reuse the pooling. `F =
n_variates · d_model` under the default `channel_aggregation="concat"`. Capture **per-channel** instance-
norm (RevIN) loc/scale → `(N, n_variates, 2)` exactly as Chronos-2 does. If a model doesn't expose loc/
scale, compute it from the input series (per-channel mean/std) — record as a `# DECISION (uncited):`.

**Multivariate-native models (Moirai-2, TTM):** feed all channels jointly; the returned per-variate
embeddings pool the same way. TTM mixes channels internally, so per-variate embeddings still exist post-
backbone; Moirai flattens variates — map its any-variate output back to `(n_variates, patches, d_model)`.

**New `Config` field — channel aggregation (the fairness knob, RQ-M):**
```python
# How the pooled per-variate embeddings collapse into the head feature vector.
# "concat" (default) preserves per-channel detail (F = n_variates * d_model) and is
# how a practitioner uses each model; "mean" collapses the variate axis (F = d_model)
# for the cross-TSFM common-representation fairness control (RESEARCH_PLAN §6, RQ-M).
channel_aggregation: str = "concat"   # {"concat", "mean"}
```
Add to `_embedding_key_fields()` **only when `!= "concat"`** (so every existing key is byte-identical and
the stable-key test stays green). Apply it uniformly to ALL models' pooled output (including Chronos-2/
Moirai) so the fairness control is genuinely common.

**Integration-spike gate (per backbone, Phase 1 Risk mitigation).** Before a backbone joins the campaign,
a one-off notebook/CLI spike confirms: (a) representation extraction works (`.embed()`/encoder hidden
states), (b) the `(n_variates, patches, d_model)` mapping + loc/scale are correct, (c) full-data FD001
Chronos-2-parity sanity — the backbone produces a finite, non-degenerate RMSE. **Documented fallback if a
model won't expose clean representations: penultimate/encoder hidden states**, recorded as a
`# DECISION (uncited):`. A backbone that cannot pass the spike is reported as such, not forced.

**`run_representation_fairness(config, models, device, embedder_factory=None)`** (new, in `sweep.py` or a
small `fairness.py`): full-data, MSE, ≥3 seeds, each model run twice — native (`channel_aggregation=
concat`, its own default pooling) and common (`channel_aggregation=mean`, `pooling=mean`) — writing
`representation_fairness.csv`. Confirms the cross-TSFM ranking is not an aggregation artifact. Restartable;
CPU-testable with two differently-shaped `MockEmbedder`s.

**Tests (100% of new embedder logic, via mocks — no backbone import):** the pooling-name→layout mapping for
each layout kind; `concat` vs `mean` output dims; loc/scale shape `(N, n_variates, 2)`; empty-context edge
case; `describe()` keys; `EMBEDDERS` registry + `make_embedder` selects each; a registry-drift test
extension asserting every `EMBEDDERS` key is a real registered module and vice-versa (mirror the datasets
drift test).

### 4.2 Scoring & the win-rule (`src/scoring.py`, new)

The formal realization of `RESEARCH_PLAN §8`. Reads the per-combo results CSVs (the campaign writes one per
`<dataset>_<model>`), assembles the **success map**, and applies the win/tie/loss rule.

Functions:
- `strongest_baseline_per_cell(rows, metric) -> {(dataset, n_units): (best_baseline, seed_mean)}` — the
  toughest bar: the best seed-mean over ALL baseline rows (`predict_mean, gbm, minirocket, cnn, lstm,
  cycle_reg, gbm_age, catch22_gbm`) in that `(dataset, n_units)` cell.
- `win_verdict(tsfm_rows, baseline_rows, config) -> {cell: verdict}` where verdict ∈ {`win`, `tie`,
  `loss`, `hollow`}:
  - primary metric = **`nasa_clipped`** (asymmetric; `RESEARCH_PLAN §8`); RMSE reported alongside.
  - **win** iff TSFM seed-mean beats the strongest baseline's seed-mean AND a paired-seed test (reuse/
    generalize `evaluate.paired_seed_ttest` to arbitrary model pairs on the main CSV) supports it at the
    configured level.
  - **tie** iff the difference is within noise (CI overlaps / p above threshold).
  - **hollow** iff the **absolute-floor guard** fires: even the winner's error exceeds a usability floor
    (e.g. NASA score worse than the predict-mean floor by less than a margin) — a "win" where everything
    fails does not count as a success condition.
- `success_map(results_glob, metric) -> table` — one row per (dataset, model, n_units, factor-level):
  verdict + margin + p. This is the headline deliverable object; `plots.py` renders it.

**New `Config` fields:**
```python
win_margin: float = 0.0            # DECISION (uncited): min seed-mean improvement to call a win (metric units)
win_alpha: float = 0.05            # paired-seed significance threshold (descriptive; low-powered at 5 seeds)
usability_floor_metric: str = "nasa_clipped"  # the floor-guard metric
```
None are cache-key fields (they score existing CSVs, never re-embed). Tests: synthetic CSVs with hand-set
means exercise win/tie/loss/hollow and the strongest-baseline selection.

### 4.3 Earliness — "too early is also bad" (extend `evaluate.py` + `horizon.py`)

`RESEARCH_PLAN §8`. NASA score stays the scalar win metric (punishes lateness); add a two-sided layer:
- `earliness_histogram(y_true, y_pred, edges) -> dict` — fraction **dangerously late** (`pred − true` in
  bins below 0, i.e. underestimated RUL... note sign: late prediction = predicted RUL smaller than actual
  when the model says "more life left than there is" — follow the existing `nasa_score` convention where
  `d = pred − true`, `d ≥ 0` is the penalized "late" side) vs **wastefully early**. Reuse the horizon
  `bias` sign convention already documented in `CHANGES.md §16`.
- `cost_curve(y_true, y_pred, ratios) -> {ratio: cost}` — sweep early-cost:late-cost ∈ e.g.
  `[1:1 … 1:100]`; cost = Σ early_cost·max(0, over) + late_cost·max(0, under). No single arbitrary ratio;
  the curve is the result.

**New `Config` fields:** `earliness_bin_edges`, `cost_ratios` (both with defaulted factories; not cache
keys). Emit `earliness.csv` / `cost_curve.csv` alongside `horizon.csv`. Tests: closed-form checks on tiny
arrays (a known over/under split gives a known cost).

### 4.4 The factor-probe harness (`src/probes.py`, new) + interventions

The engine for every Tier-2 chapter. A probe = "sweep one factor on anchor datasets with the reduced
roster (top-2 TSFMs + top-2 cheap foils + best NN), score with the win-rule, emit success-map rows."

- `run_factor_probe(config, factor, levels, models, baselines, device, embedder_factory=None) -> Path`
  — for each level: apply the intervention (below), build the Stage-A cache (idempotent), run the head +
  reduced baselines at the ablation-winner shape, append to `probe_<factor>.csv` keyed by
  `(dataset, model, factor, level, n_units, seed, loss)`. Restartable; reuses `run_sweep` internals.

**Interventions (additive/subtractive per `RESEARCH_PLAN §1`; perturbative sim-only):**
- **Channel selection (subtractive, RQ-C):** `channel_subset` — a named subset of `sensor_columns`.
  Already expressible via `sensor_columns`; the probe iterates subsets. Subtractive → part of the window
  key already (`sensor_columns` is in `_window_key_fields`). No perturbation of kept values.
- **Sampling / aggregation (subtractive, RQ-G):** `aggregation_stride` / coarser per-cycle stats —
  applies where sub-cycle data exists (N-CMAPSS, XJTU, MetroPT). New loader-level knob (added per dataset
  in Phase B); the probe sweeps it. Part of the window key.
- **Noise tolerance (perturbative, SIM ONLY, RQ-H):** new `Config` field
  ```python
  # Controlled degradation of SIMULATED sensor readings to map the noise-tolerance
  # frontier (RESEARCH_PLAN §1). {} = off. Applied in data.load_prepared AFTER labels,
  # BEFORE windowing. RAISES if config.dataset is a REAL dataset (XJTU/MetroPT/Hydraulic/
  # Backblaze) — perturbation of real readings is out of scope by design.
  noise_injection: dict = field(default_factory=dict)  # e.g. {"kind":"gaussian","snr_db":20}
  ```
  Add to `_window_key_fields()` **only when non-empty** (existing keys unchanged). Enforce the sim-only
  guard fail-loud. Kinds: `gaussian` (additive at SNR), `drift` (slow bias ramp), `dropout` (random
  channel-blanking). Tests: guard raises on a real dataset; a fixed seed + SNR reproduces the same
  perturbed series; the key changes only when set.

**Reduced roster resolution:** `probe_roster(results_glob) -> (top2_tsfm, top2_foil, best_nn)` picks the
Tier-2 roster from the Tier-1 success map (best performers), so probes automatically use the strongest
comparators. Deterministic; tested on synthetic CSVs.

### 4.5 Zero-shot health-index forecasting (`src/zeroshot.py`, new) — RQ-Z

The **0-failures endpoint** of RQ-B: no head, no training. Use a TSFM's **forecasting** mode to predict a
health index forward and derive RUL from a threshold crossing.
- `run_zeroshot(config, device, forecaster_factory=None) -> Path` — for each test unit, form a health
  index (e.g. the first PCA component of normalized sensors, or the mean of monotone-trending channels;
  a `# DECISION (uncited):` documents the index), forecast it with the TSFM (Chronos-2/Moirai/TimesFM
  native forecasting; MOMENT via its forecasting head), find the horizon at which it crosses a failure
  threshold → predicted RUL. Score with the standard metrics + win-rule vs the `predict_mean` and
  `cycle_reg` floors. Writes `zeroshot.csv`.
- `forecaster_factory` is the CPU-test injection seam (a mock forecaster returning a fixed trajectory),
  mirroring `embedder_factory`.

Tests: mock forecaster → threshold-crossing logic yields the expected RUL on a constructed trajectory;
metrics finite; restartable.

### 4.6 Reporting (extend `src/plots.py`)

- `plot_success_map(success_table, ...)` — the headline figure: a win/tie/loss/hollow heatmap, models ×
  conditions (data-volume, factor-levels), faceted per dataset. Colors follow the `dataviz` skill palette.
- `plot_earliness(earliness_csv, ...)` and `plot_cost_curve(cost_curve_csv, ...)`.
- `plot_cross_tsfm(results_glob, ...)` — the five-model comparison (native vs common-representation).
- Keep the existing `plot_data_scaling` / `plot_horizon`; extend faceting to carry the model tag.

All figures: `prefix=config.result_prefix()`, saved under `config.figures_dir()`, `show=False` in the
campaign. Tests assert files are produced from tiny synthetic CSVs (no display).

---

## 5. Milestone 2 — C-MAPSS full-depth vertical slice (the template experiment)

A single orchestrator that runs **every C-MAPSS-answerable chapter** end-to-end across all five TSFMs and
produces the complete answer set. This is the "prove it works perfectly" gate before scaling.

`src/campaign.py` — add `run_full_depth(base_config, dataset="FD001", device=...)` (or a notebook stage)
that, for FD001–FD004:
1. **RQ-M / Tier-1 core:** `run_campaign` over `{chronos-2, moirai-2, moment, timesfm-2.5, ttm}` + foils
   (add `catch22_gbm` to `baselines.py`, §5.x) → per-model `results_v2.csv`.
2. **RQ-A history:** `run_ablation` context sweep (already exists) per top TSFM.
3. **RQ-B data-efficiency:** the unit-count sweep (exists) + the zero-shot endpoint (`run_zeroshot`).
4. **RQ-C channels:** `run_factor_probe(factor="channels", levels=<subsets>)`.
5. **RQ-E labeling:** MSE vs CORN arms (exist) + a `max_rul ∈ {125, 200}` label-cap arm (the horizon
   raised-cap machinery already exists, `CHANGES.md §18`).
6. **RQ-H noise (sim-only):** `run_factor_probe(factor="noise", levels=<snr grid>)`.
7. **Scoring + reporting:** `scoring.success_map` + `plot_success_map` + earliness/cost + cross-TSFM.

`catch22_gbm` baseline (add to `baselines.py` + `BASELINES` registry): `pycatch22` features → LightGBM,
same `Baseline` interface, lazy import. The "hand-crafted indicators" foil for RQ-D and the cheap-model
question. Tested with `importorskip("pycatch22")` like the gbm/minirocket tests.

**Milestone-2 acceptance (all must pass):**
- `pytest -q --cov=src --cov-branch` → **100%**, CPU-only, no downloads.
- `grep -rn "DECISION (uncited):" src/` lists every new judgment call named in §4–§5.
- FD001 window/embedding cache keys **byte-identical to `main`** (stable-key test green).
- A synthetic-data `run_full_depth` (mock embedders/forecasters) produces: per-model results CSVs, the
  success-map table + figure, earliness + cost curves, the cross-TSFM comparison, the zero-shot CSV — i.e.
  it answers RQ-A/B/C/E/H/M/Z on synthetic C-MAPSS with zero GPU.
- **User, on Colab (real Chronos-2 + real backbones as they pass their spikes):** `run_full_depth`
  reproduces the recorded FD001 Chronos-2 sanity (`clipped RMSE ≈ 10.7`, `CHANGES.md §12`) and every new
  artifact renders.

---

## 6. Milestones 3–7 — Aggressive dataset scale-up

Each milestone = a loader (or loader extension) + the chapter(s) unique to that dataset, following the
`DATASET_EXPANSION_PLAN.md` template (synthetic fixture, canonical frame, cache-key discipline,
non-comparability warning, CHANGES section). Datasets already loaded (XJTU, N-CMAPSS) get **extensions**,
not new loaders.

### 6.1 Milestone 3 — XJTU: raw-vs-indicators (RQ-D)

XJTU has raw 25.6 kHz waveforms; today `datasets/xjtu.py` emits only 16 indicator channels. Add a
**feature-mode knob** so the SAME bearings can be modeled as raw (downsampled) channels, indicators, or
both — the direct test of "do TSFMs make hand-crafted indicators obsolete."
```python
xjtu_feature_mode: str = "indicators"   # {"indicators","raw","raw+indicators"}; DECISION (uncited)
```
- `raw` = per-snapshot downsampled waveform (a fixed decimation of the 32768 samples → N channels), a
  subtractive/aggregation choice (never mutating kept samples). `raw+indicators` = concat.
- Part of the window key **only when `!= "indicators"`** (existing XJTU keys unchanged).
- `DEFAULT_SENSOR_COLUMNS`/loader resolve the channel set per mode.
- Probe: `run_factor_probe(factor="feature_mode", levels=["raw","indicators","raw+indicators"])` on XJTU
  (+ MetroPT once available), scored with the win-rule → the RQ-D chapter.
- Tests: synthetic XJTU already exists; add cases for each mode's channel count + key behavior + a probe
  smoke test.

### 6.2 Milestone 4 — N-CMAPSS: sampling/aggregation (RQ-G) + high-data (RQ-B)

N-CMAPSS is 1 Hz within flights; today it aggregates to per-cycle mean/std + `cycle_len_s`. Add an
**aggregation-granularity knob** to sweep RQ-G:
```python
ncmapss_agg_stride: int = 1   # sub-sample flights 1/stride before aggregation; DECISION (uncited)
# and/or a richer stat set toggle: {"mean_std","mean_std_minmax_slope"}
ncmapss_agg_stats: str = "mean_std"
```
- Bump `NCMAPSS_AGG_VERSION` when stats change; add the knobs to the ncmapss window key (ncmapss-only,
  as `ncmapss_test_truncation` already is). DSALL (high-data arm) already exists.
- Probe: `factor="aggregation"` on N-CMAPSS. `factor="channels"` (RQ-C) at N-CMAPSS's many channels.
- Tests: extend the synthetic-h5 fixture cases for stride/stats + key behavior; probe smoke test.

### 6.3 Milestone 5 — MetroPT-3 (`datasets/metropt.py`, new): alarms, lead-time, censoring

Real Porto-Metro APU, 15 analog+digital signals at 1 Hz, unlabeled + company failure reports (air-leak
events). **This is the first censored, adapted-label dataset — build the censoring machinery here.**

- **Loader** into the canonical frame: resample/aggregate 1 Hz → per-window cycles; unit = a run between
  interventions (clock resets at each documented failure event, `RESEARCH_PLAN §4`); channels = the 15
  signals (+ optional indicators). Failure-report timestamps → intervention labels. `# DECISION (uncited):`
  documents the exact windowing, the intervention-reset RUL derivation, and the alarm lead-time target.
- **Censoring machinery (reused by Backblaze):** most windows are far-from-failure/healthy. Add a
  **time-to-event target + lead-time metric** rather than forcing RUL on censored survivors:
  ```python
  # Censoring-aware target: predict P(failure within `alarm_horizon` cycles). Enables
  # a mostly-healthy fleet to train (RESEARCH_PLAN §4). RUL spine still used for the
  # failed runs. DECISION (uncited): horizon + labeling recorded per dataset.
  alarm_horizon: Optional[int] = None      # None => pure RUL (run-to-failure datasets)
  ```
  New loss arm `failure_within_horizon` (binary, reuses `heads.MLPHead` with a 1-logit output + BCE) and a
  **lead-time metric** in `evaluate.py`: precision/recall at the alarm horizon + the cost curve (§4.3).
  Ordinal/CORN connects here as the censored-label tool (Vishnu et al.). Metrics for censored datasets are
  **never tabled against NASA scores** (non-comparability, like XJTU/N-CMAPSS).
- **Fault types** → the RQ-F secondary-label plumbing (§6.4).
- **Synthetic fixture** `write_synthetic_metropt(...)`: a few units with injected leak events + healthy
  survivors, so the loader, censoring target, and lead-time metric are 100%-tested on CPU.
- `is_available` + `DATASET_LOADERS`/`DATASET_FAMILIES` registration; `DEFAULT_DATASET_OVERRIDES` entry
  (alarm_horizon, window/max_rul choices).

### 6.4 Milestone 6 — UCI Hydraulic (`datasets/hydraulic.py`, new): adjustment vs replacement (RQ-F)

Real rig, 60 s cycles, ~17 sensors at mixed rates, four components each with **graded severity labels per
cycle** — the native adjust→replace gradient.

- **Loader:** per-cycle aggregation of the multi-rate sensors into the canonical frame; the component
  severity annotations become the **secondary label** (per `RESEARCH_PLAN §4`). Because the rig is cyclic
  controlled-fault-injection (not smooth run-to-failure), its RUL role is limited — it is primarily the
  **RQ-F anchor**. `# DECISION (uncited):` records the severity→{adjust, replace} thresholding.
- **The RQ-F probe (`src/taxonomy.py`, new):** a **few-shot classification on frozen embeddings** —
  `run_taxonomy_probe(config, secondary_label, shots, device, embedder_factory=None)`: from the Stage-A
  embedding cache, train a light classifier (logistic / kNN) with `k` labeled examples per class, measure
  separability (accuracy / macro-F1 / AUROC) of adjustment-vs-replacement, and compare **TSFM embeddings
  vs catch22 indicators** as the feature source. Reuses the embedding cache (no new backbone work). Writes
  `taxonomy.csv`. Anchors: Hydraulic (native severity) + MetroPT (fault types).
- **Synthetic fixture** + tests: separable synthetic severity classes → the probe recovers them; the
  indicator-vs-embedding comparison runs; few-shot `k` sweep restartable.

### 6.5 Milestone 7 — Backblaze (`datasets/backblaze.py`, new): censored fleet-scale

The real-world C-MAPSS-scale ideal case. Daily SMART snapshots across a large multi-model drive fleet;
mostly-healthy (right-censored), rare failures. **Real engineering, not a drop-in.**

- **Scope control:** `# DECISION (uncited):` restrict to a small set of high-volume drive models
  (config field `backblaze_models: list`), a chosen date range, and a curated SMART attribute set (SMART
  availability varies by model → this is itself the RQ-C "what to record" question at fleet scale).
- **Censoring-aware protocol (reuses M5 machinery):** failed drives → RUL-to-failure (clock from first
  observation); healthy drives → right-censored survivors feeding the `failure_within_horizon` target.
  Heavy class imbalance → the lead-time/precision-recall + cost-curve metrics (never NASA-tabled).
- **Scale:** load via `pyarrow`; a **parsed-frame cache** exactly like N-CMAPSS (`backblaze_agg_v<N>.npz`,
  location-independent, version-bumped on logic change) so the multi-GB parse happens once. Downsample/
  aggregate to per-window cycles.
- **The data-efficiency headline at real scale (RQ-B):** "how many **failure events** before deploying" —
  the unit-count grid expressed in *failures observed*, the censoring twist that C-MAPSS can't provide.
- **Synthetic fixture** `write_synthetic_backblaze(...)`: a few failed + many censored synthetic drives
  (small) so the loader, censoring target, imbalance handling, and metrics are 100%-tested on CPU. Real
  data is validated by a user-run sanity gate, never a unit test.
- Registration + `DEFAULT_DATASET_OVERRIDES` (models, horizon, metrics).

---

## 7. Testing strategy for 100% coverage (the gate)

- **CPU-only, no GPU, no downloads.** Every backbone is mock-injected (`embedder_factory` /
  `forecaster_factory` seams already exist in `run_ablation`, `run_campaign`, `run_transfer_eval`; the new
  runners must expose the same seam). Every dataset has a synthetic fixture in `tests/synthetic.py`
  (C-MAPSS, XJTU, N-CMAPSS exist; add MetroPT, Hydraulic, Backblaze).
- **Coverage config (`.coveragerc`):** `--cov=src --cov-branch`, `fail_under = 100`. The **only** allowed
  `# pragma: no cover` is the lazy heavy-import line inside each `_load_pipeline`/loader-library import
  (the same boundary that keeps `chronos` out of tests today). Every branch above it is covered by mocks.
- **What each new module's tests must cover:** happy path + every fail-loud branch (shape/name/registry
  drift, sim-only noise guard on real datasets, censoring-target guard on run-to-failure datasets,
  schema-drift guards), restart-safety (rerun → row count unchanged), cache-key behavior (new field
  changes the key only when non-default; FD001 key unchanged), and the scoring/metric closed-form checks.
- **Determinism:** seed every fixture and probe; `torch.use_deterministic_algorithms(warn_only=True)` path
  already threaded via `train.set_seed`.
- **Registry-drift alarms:** extend the existing datasets drift test with the new families; add the models
  drift test (§4.1).
- Run `pytest -q` after **every** task (all existing 48+ tests must stay green — never renumber or weaken
  them; the two 8-unit smoke/fairness tests already account for the auto-appended full-fleet cell).

---

## 8. `CHANGES.md` sections to append (next free = §32; append, never edit)

- **§32** Coverage gate + the `# pragma: no cover` lazy-import policy (M0).
- **§33** Four new TSFM embedders (Moirai-2, MOMENT, TimesFM 2.5, TTM): the semantic-pooling contract per
  layout, per-channel loc/scale for univariate models, the fallback-to-hidden-states policy, and
  `channel_aggregation` (conditional key inclusion; FD001 keys unchanged).
- **§34** Cross-TSFM representation-fairness run (native vs common representation) — RQ-M.
- **§35** Scoring & win-rule: NASA-primary win/tie/loss/hollow vs strongest-per-cell baseline, paired-seed
  test generalization, absolute-floor guard, the success-map object.
- **§36** Earliness layer: earliness histograms + cost curve (sign convention tied to §16 `bias`).
- **§37** Factor-probe harness + interventions: channel-subset (RQ-C), aggregation stride/stats (RQ-G),
  sim-only `noise_injection` with the real-dataset guard (RQ-H); reduced-roster selection.
- **§38** Zero-shot health-index forecasting arm (RQ-Z) + the `forecaster_factory` seam.
- **§39** XJTU `xjtu_feature_mode` raw-vs-indicators (RQ-D; conditional key).
- **§40** N-CMAPSS aggregation knobs (RQ-G; ncmapss-only key; `NCMAPSS_AGG_VERSION` bump policy).
- **§41** MetroPT-3 loader + the censoring machinery: intervention-reset RUL, `alarm_horizon` +
  `failure_within_horizon` target, lead-time metric, non-comparability warning.
- **§42** UCI Hydraulic loader + the RQ-F few-shot taxonomy probe (`src/taxonomy.py`).
- **§43** Backblaze loader: scope control, censored fleet-scale protocol, parsed-frame cache, imbalance
  handling, "failures-before-deploy" framing.
- **§44** `catch22_gbm` baseline (the hand-crafted-indicator foil).

`README.md`: extend the layout block (new `src/` modules + `datasets/` loaders), the Data/ tree (MetroPT/
Hydraulic/Backblaze subfolders + where each is dropped), the model roster, and the campaign description.

---

## 9. Global acceptance checklist

1. `pip install -r requirements.txt && pytest -q --cov=src --cov-branch` → **100%**, CPU-only, no downloads.
2. `grep -rn "DECISION (uncited):" src/` — every judgment call in §4–§6 is present and audited.
3. Stable-key test green: `Config(dataset="FD001")` window + embedding keys **identical to `main`**; every
   new key field is conditional-on-non-default.
4. Registry-drift tests green for both `datasets/` and `models/` (every served name registered; vice-versa).
5. `run_full_depth` on synthetic C-MAPSS (mock backbones) answers RQ-A/B/C/E/H/M/Z end-to-end and emits the
   success map, earliness/cost, cross-TSFM, and zero-shot artifacts.
6. `run_campaign` on the repo as-is (only C-MAPSS present) runs C-MAPSS combos and reports
   `skipped_no_data` (with the documented Data/ path) for every not-downloaded dataset — never crashes.
7. **User, Colab:** each backbone passes its integration spike (finite FD001 parity) before joining;
   MetroPT/Hydraulic/Backblaze loaders reproduce a spot-checked unit against source; censored metrics and
   the success map render.

---

## 10. Explicit non-goals (do not do these)

- No multi-task joint model in v2 (RUL + type + lead-time jointly) — the RQ-F probe is few-shot on frozen
  embeddings; multi-task is a deliberate later phase (`RESEARCH_PLAN §4`), not this build.
- No perturbation of **real** sensor readings (sim-only `noise_injection`, guarded fail-loud).
- No raw sub-cycle / waveform *deep* modeling beyond the RQ-D downsampled-raw XJTU channels and the RQ-G
  aggregation sweep — the pipeline stays cycle/window-level.
- No changes to recorded FD001 winners, existing cache keys, or CSV schemas beyond the additive, guarded
  extensions above (`RESULTS_SCHEMA_VERSION` stays 2 unless a genuinely new *column set* is required, in
  which case bump + `ensure_csv_schema` guards it).
- No result numbers/claims written anywhere; no experiment-tracking services; no CLI framework.
- Do not commit any dataset files (`.gitignore` already covers `Data/` except `CMAPSSData/`).
```
