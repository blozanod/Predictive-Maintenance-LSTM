# CHANGES — deviations & protocol interpretations

This file records where the implementation makes a choice that is not spelled out
verbatim in `RESEARCH_PLAN.md`, so every judgment call is auditable. Uncited
design decisions are additionally tagged `# DECISION (uncited):` in code:

```bash
grep -rn --include='*.py' "DECISION (uncited):" src/
```

## 1. Official `Chronos2Pipeline.embed()` (not `BaseChronosPipeline`)
The WIP notebook loaded `BaseChronosPipeline`; `embeddings.py` uses the official
`chronos.Chronos2Pipeline.embed()` (chronos-forecasting 2.x), which returns, per
window, `(n_variates, num_patches+2, d_model)`. This matches the plan's intent
(§1) and Task 2.1 (reuse the maintained reference implementation).

## 2. No fitted scaler on the TSFM path → single-pass embedding cache
Chronos-2 instance-normalizes each series internally (`embed()` returns per-series
loc/scale), so the embedding path needs no cross-unit scaler and its embeddings —
**and the loc/scale it now caches** — are **independent of the data fraction**.
Consequently they are computed once over *all* FD001 train units (Stage A) and
cached. The no-leakage rule (Task 2.4, plan §6) is enforced where a scaler is
actually fit: the **baselines** (`data.fit_channel_scaler`, per-fraction train
windows) and the **head-feature standardizer** for the appended loc/scale and
raw-last columns (`features.HeadFeatureBuilder`, fit on the current fraction's
train-split rows only — §9).

**Update (loc/scale is now used, not discarded).** The prior implementation threw
away `embed()`'s loc/scale. Because Chronos-2 normalizes each 30-cycle window, the
slow C-MAPSS degradation-level signal lived *only* in that discarded loc/scale — the
dominant cause of the 17.4-RMSE regression. It is now cached per window as
`(n_windows, n_variates, 2)` and optionally fused into the head input (§9).

## 3. Sensor selection is a fixed a-priori list, not fit per fraction
Task 2.4 says "any sensor selection fit on the training split of that data fraction
only." We interpret **constant-sensor removal** as a property of the FD001 sensor
set (a-priori, community convention — Li et al. 2018), *not* a data-driven selection,
so it leaks no fraction-specific information and preserves the single-pass cache.
The 14 non-constant sensors are `config.sensor_columns` (see `config.py`), fully
overridable (`ALL_COLUMNS[2:]` for all 24). If genuinely *data-driven* selection is
wanted later, it must be fit per fraction — a documented hook, not implemented now.

## 4. Loss arms apply to the TSFM head; baselines are native regressors
The plan's grid notation "{Chronos-2, MiniRocket, GBM, CNN, LSTM} × {MSE, CORN}"
(§7 Phase 2) is ambiguous: GBM / MiniRocket+ridge / predict-mean have no ordinal
form. We apply the loss arms (`mse`, `corn`, optional `quantile`) to the **MLP head
on embeddings** — the ordinal-×-TSFM contribution the plan calls novel (§1, §5) —
and run baselines in their native regression form (results `loss` column =
`"native"`). CNN/LSTM could take CORN through the same `heads.py` machinery; left
un-wired to respect scope discipline (Task 2.6). Revisit if the paper needs
ordinal from-scratch NNs.

## 5. Both test-label protocols are ALWAYS reported (no toggle)
The `clip_test_labels` config flag is **removed**. `evaluate.evaluate_predictions`
now computes, from the unclipped ground truth, BOTH protocols as separate columns:
- `rmse_clipped` / `mae_clipped` / `nasa_clipped` — ground-truth RUL clipped at
  `max_rul` (predictions are already in `[0, max_rul]`). These are the
  **literature-comparable** numbers and the ones the sanity gate (§10) uses.
- `rmse_unclipped` / `mae_unclipped` / `nasa_unclipped` — against the raw
  `RUL_FDxxx.txt` PHM08 target. Inflated because 11/100 FD001 test units have true
  RUL > 125 that the clipped-trained head cannot reach (irreducible floor ≈ 3.7
  RMSE, ≈ +0.6 in quadrature). Reporting both makes that inflation explicit rather
  than a silent protocol choice.

Training labels are always clipped at `max_rul=125` (Heimes 2008; Li et al. 2018).
The cache stores the unclipped `actual_rul` as the test label; evaluate derives both.

## 6. Data fractions expressed as engine-unit counts
The notebook's row fractions `[0.1 … 1.0]` are replaced by unit counts
`data_unit_counts=[2, 5, 10, 25, 50, 100]` (plan §6): subsampling is **by unit**,
seeded, with the sampled unit IDs saved to the run directory.

## 7. Pooling: corrected token layout + content-only content poolings
**Correction of the earlier documentation error.** `embed()`'s per-window output is
`[content patches …, REG, masked-output/forecast patch]`: index **-1** is the masked
output/forecast token (a defensible CLS-like window summary), index **-2** is REG,
and content patches are `emb[:, :-2, :]`. The prior `CHANGES.md` §7 and the
`pool_window_embedding` comment described this wrongly (calling -1 a
"register/boundary token"), and `mean` averaged the 2 special tokens in with content.

Pooling options now (`POOLING_CHOICES`, part of the cache key):
- `forecast_token` (index -1) — the masked output patch; **default**
  (`# DECISION (uncited)`, a reasonable CLS-like summary; the ablation §9 chooses
  the empirical winner). This is the renamed old `last_patch`.
- `last_content` (index -3) — the last real content patch.
- `mean` — `emb[:, :-2, :].mean(1)`, content patches **only** (special tokens
  excluded; this was the bug).
- `flatten` — `emb[:, :-2, :]` flattened (content patches only). Valid only for
  fixed-length contexts (every window same patch count); excluded from the ablation.

## 8. Reproducibility & provenance (added per Task 2.3)
Seeds are threaded through python/numpy/torch/CUDA/DataLoader (`train.set_seed`);
`torch.use_deterministic_algorithms(warn_only=True)` is enabled when
`deterministic=True`. Every sweep writes `run_metadata.json` (resolved config +
git commit/describe/dirty + package versions) beside its metrics CSV, and per-cell
sampled unit IDs to `units_n{N}_seed{S}.json`.

## 9. Head-feature composition + leakage-safe fusion (Task 1.1)
`head_features ∈ {emb, emb+locscale, emb+locscale+raw}` selects, at Stage B, which
cached signals feed the head (`features.HeadFeatureBuilder`). It does **not** change
the embedding cache — loc/scale and the fixed raw windows are always cached, so all
three arms share one Stage A pass per (context, pooling).
- `emb` — pooled embedding only.
- `emb+locscale` — + the flattened per-window loc/scale (`2·n_variates` columns),
  standardized. This restores the degradation-level signal (§2).
- `emb+locscale+raw` — + the window's last-cycle raw sensors (`n_channels` columns),
  standardized (Wide & Deep-lite, mirrors the PHM 10.32 paper).

**Leakage rule.** The embedding block passes through unstandardized (as before). The
**appended** columns (loc/scale, raw-last) are standardized with statistics fit on
the **current data fraction's train-split rows only**, then applied to train/val/
test. The last-cycle raw sensors come from `train_windows[:, -1, :]` (fixed windows
front-pad short units, so `[-1]` is always the true prediction cycle, aligned to the
TSFM context's last cycle).

## 10. Variable-length TSFM context, independent of the baseline window (Task 1.2)
`tsfm_context_length` (default None → `window_size`) sets how much history the TSFM
sees, decoupled from the baseline `window_size` (kept at 30). The TSFM path feeds
`embed()`'s **native variable-length list input** instead of fixed padded windows:
- Train: `data.make_windows_varlen` emits one context per prediction cycle for the
  **same** cycles as `make_windows` (window_size..n), each the last
  `min(c, tsfm_context_length)` real cycles. Labels and unit_ids are **identical**
  to the fixed path (asserted at cache-build time and in tests), so the head trains
  on exactly the rows the baselines do.
- Test: `data.make_test_last_contexts` emits one context per unit = the last
  `min(n, tsfm_context_length)` real cycles, **never padded**. This removes the
  repeat-first-cycle fabrication (the old `pad_short` would invent up to ~89 cycles
  for the 37/100 FD001 test units shorter than 120, corrupting instance-norm stats).
  Short histories are left-pad-**masked** inside `embed()`. Baselines keep fixed,
  front-padded window-30 windows.

The **sanity gate** (Task 1.6): after these fixes, full-data FD001 Chronos-2+MLP
should reach clipped RMSE ≤ ~14 (≤ ~12.5 with the raw-fusion arm). If it does not,
STOP and write up observations (learning curves, pred-vs-truth scatter, per-unit
errors) — no silent hyperparameter fishing.

## 11. Cache schema v2: loc/scale + variable-length embeddings + fp16 storage
`CACHE_SCHEMA_VERSION = 2` is part of `embedding_cache_key()`, so every pre-fix cache
(per-window-normalized embeddings, no loc/scale) is invalidated. The key also now
includes `tsfm_context_length`. Cache contents: pooled embeddings, per-window
loc/scale `(N, n_variates, 2)`, fixed raw windows, labels, unit IDs (train + test).
- **`# DECISION (uncited)`**: embeddings are stored **float16** on disk
  (`embedding_storage_dtype`, default `float16`) and upcast to float32 on load;
  `np.savez` **uncompressed** (`cache_compressed=False`). This roughly halves the
  ~700 MB Drive write and skips the slow compressor. The **RMSE effect of fp16 must
  be measured at full data** (expected negligible); if it is not negligible, set
  `embedding_storage_dtype='float32'` and note it here. Raw windows and loc/scale
  stay float32. embed()'s *compute* dtype (`embed_dtype`, bf16) is unchanged.

## 12. Ablation → winner selection, then the full sweep (Task 1.5)
`sweep.run_ablation` runs full-data, MSE, 3 seeds over
`tsfm_context_length ∈ {30,60,120,256} × head_features ∈ {emb, emb+locscale}` at the
default pooling, then adds `{best context} × emb+locscale+raw` and the pooling
variants `{mean, last_content}` at the best `(context, features)` cell. It builds one
Stage A cache per `(context, pooling)` (idempotent) and checkpoints every row to
`ablation.csv` (restartable). `select_best_ablation_cell` picks the winning
`(context, features, pooling)` by **seed-mean `rmse_clipped`** (the
literature-comparable metric). The notebook adopts the winner and reruns the full
data-fraction × loss × seed sweep + baselines at it (`run_sweep`).

**Winner (recorded from the July 2026 Colab run, `ablation.csv`).**
`tsfm_context_length=256, head_features=emb+locscale, pooling=mean`: clipped RMSE
**10.81 ± 0.66** (3 seeds), vs 10.92 ± 0.14 for `forecast_token` at the same cell,
11.06 ± 0.58 for `emb+locscale+raw`, and 13.18 ± 0.32 for the old context-30 cell.
Context length dominates (16.4 → 13.2 emb-only from 30 → 60+; a further −1.2 from
120 → 256 with locscale); locscale fusion is worth ~1–3 RMSE at every context.
The full sweep at this winner (`results_v2.csv`) passes both sanity gates:
full-data clipped RMSE 10.66 ± 0.51 (mse arm, 5 seeds).

**Interpretation caveats on "context 256" (recorded, not yet acted on):**
1. Contexts are truncated to available history and never padded (§10), so no data
   is fabricated — but only 17/100 FD001 *train* units have ≥256 cycles (median
   199) and 1/100 *test* units do (median 134). "256" therefore effectively means
   "**all available history**"; the ablation grid is really {30, 60, 120, ~full}.
   A finer grid (e.g. {80, 190=median, full}) would be needed to claim an optimum.
2. With truncate-to-history contexts, context *length* correlates with elapsed
   cycles, itself predictive of RUL. Part of the long-context gain may be the head
   reading "engine age" out of the embedding — legitimate at deployment (age is
   always known) but the baselines don't get an elapsed-cycles feature. Fairness
   follow-ups: (a) run the plan §4 "linear regression on cycle count" floor,
   (b) give GBM an elapsed-cycles feature, (c) run
   `run_baseline_window_comparison` (implemented, §14) so baselines also get a
   long-history variant. Until then, cross-model comparisons at context 256 favor
   the TSFM's information set, not necessarily its representations.

## 13. On-GPU head training; embedding-pass cuDNN autotune (Task 2)
`train.train_head` moves features/labels to the device once and minibatches with a
seeded on-device permutation (`torch.randperm` + a device `Generator`) — no
DataLoader, no workers, no per-batch host↔device copies. The sweep moves the whole
cache to GPU tensors once and slices per cell. Determinism is preserved
(`use_deterministic_algorithms(warn_only=True)`, `cudnn.benchmark=False` for heads).
The embedding pass is inference-only and cached once, so `cudnn.benchmark=True` is
enabled there only (it never touches the seeded training path). Stage A logs and
sidecars throughput (windows/s); pooling is done on-device per batch so only the
small pooled vectors are transferred to host.

## 14. Baselines: parallelism + optional per-family window (Task 1.5, Task 2)
LightGBM runs `n_jobs=-1`; MiniRocket's transform runs `n_jobs=-1` (stays CPU/sktime);
CNN/LSTM already train on CUDA when available. `config.baseline_windows` (name→cycles,
default empty ⇒ all use `window_size`) lets a baseline family adopt a longer window
if `run_baseline_window_comparison` shows it helps at full data (equal-tuning-budget
fairness, plan §6). Override sizes are re-windowed from the raw series (loaded once);
padding a longer test window may fabricate cycles for short units — a known baseline
limitation (the TSFM path is padding-free), noted for provenance.

## 15. Results file: v2 schema + never overwrite v1 (Task 1.4)
The sweep writes `results_v2.csv` (default) with a `schema_version` column
(`RESULTS_SCHEMA_VERSION=2`), both-protocol metric columns, and the config axes
(`tsfm_context_length`, `head_features`, `pooling`, `baseline_window`). Any
pre-existing `results.csv` is archived to `results_v1.csv` before v2 writing begins
(`evaluate.archive_results_v1`, idempotent). Row keys are emitted in a fixed order so
the CSV columns stay aligned across TSFM and baseline rows.

## 16. Horizon-stratified evaluation (src/horizon.py)
The standard C-MAPSS protocol scores ONE prediction per test unit (final observed
cycle), which cannot answer "how good are far-from-failure predictions?" — the ones
that buy planning lead time. `build_horizon_cache` embeds EVERY test cycle
≥ `window_size` (the training-row context construction applied to test trajectories)
into a sidecar cache `horizon_<embedding_cache_key>.npz` — same key, so it
invalidates with the main cache but never touches it. `run_horizon_eval` trains the
standard arms at chosen unit counts and writes per-RUL-bin metrics (`horizon.csv`,
default bins {0–25, 25–50, 50–75, 75–100, 100–125, ≥125}) + per-cycle predictions
(`horizon_predictions.csv` for trajectory plots). Metrics per bin: RMSE/MAE vs the
clipped target, `bias` = mean(pred − clipped truth) (negative ⇒ conservative/early),
and `nasa_mean` (per-cycle mean PHM08 score; the raw sum is not comparable across
bins of different size).

**Protocol honesty:** (1) the ≥125 bin measures SATURATION quality only — with
training labels clipped at `max_rul`, no model here can express "fails in 180
cycles"; claims about horizons beyond 125 are impossible under this protocol.
(2) Raising `max_rul` is the real long-horizon experiment; it re-keys both caches
(labels are stored with the windows) and costs a fresh Stage A pass per value —
deliberate follow-up, not done silently. (3) Test units shorter than `window_size`
contribute no rows (none in FD001).

## 17. Cold-start transfer evaluation (src/transfer.py)
`run_transfer_eval` answers the day-one deployment question: head trained on a
SOURCE fleet, evaluated on a TARGET fleet's standard test protocol with 0..k target
failures. Arms (`transfer.csv` column `mode`): `zero_shot` (all source units, no
target data), `target_only` (k target units), `source+target` (all source + k
target). Decisions:
- **Statistics travel with the training rows.** The head-feature standardizer
  (loc/scale, raw-last) is fit on each arm's train rows only — source rows for
  zero-shot, so the target is scored under source statistics exactly as a day-one
  deployment would be. The TSFM path needs no other scaler (Chronos-2 instance-norm,
  §2); GBM's window-statistic features are likewise scaler-free.
- **From-scratch NN baselines (CNN/LSTM) are excluded by default** — they would
  need a cross-dataset scaler policy (fit-on-source vs fit-on-target is itself a
  research choice); add deliberately, not silently. Default baseline: GBM.
- **FD001↔FD003 is the default pair**, a-priori valid: both single-operating-
  condition with the same non-constant sensor set (§3). FD002/FD004 print a loud
  warning: condition-wise normalization (plan §6) is not implemented, so those
  numbers are exploratory only.
- **shots ≥ 2 enforced** (k=1 leaves no unit for the val split); the k-unit
  train/val split reuses `unit_train_val_split` exactly as the main sweep does.

## 18. Horizon follow-ups: 5 seeds, paired test, raised label cap (a-b)
- **Seeds.** `run_horizon_eval` now defaults to the FULL `sweep_seeds` (5, plan §6)
  instead of the first 3: the per-bin CORN-vs-MSE comparison became a headline
  claim and needs the seeds. Existing 3-seed rows stay valid (restartable cells).
- **Paired significance.** `evaluate.paired_seed_ttest` runs a paired-by-seed
  t-test per (max_rul, n_units, bin) cell. Pairing on seed is valid because both
  loss arms share each seed's sampled units and split. Zero-variance differences
  return nan rather than ±inf. With 5 seeds the test is low-powered — p-values are
  reported as descriptive support next to the per-bin means, never alone.
- **Raised label cap (the real long-horizon experiment).** The 125-cap runs are
  KEPT untouched (literature comparability). A second arm at `max_rul=200` is run
  afterwards: it re-keys both caches (labels are cached with the windows ⇒ a fresh
  Stage A pass) and shares `horizon.csv`/`horizon_predictions.csv` with the 125
  arm — `max_rul` joined `HORIZON_KEYS` and the predictions schema (new column).
  Bin edges are now `default_bin_edges(max_rul)` (25-cycle bins to the cap, then
  the ≥cap saturation bin), so edges BELOW 125 are identical across arms and
  directly comparable; the 125–200 bins exist only in the 200 arm and measure
  whether degradation is detectable that early at all.
- **Schema guard.** `evaluate.ensure_csv_schema` fails loudly when appending
  changed-schema rows to an old CSV (silent column misalignment otherwise). A
  pre-§18 `horizon_predictions.csv` (no `max_rul` column) must be moved/archived;
  `horizon.csv` is unchanged and keeps working.

## 19. Fairness arms: cycle-age floor + GBM-with-age (c)
`sweep.run_fairness_baselines` adds the two arms that bound the §12-caveat-2
age confound (the TSFM's variable-length context implicitly carries engine age;
baselines were never given it):
- `cycle_reg` — linear regression clipped-RUL ~ elapsed cycles (the plan §4
  "linear regression on cycle count" floor), fit per (n_units, seed) cell on the
  cell's train-split rows, predictions clipped to [0, max_rul]. Drawn as a floor
  reference line in the data-scaling figure.
- `gbm_age` — the UNMODIFIED GBM baseline whose windows are built with
  `time_cycles` as an extra leading channel, so `window_statistics` includes
  elapsed cycles (last value of that channel), its slope, etc. No new model code;
  the age signal enters through the standard feature path. Same known caveat as
  all fixed-window baselines (§14): front-padded short test units repeat the
  first cycle's `time_cycles`, but the LAST value (the true age at prediction
  time) is always real.
Rows append to the main `results_v2.csv` over the standard grid, so the
data-scaling figure includes them automatically. If `gbm_age` closes most of the
gap to the TSFM, the long-context advantage was age, not representation — that is
the honest test the caveat demanded. `run_baseline_window_comparison` (§14) is
now wired into the notebook (§4b) at windows {30, 60, 120}.

## 20. Horizon file-sync guard (bugfix)
`horizon.csv` (metrics) and `horizon_predictions.csv` (per-cycle predictions) are
two append-only files written together per cell but formerly gated on `horizon.csv`
alone. Archiving/deleting only ONE (e.g. the §18 note said to archive
`horizon_predictions.csv` before rerunning, but not `horizon.csv`) desynced them:
the kept `horizon.csv` marked seeds 0-2 "done", so those cells were skipped and
never re-emitted predictions, leaving `horizon_predictions.csv` with only the newly
run seeds. `plot_horizon_trajectories`'s default `seed=0` then found zero rows and
matplotlib raised the opaque "Number of columns must be a positive integer, not 0".

Two fixes:
- `run_horizon_eval` now gates skips on BOTH files (`done = metrics ∩ predictions`)
  and, if `horizon.csv` has cells whose predictions are missing, raises a clear
  error naming the desync and the remedy (archive/delete BOTH together) instead of
  silently producing an incomplete predictions file.
- `plot_horizon_trajectories` selects an AVAILABLE `(n_units, seed)` from the file
  rather than assuming `seed=0`/max exist; if the requested seed is absent it falls
  back to a present one with a printed note, and raises a clear message only when a
  unit count is genuinely absent.

## 21. Multi-dataset support: condition-wise normalization + one loading path
The breadth arm (plan §7 Phase 4) starts here. Changes:
- **One loading path.** `data.load_prepared(config)` is now the ONLY way any
  pipeline stage (Stage A caches, horizon cache, baselines, window comparison,
  fairness arms) obtains data: it loads the dataset, attaches RUL labels, and
  applies condition-wise normalization when resolved ON — no stage can disagree
  about preprocessing.
- **Condition-wise normalization (plan §6).** Rows are grouped by their discrete
  operating point — the 3 settings snapped onto their grid
  (`CONDITION_SETTING_DECIMALS` = (0, 2, 0) decimals) — and each sensor channel is
  z-normalized per condition. The scaler is keyed by the setting VALUES, not
  per-frame ranks, so train/test rows at the same operating point always share
  statistics even if one frame is missing a condition; unseen test conditions
  fall back to global train stats. Channels flat within a condition get std=1
  (they normalize to ~0; this is why the FD001 14-sensor list stays valid for
  FD002/FD004 — the 7 dropped sensors move only WITH the condition).
- **`condition_norm` config flag**, None ⇒ auto: ON for FD002/FD004 and XJTU-SY,
  OFF for FD001/FD003 (all earlier FD001 numbers remain produced by byte-identical
  preprocessing). Part of the cache key: **adding the field invalidates every
  pre-§21 Stage A cache** (one re-embed per dataset on first run after this).
- **Deliberate deviation:** normalization statistics are fit ONCE on the full
  train split, not per data fraction (plan §6 strictly read). Per-fraction stats
  would make the embedding cache fraction-dependent (~6× the Stage A GPU cost).
  Condition statistics are properties of the operating points (no labels
  involved), so the residual leakage is limited to sensor means/stds across
  train units — accepted and recorded. Test statistics are never used.
- **Multi-dataset restart keys.** `dataset` joined `CELL_KEYS`, `ABLATION_KEYS`,
  `HORIZON_KEYS`, the window-comparison keys, and the horizon predictions schema;
  `TRANSFER_KEYS` gained (source_dataset, target_dataset). Without this,
  switching `config.dataset` against the same CSVs marked every cell of the new
  dataset "done". Old metric CSVs already carry these columns and keep working;
  a pre-§21 `horizon_predictions.csv` (no `dataset` column) trips the §18 schema
  guard — archive it. Multi-dataset figures: `plot_horizon` emits one figure per
  (dataset, cap, n_units); `plot_horizon_trajectories` requires a `dataset=`
  selection when the predictions file mixes datasets (unit IDs collide).

## 22. XJTU-SY bearing loader (src/datasets/xjtu.py) — the non-CMAPSS stress test
Adapts the XJTU-SY run-to-failure bearing dataset (15 bearings, 3 operating
conditions, 25.6 kHz vibration snapshots once per minute; download:
https://biaowang.tech/xjtu-sy-bearing-datasets/) into the SAME canonical frame
C-MAPSS uses, so every downstream stage runs unchanged. Decisions (all
`DECISION (uncited)` — no community-standard protocol exists for XJTU RUL):
- One "cycle" = one 1-minute snapshot; one "unit" = one bearing; "sensors" =
  8 classic time-domain condition indicators per axis (`XJTU_FEATURE_COLUMNS`,
  16 channels: rms, kurtosis, skewness, peak, p2p, crest, impulse, shape),
  computed per snapshot — the standard indicator-trend formulation, not the raw
  waveform.
- `setting_1..3` = condition index / speed / radial force, so §21's condition
  normalization groups by operating condition exactly as for FD002/FD004
  (auto-ON).
- Split protocol: fixed held-out test bearings (`xjtu_test_bearings`, default
  the last 2 of 5 per condition) truncated at `xjtu_test_truncation` (default
  0.6) of life, mimicking the C-MAPSS "predict at last observed cycle" protocol;
  provided RUL = remaining minutes. Both fields are part of the cache key.
- `max_rul` is in MINUTES here; the FD-convention 125 is arbitrary for bearings
  (lifetimes span ~35 min to ~42 h) — choose per experiment and record it.

## 23. Source reorg: datasets/ + models/ registries, one Data/ root, named results
Structural cleanup only — **no numeric result, cache key, or CSV schema changes**;
all 48 CPU tests pass unchanged and every recorded run (§12) stays valid.
- **`src/datasets/` (one module per dataset family, behind a registry).** The raw
  loaders moved out of `data.py`/`src/xjtu.py` into `datasets/cmapss.py` (FD001–FD004)
  and `datasets/xjtu.py`; `datasets/__init__.load_raw` dispatches by
  `config.dataset_kind()`. `data.py` keeps the preprocessing hub + the unified
  `load_prepared` entry point (CHANGES §21) and **re-exports** `load_cmapss`/`load_xjtu`
  so `data.load_cmapss` stays valid. Adding N-CMAPSS is one new module + one registry
  entry.
- **`src/models/` (one module per TSFM, behind a registry).** `ChronosEmbedder` moved
  from `embeddings.py` to `models/chronos.py`; `models/make_embedder` selects the class
  for `config.model_name` (`EMBEDDERS` registry). This is the concrete realization of
  the TimesFM/MOMENT/TTM/Moirai slot-in point. `embeddings.py` keeps the model-agnostic
  cache/pooling/loc-scale plumbing and the injectable-embedder contract (tests still
  pass a mock). The specialized from-scratch models stay in `baselines.py` (the plan's
  foundation-vs-baseline split).
- **One `Data/` root for every dataset (`config.data_root`, default `Data`).** Each
  dataset declares its subdirectory (`CMAPSSData`, `XJTU-SY`); `datasets.resolve_data_dir`
  maps `data_root/<subdir>`, or honours an explicit `config.data_dir` override (unchanged
  test behaviour — tests set `data_dir`). The committed C-MAPSS files moved to
  `Data/CMAPSSData/`; `.gitignore` keeps them tracked and ignores other large datasets
  dropped under `Data/`. **`data_root`/`data_dir` are NOT in any cache key** (embeddings
  are location-independent).
- **Experiment-named result files (`config.experiment_name`).** Every result CSV,
  figure, and per-run bookkeeping dir is prefixed via `config.results_path(name)` /
  `config.result_prefix()` / `config.figures_dir()` (plots take a `prefix=`), e.g.
  `results/<exp>_results_v2.csv`, `results/figures/<exp>_data_scaling_rmse_clipped.png`,
  `results/<exp>_runs/`, so separate experiments never clobber each other. Default `""`
  reproduces the historical flat names byte-for-byte (why the tests are untouched).
  **Not in any cache key** — it names outputs only.

## 24. Run-all campaign, per-dataset sensor defaults, dataset-faceted figures
Follow-ups to the §23 reorg review (four fixes + the run-all button):
- **`plot_data_scaling` no longer pools datasets (bugfix).** Results CSVs may hold
  several datasets (§21 keys them into the sweep cells), but the aggregation
  grouped by (model, loss) only — two datasets under one experiment name silently
  averaged into one curve. It now facets: one figure per (dataset, metric), the
  dataset in the title (killing the hardcoded "FD001") and in the filename when
  the CSV holds more than one. `aggregate_data_scaling` gained a `dataset=` filter.
- **Per-dataset default sensor columns.** `sensor_columns=None` (the new default)
  resolves in `__post_init__` via `DEFAULT_SENSOR_COLUMNS[dataset_kind()]`
  (C-MAPSS → the FD001 14-sensor list, XJTU-SY → its 16 indicator channels), so
  switching datasets is one knob instead of a cryptic KeyError deep in
  preprocessing. The resolved defaults equal the previously-required explicit
  lists, so every cache key is unchanged (asserted in tests).
  `XJTU_FEATURE_COLUMNS` moved to config.py (re-exported by datasets/xjtu.py) to
  avoid an import cycle. An explicit list still wins and survives `replace()`.
- **Registry drift alarm.** `config.dataset_kind()` and the `datasets/` registry
  are cross-checked by a test (every served name maps to a registered family and
  vice versa) so adding N-CMAPSS can't silently miss one of the two.
- **`experiment_name` validation**: letters/digits/`._-` only — it lands in every
  result filename.
- **`src/campaign.py` — the Run-all button.** `run_campaign(base_config)` sweeps
  `datasets.all_dataset_names()` × `models.EMBEDDERS`; per combo it runs
  cache → sweep → fairness → horizon → figures (each restartable, so re-running
  resumes). Datasets missing from `Data/` are SKIPPED with a notice; a failing
  combo is reported with its traceback and the campaign continues, raising only
  when every combo failed. Each combo runs under experiment namespace
  `<dataset>_<model-tag>` (base `experiment_name` prepended when set), so every
  CSV/figure/run-dir filename states its dataset and TSFM, e.g.
  `results/FD002_chronos-2_results_v2.csv`. `dataset_overrides` carries
  per-dataset protocol choices (XJTU-SY needs deliberate `max_rul`/`window_size`
  — its cycles are minutes); `sensor_columns` always resolves to the dataset
  default inside the campaign (DECISION: a base-config list would silently be
  wrong for every other dataset — put custom channels in `dataset_overrides`).
  Baselines rerun per combo so each experiment file is self-contained for its
  figures (duplicate CPU work across models of one dataset — accepted).
- **Notebook**: campaign-first layout — "Run all" executes §3 (the campaign);
  the single-dataset deep-dives (ablation, learning curves, significance table,
  raised-cap arm, transfer) are gated behind `RUN_DEEP_DIVES=False` in the
  Config cell, which now carries the recorded §12 winner as its defaults.

## 25. XJTU-SY condition-3 folder/force fix + unmatched-folder guard
`XJTU_CONDITIONS` mapped condition 3 to `"40Hz12kN"` at 12 kN. Per the dataset
documentation (Wang et al. 2020, Table 2) condition 3 is **2400 rpm (40 Hz) / 10 kN**,
shipped in a folder literally named `40Hz10kN`. The old entry had **both** the folder
name and the force wrong, so:
- the folder was never found → condition 3 (bearings 3_1..3_5) never loaded;
- because the default `xjtu_test_bearings` includes `Bearing3_4/3_5`, `load_xjtu`
  raised "not on disk" — **XJTU-SY never actually ran**.
Fixed to `"40Hz10kN": (2, 40.0, 10.0)`. Added `_check_unmatched_conditions`: any
directory matching `^[\d.]+Hz\d+kN$` that is not a known condition now raises a loud
`ValueError` naming the folder and the expected set, so a future rename can never again
silently drop a condition. Stray non-condition dirs (`__MACOSX`, etc.) are ignored.
**Cache safety:** this changes XJTU data content (condition 3 appears, `setting_3`
becomes 10.0) but touches no cache-key field. It is safe because **no valid XJTU cache
could exist** (the old loader raised on the default split); if you built a cache with a
hand-hacked config, delete `cache/emb_XJTU-SY_*.npz` and `cache/windows_XJTU-SY_*` before
rerunning.

## 26. Tolerant data-dir resolution: subdir candidates + depth-1 nesting
Two real-world layout frictions, absorbed so the user never renames or reshuffles a
downloaded dataset:
- **Alternate subdir names.** `resolve_data_dir(config, subdir)` now accepts a tuple
  of candidate names and returns the first that exists under `config.data_root` (else
  the first candidate, so "not found" errors name the documented path). XJTU declares
  `("XJTU-SY", "XJTU-SY_Bearing_Datasets")` — the zip's own name loads as-is.
- **Zip-in-a-folder nesting.** `xjtu._descend_to_conditions` checks the resolved root
  for the condition folders and, if absent, scans its IMMEDIATE subdirectories
  (depth-1 only, no recursive walk) for one that holds them, descending with a printed
  notice. Absorbs `XJTU-SY/XJTU-SY_Bearing_Datasets/35Hz12kN/...`.
An explicit `config.data_dir` still wins verbatim (tests point it straight at a folder).
Paths are **not** part of any cache key (§23), so this changes no embeddings/results.
The same tuple mechanism is reused by the N-CMAPSS loader (§27).

## 27. N-CMAPSS loader (src/datasets/ncmapss.py) — cycle-aggregated frames
Adds the NASA N-CMAPSS run-to-failure dataset (Arias Chao et al. 2021; one `.h5` per
sub-dataset DS01–DS08d) into the canonical C-MAPSS-shaped frame, so every downstream
stage runs unchanged. All choices are `DECISION (uncited)` — there is no community
*cycle-level* N-CMAPSS protocol.
- **Cycle aggregation.** The raw data is 1 Hz WITHIN each flight; one flight = one
  cycle. Each `(unit, cycle)` group is reduced to per-cycle summary statistics:
  `mean` + `std` of each of the 18 raw channels (4 flight-condition `W` + 14 measured
  `X_s`), plus `cycle_len_s` = the number of 1 Hz rows in the flight (observable flight
  duration). **37 channels** = `NCMAPSS_FEATURE_COLUMNS` (config). `std` is pandas'
  sample std (ddof=1); one-row cycles → NaN → 0.
- **Oracles excluded.** Virtual sensors `X_v`, health-parameter ground truth `T`, and
  the per-row RUL `Y` are simulation oracles and are **never read**. RUL is re-derived
  from cycle counts by `data.add_train_rul`, exactly as for C-MAPSS. The synthetic test
  fixture writes those keys full-length to prove the loader ignores them.
- **Channel-name fail-loud.** The decoded `W_var`/`X_s_var` from the file must equal
  `NCMAPSS_W_VARS`/`NCMAPSS_XS_VARS` *as sets* (the file's order is used for reading);
  a mismatch raises listing both sets rather than silently reordering.
- **`setting_1 = Fc`** (flight class 1/2/3, constant per unit); `setting_2/3 = 0`.
  `condition_norm` resolves **auto-OFF** (flight conditions are continuous, already
  carried as channels); force `condition_norm=True` for per-flight-class normalization.
- **Split & truncation.** Train = the file's `*_dev` units (full run-to-failure); test =
  the file's `*_test` units (preserving the dataset's deliberate distribution shift),
  truncated at `config.ncmapss_test_truncation` (default 0.6) of life so the predict-at-
  last-observed-cycle protocol applies — same device as XJTU (§22). `rul_truth` =
  remaining cycles. New `ncmapss_test_truncation` config field is in the window cache key
  **only** when `dataset_kind()=="ncmapss"` (FD001/XJTU keys byte-identical to before —
  verified: `windows_FD001_1da313c871251cec`).
- **`max_rul` inactive.** N-CMAPSS end-of-life is ~60–100 cycles, so the default cap 125
  never binds → the target is plain linear RUL (matches N-CMAPSS community practice). Do
  not "fix" this.
- **Parsed-frame cache.** Parsing 1–3 GB of h5 is minutes; the aggregate is ~10²–10³
  rows. Cached to `cache/ncmapss_agg_<ds>_v<NCMAPSS_AGG_VERSION>.npz` (untruncated, so
  truncation re-applies from config without re-parsing). `NCMAPSS_AGG_VERSION=1` plays
  the cache-schema role for aggregation logic; the aggregate is otherwise
  config-independent. The cache is keyed by `ds`+version only (location-independent,
  like embeddings, §23) — pointing at a different N-CMAPSS directory with the same DS
  name reuses the cache; delete it to force a re-parse.
- **Non-comparability warning.** Published N-CMAPSS RMSEs use 1 Hz sub-cycle windows over
  full test trajectories. These cycle-aggregated, truncation-protocol numbers are **not
  comparable** to them and must never share a table (role: same-protocol cross-model
  comparison for RQ1/RQ4, like XJTU-SY).
- **Registry.** `dataset_kind()` maps `DS*` → `ncmapss`; `datasets/__init__` registers
  the family; `DEFAULT_SENSOR_COLUMNS["ncmapss"]` = the 37 channels. `h5py>=3.10` added
  to requirements (core: tests write synthetic h5). The registry-drift test covers the
  new family automatically.

## 28. DSALL — the combined N-CMAPSS fleet (RQ1 high-data arm)
**Per-file N-CMAPSS is a LOW-unit dataset** (6–9 dev units): by-unit it sits at the
*low* end of the data-efficiency sweep, not the high end RESEARCH_PLAN §3 wanted. The
high-data arm is the **union of every file** — ~100+ units with heterogeneous failure
modes and flight classes, a realistic mixed fleet. `dataset="DSALL"`:
- Iterates every resolved member file, each loaded through its own per-file aggregate
  cache (§27) — so DSALL costs nothing beyond the per-file parses.
- **Unit renumbering** `file_index*1000 + unit` (collision-proof, reversible:
  `file_index = uid // 1000`, `unit = uid % 1000`). Each file keeps its dev/test roles
  and per-unit truncation.
- **Member determinism.** `config.dsall_datasets` set → EXACTLY those members
  (reproducible), raising on any non-member name, any named-but-absent file, or fewer
  than 2 members; the sorted member list joins the window cache key. None → whatever is
  on disk (≥2 required), keyed literally `"auto"` so an exploration union never
  masquerades as a fixed dataset. The campaign pins the full list (§30). The resolved
  members are printed at load and captured in run-metadata via the resolved config.
- `is_available("DSALL")` requires ≥2 `N-CMAPSS_DS*.h5` present (a 1-file union is just
  that file). DSALL rows are keyed `dataset="DSALL"` — no schema change (the `dataset`
  column has been a restart key since §21).

## 29. Unit-count grid auto-appends the full fleet
`run_sweep` and `run_fairness_baselines` previously **skipped** any
`n_units > available` (`if n_units > len(all_units): continue`), so a dataset smaller
than `max(data_unit_counts)` never got a full-data cell — XJTU-SY (9 train bearings)
ran only {2,5}; N-CMAPSS DS02 (6 dev units) only {2,5}; neither ever reached its own
full fleet. New `sweep.resolve_unit_counts(counts, available)` returns
`sorted({n for n in counts if n < available} | {available})` — every requested count
below the fleet size **plus the full-fleet cell**. Wired into both functions' default
grid.
- FD001–FD004 (100 train units, grid max 100): result is exactly the requested grid, so
  **every existing restart key and recorded result stays valid** (asserted in tests).
- XJTU-SY → {2,5,9}; DS02 → {2,5,6}; DSALL → {2,5,10,25,50,…,N}.
- `run_horizon_eval` already defaults `n_units_list` to `[len(all_units)]` (the full
  fleet) and the campaign passes it all-units, so horizon needed no change.
Tests: `resolve_unit_counts` unit cases + a 5-train-unit sweep that yields exactly
{2,5} from `data_unit_counts=[2,50]`. Two pre-existing 8-unit smoke/fairness tests were
updated (they now legitimately gain the 8-unit full-fleet cell).

## 30. Campaign default overrides + notebook data-layout instructions
- **`campaign.DEFAULT_DATASET_OVERRIDES`** records per-dataset protocol choices ONCE
  instead of every notebook re-deciding them: XJTU-SY (`max_rul=125` min, `window_size=30`,
  `tsfm_context_length=256` — cycles are minutes, §22) and DSALL (`dsall_datasets` pinned
  to all 10 members for a deterministic cache key, §28). `run_campaign`'s
  `dataset_overrides` now means: `None` (default) → the recorded defaults; a non-empty
  dict → merged OVER them per dataset per key (user wins); explicit `{}` → opt out of all
  overrides. `merge_dataset_overrides` deep-copies so the module constant is never
  mutated. The per-combo log line prints the resolved override for provenance.
- **Notebook** (`colab_main.ipynb`): the Config markdown documents the one-`data_root`
  layout (`Data/CMAPSSData`, `Data/XJTU-SY`, `Data/N-CMAPSS` flat `.h5`), the accepted
  XJTU folder-name/nesting variants (§26), and the first-run N-CMAPSS aggregate-cache
  parse (§27). The campaign markdown lists the full dataset set (FD001–FD004 + XJTU-SY +
  DS01…DS08d + DSALL) and explains the override semantics; the campaign cell now calls
  `run_campaign(config, device=…)` with the recorded defaults (the old hand-written XJTU
  override that conflicted with the pinned protocol is removed). Only cells 2–3 changed;
  the deep-dive sections are untouched.

## 31. DSALL default excludes DS08d (unavailable large file)
`N-CMAPSS_DS08d-010.h5` (~2.9 GB, the largest sub-dataset) truncates on download and is
frequently not obtainable in full. Because `campaign.DEFAULT_DATASET_OVERRIDES` pinned
DSALL's members to all 10 files (§30) and a pinned-but-absent member **raises**
(`resolve_dsall_members`, §28), a missing/corrupt DS08d made **DSALL fail entirely** rather
than skip — the RQ1 high-data arm never ran. The default pin is now the **9 reliably
available files** (DS01–DS07 + DS08a + DS08c); DSALL unions those into its ~60+-unit fleet.
- **Per-file DS08d** is unaffected: when the file is absent the campaign skips that combo
  with the standard "not downloaded" notice (`is_available` globs the exact name), and runs
  it normally if a verified copy appears.
- **Cache/reproducibility:** this changes DSALL's member list, hence its window cache key
  (the sorted member list joins the key, §28) — correct, since **no valid DSALL cache exists**
  (DSALL never completed). To include DS08d later, add `"DS08d"` back to the DSALL pin in
  `campaign.py` once a verified copy is on disk; that is a deliberate new dataset (new key),
  not a silent change. No other dataset, cache key, or CSV schema is affected; the FD001 keys
  remain byte-identical.

## Not implemented (deliberately out of Phase-1 scope, Task 2.6)
TimesFM/MOMENT/TTM/Moirai (register a new `src/models/` module under its
`model_name`, §23); experiment-tracking services; CLI frameworks. No result numbers,
comparisons, or conclusions are written anywhere (Task 2.5) — recorded winners (§12)
come only from completed runs.

*(N-CMAPSS moved OUT of this list — implemented in §27; see DATASET_EXPANSION_PLAN.md.)*
