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
  DS01…DS08c + DSALL) and explains the override semantics; the campaign cell now calls
  `run_campaign(config, device=…)` with the recorded defaults (the old hand-written XJTU
  override that conflicted with the pinned protocol is removed). Only cells 2–3 changed;
  the deep-dive sections are untouched.

## 31. Exclusion of corrupted N-CMAPSS DS08d dataset
The sub-dataset `DS08d` (`N-CMAPSS_DS08d-010.h5`) was found to be corrupted in the official NASA Prognostics Center of Excellence (PCoE) source repository (`17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip`). The file has a physical size of 2,885,034,848 bytes, which is exactly 32 bytes short of the expected 2,885,034,880 bytes recorded in its HDF5 superblock.
- **Workarounds fail.** Attempting to open the file raises `OSError: Unable to synchronously open file`. Artificially padding the file with 32 trailing zero-bytes resolves the size mismatch but raises `RuntimeError: Unable to get group info (bad symbol table node signature)` because the missing bytes contain critical root-group symbol table metadata.
- **Exclusion policy.** Because this corruption exists in the NASA source file itself, all public mirrors (such as Kaggle) suffer from the same truncation. Standard practice in the research community is to exclude `DS08d` from runs.
- **Code modifications.** Removed `"DS08d"` from `NCMAPSS_DATASETS` in `src/config.py` and from the active campaign overrides (`DEFAULT_DATASET_OVERRIDES["DSALL"]["dsall_datasets"]` in `src/campaign.py`) so the combined `DSALL` dataset and campaign sweeps run cleanly over the remaining nine valid datasets.

## 32. Milestone 0 — coverage gate, provenance backbones, mock parametrization
First foundations of the v2 "when do TSFMs work" build (IMPLEMENTATION_PLAN §3, Phase A).
Small, additive, unblocks Milestones 1–2. No cache key, CSV schema, or recorded-result
change; the FD001 stable-key test is untouched and green.
- **Coverage gate (M0.1).** `.coveragerc` pins branch coverage of `src/` with
  `fail_under = 100`; `pytest-cov` added to `requirements.txt` (and `matplotlib`, a
  module-level import in `src/plots.py`, promoted from "Colab ships it" to a listed core
  dep so the gate runs standalone in CI). `README.md` documents
  `pytest -q --cov=src --cov-branch`. **The `# pragma: no cover` boundary
  (`DECISION (uncited)`):** the single line where a lazy backbone/dataset library is first
  imported (e.g. `from chronos import Chronos2Pipeline`) is the ONLY sanctioned pragma;
  everything above it — shape handling, pooling, loc/scale, caching, scoring — is covered
  by mocks (`tests/synthetic.py`). coverage.py cannot enforce *where* a pragma sits, so
  that boundary is a review rule, recorded in `.coveragerc` comments and here.
  **Phasing note:** 100% is the *Milestone-2* acceptance gate; M0 only stands up the
  tooling. Until Milestones 1–2 cover every module (and pragma the lazy imports), the
  command reports the current `src/` coverage as below 100% by design — plain `pytest -q`
  (no `--cov`) stays fully green.
- **Provenance backbones (M0.2).** `evaluate.package_versions()` now also records
  `momentfm`, `uni2ts`, `timesfm`, `tsfm_public`, `pycatch22`, `sksurv`, and `lifelines`
  (the four new TSFMs + the catch22 foil + the censored-metric libs, §2). Absent modules
  report `"not-installed"`, so the run-metadata JSON states exactly which backbone/library
  builds produced a run without requiring any of them to be installed.
- **Mock parametrization (M0.3).** `tests/synthetic.py::MockEmbedder` gains `layout`
  (`"multivariate"` default, Chronos-2/Moirai/TTM-like joint embed with special tokens vs
  `"univariate"`, MOMENT/TimesFM-like per-channel embed with no special tokens) and
  `channel_aggregation` (`"concat"`/`"mean"`, the RQ-M fairness knob M1 adds as a real
  `Config` field). Feature width `F`: multivariate → `feature_dim` (both agg modes
  coincide — the joint summary is already channel-collapsed, a documented mock
  simplification); univariate → `C·feature_dim` (concat) or `feature_dim` (mean). The
  defaults (`multivariate`, `concat`) reproduce the original fixture byte-for-byte
  (`F == feature_dim`, `test_smoke` still asserts `emb.shape == (N, 16)`), so every
  pre-M0 test stays green. New tests in `test_embeddings.py` cover both layouts, the
  aggregation modes, the empty-context edge case, `describe()` keys, and the param guards.
- **Stale-test fix (housekeeping).** `test_campaign.py`'s DSALL assertion hard-coded 10
  members; §31 removed the corrupted DS08d, leaving 9. The assertion now compares against
  `DEFAULT_DATASET_OVERRIDES["DSALL"]["dsall_datasets"]` directly so a member-list change
  never re-stales it.

## 33. Notebooks: per-dataset-family, self-cloning from GitHub
The Drive-hosted monolith `notebooks/colab_main.ipynb` (one serial run-all across every
dataset, with the whole repo mirrored on Drive and re-uploaded on every change) is replaced
by **three parallel per-family notebooks** — `notebooks/cmapss.ipynb` (FD001–FD004, plus the
gated FD001 deep-dives), `notebooks/xjtu.ipynb` (XJTU-SY), `notebooks/ncmapss.ipynb`
(DS01–DS08c + DSALL) — so each family runs on its own Colab runtime simultaneously instead
of one after another. Each notebook restricts `run_campaign(config, datasets=…)` to its
family (derived from the registry via `datasets.<family>.DATASETS`, so it self-maintains).
- **Self-cloning setup (the reason Drive can shed everything but notebooks + data).** The
  setup cell `git clone`s the **public** repo (`https://github.com/blozanod/Predictive-
  Maintenance-LSTM.git`) fresh into the ephemeral Colab filesystem and puts *that* clone on
  `sys.path` — re-run-safe (`git pull --ff-only` if already cloned). Drive now holds ONLY
  `Data/`, the embedding `cache/`, and `results/` (the persistent artifacts); code is never
  re-uploaded. A `REPO_BRANCH` knob (default `main`) selects the branch to pull.
- Drive layout, per-dataset overrides, and the recorded §12 winner config are unchanged; the
  deep-dive sections (ablation, raised-cap, transfer) live in the C-MAPSS notebook, gated on
  `RUN_DEEP_DIVES` exactly as before. `README.md`'s "Run on Colab" section is rewritten for
  the three notebooks.

## 34. Milestone 1 — four new TSFM embedders + `channel_aggregation` (RQ-M)
The v2 roster grows from one backbone to five (IMPLEMENTATION_PLAN §4.1). Four new
`src/models/` modules register under their `model_name`: `moirai.py`
(`Salesforce/moirai-2`, multivariate-native), `moment.py` (`AutonLab/MOMENT-1-large`,
univariate), `timesfm.py` (`google/timesfm-2.5`, univariate), `ttm.py`
(`ibm-granite/granite-timeseries-ttm-r2`, tiny channel-mixing) — each in `EMBEDDERS`.
- **Semantic (not index-based) pooling contract.** `embeddings.py`'s pooling is
  refactored into two stages: `pool_patches` reduces one window's patch axis to a
  per-variate vector honoring the four pooling NAMES, then `aggregate_variates`
  collapses the variate axis. A new `n_special_tokens` knob makes the names mean the
  same thing across layouts — Chronos-2 appends 2 trailing special tokens (REG,
  forecast; `n_special_tokens=2`), the four new backbones append none
  (`n_special_tokens=0`), so `forecast_token` maps to the forecast token for Chronos-2
  and to the last patch (the closest "predict-next" summary) elsewhere. The four new
  backbones share `models/base.py` (`TSFMEmbedderBase`): only their backbone
  load/call (`_load_pipeline` / `_encode_batch`, the `# pragma: no cover` boundary)
  differ; batching, pooling, the loc/scale fallback, and `describe` are shared and
  CPU-tested via a fake `_encode_batch`.
- **`channel_aggregation` (the RQ-M fairness knob).** New `Config` field
  (`"concat"` default → `F = n_variates·d_model`; `"mean"` → `F = d_model`, the common
  representation), applied UNIFORMLY to all five models (Chronos-2 threads it too).
  Added to the embedding key ONLY when `!= "concat"`, so every recorded FD001 key is
  byte-identical (the default `concat`/`n_special_tokens=2` pooling reproduces the old
  Chronos-2 output byte-for-byte; stable-key test green).
- **Per-channel loc/scale fallback.** Univariate/plain backbones that do not surface
  their RevIN loc/scale fall back to the per-channel INPUT mean/std
  (`TSFMEmbedderBase.loc_scale_from_contexts`), keeping the canonical `(N, n_variates,
  2)` shape. Documented fallback if a backbone won't surface clean per-patch states:
  its encoder/penultimate hidden states (RESEARCH_PLAN §11) — recorded per module as a
  `# DECISION (uncited):`.
- Tests (`tests/test_models.py`, no backbone import): the pooling-name→layout mapping
  for both layout kinds, `concat` vs `mean` dims, the loc/scale shape, empty context,
  `describe` keys, `make_embedder` selecting each, and a models registry-drift test
  mirroring the datasets one.

## 35. Cross-TSFM representation-fairness run (native vs common) — RQ-M
`sweep.run_representation_fairness` runs every model TWICE at full data / MSE / ≥3
seeds — **native** (`channel_aggregation="concat"`, own default pooling) and **common**
(`channel_aggregation="mean"`, `pooling="mean"`) — writing `representation_fairness.csv`
so the cross-TSFM ranking can be checked for aggregation artifacts. Each (model, mode)
has its own Stage-A cache (keys differ by aggregation/pooling), built idempotently;
`embedder_factory` injects a CPU mock; restartable on (model, aggregation, pooling,
seed). `plots.plot_cross_tsfm` renders the native-vs-common grouped bars.

## 36. Scoring & the win-rule (`src/scoring.py`) — the success map
The formal realization of RESEARCH_PLAN §8. `strongest_baseline_per_cell` finds the
toughest COMPETITOR bar per `(dataset, n_units[, factor, level])` cell; `win_verdict`
returns win/tie/loss/hollow per (cell, TSFM); `success_map` reads the per-combo CSVs
(glob / directory / file) into the headline table (verdict + margin + p + seed-means,
RMSE alongside).
- **Primary metric = `nasa_clipped`** (asymmetric); **win** iff the TSFM seed-mean
  beats the strongest competitor by more than `config.win_margin` AND a paired-seed
  t-test supports it at `config.win_alpha`; the significant reverse is a **loss**;
  otherwise a **tie**. The paired-seed core is generalized out of
  `evaluate.paired_seed_ttest` into `evaluate.paired_ttest` (nan-safe: <2 pairs or a
  constant difference → nan, never scipy ±inf) and reused here.
- **Absolute-floor (hollow) guard.** `predict_mean`/`cycle_reg` are treated as
  *floors*, NOT competitors (RESEARCH_PLAN §6 lists them apart), which is what makes
  the guard reachable: a TSFM that beats every real baseline but is no better than the
  trivial predict-mean floor is downgraded from win to **hollow**
  (`# DECISION (uncited):`). New non-cache-key `Config` fields: `win_margin`,
  `win_alpha`, `usability_floor_metric`. `plots.plot_success_map` renders the
  win/tie/loss/hollow heatmap (models × conditions, faceted per dataset).

## 37. Earliness layer: histograms + cost curve ("too early is also bad")
`evaluate.earliness_histogram` and `evaluate.cost_curve` (RESEARCH_PLAN §8), tied to the
horizon `bias` / `nasa_score` sign convention (§16): `d = pred - true`, `d ≥ 0` is the
penalized "dangerously LATE" side (claims more life than remains), `d < 0` is
"wastefully EARLY". The histogram reports `frac_late` vs `frac_early` and the per-bin
distribution; the cost curve sweeps `cost = Σ max(0, true-pred) + ratio·Σ max(0,
pred-true)` over a range of late:early ratios — no single arbitrary ratio.
`horizon.run_earliness` emits `earliness.csv` + `cost_curve.csv` from
`horizon_predictions.csv` (restartable); new non-cache-key `Config` fields
`earliness_bin_edges`, `cost_ratios`. `plots.plot_earliness` / `plots.plot_cost_curve`
render them.

## 38. Factor-probe harness (`src/probes.py`) + sim-only interventions
`run_factor_probe` sweeps ONE playbook factor over levels on an anchor dataset with a
reduced roster (top-2 TSFMs + top-2 foils + best NN), applying each level's
intervention as a `Config` override, building the (idempotent) Stage-A cache at the
intervened shape, running the head + reduced baselines, and appending
`probe_<factor>.csv` rows keyed by `(dataset, model, factor, level, n_units, seed,
loss)` — a success-map input. `probe_roster` resolves the reduced roster from a Tier-1
glob. `embedder_factory` injects a CPU mock; restartable.
- **Channel selection (RQ-C, subtractive):** each level is a `sensor_columns` subset
  (already in the window key, no perturbation of kept values).
- **Noise tolerance (RQ-H, perturbative, SIM ONLY):** new `noise_injection` `Config`
  dict (`gaussian` at an SNR / `drift` ramp / `dropout` blanking; magnitudes in
  per-channel std units, deterministic in a seed). Applied in `data.load_prepared`
  AFTER labels/normalization, BEFORE windowing; added to the window key ONLY when
  non-empty (existing keys unchanged). `data.apply_noise_injection` **RAISES on a REAL
  dataset** (XJTU/MetroPT/Hydraulic/Backblaze) reporting the allowed simulated families
  and the observed dataset — perturbing real readings is out of scope by design
  (RESEARCH_PLAN §1). `# DECISION (uncited):` records the three kinds + their params.
- Any other factor whose levels are already `Config`-override dicts slots in with no
  harness change (the Phase-B aggregation / feature-mode knobs).

## 39. Zero-shot health-index forecasting arm (`src/zeroshot.py`) — RQ-Z
The 0-failures endpoint of RQ-B: no head, no training. `run_zeroshot` builds an
unsupervised HEALTH INDEX (first PC of the z-standardized sensors, oriented to increase
toward failure — no RUL labels used), calibrates a failure threshold from the fleet's
run-to-failure endpoints, forecasts the index forward with a TSFM's native forecasting
mode, and reads predicted RUL off the threshold crossing. Scored with both-protocol
metrics against the `predict_mean` and `cycle_reg` floors → `zeroshot.csv`. The
`forecaster_factory` seam mirrors `embedder_factory` (a mock returns a fixed
trajectory; the default `ChronosForecaster`'s backbone load/call is the
`# pragma: no cover` boundary). `# DECISION (uncited):` records the index construction
and the threshold calibration.

## 40. Milestone 0/1 review fixes (win-rule/zero-shot, noise key, Chronos coverage, deps)
Four defects found in an adversarial review of the M0/M1 build, fixed here. No recorded
result changes; the FD001 window/embedding keys stay byte-identical
(`windows_FD001_1da313c871251cec`, `emb_FD001_chronos-2_forecast_token_w30_c30_v2_…`).

- **Zero-shot is now scoreable by the win-rule (`src/scoring.py`).** IMPLEMENTATION_PLAN
  §4.5 scores the RQ-Z arm "with the win-rule vs the `predict_mean`/`cycle_reg` floors,"
  but `run_zeroshot` tags its model `<tag>_zeroshot` while `is_tsfm_model` recognized
  only `_mlp` — so `success_map` on a `zeroshot.csv` returned ZERO rows (the zero-shot
  prediction was mis-read as a *competitor baseline*, leaving no TSFM row to judge).
  `is_tsfm_model` now accepts both suffixes (`TSFM_SUFFIXES = ("_mlp", "_zeroshot")`),
  and `win_verdict` / `success_map` gain `compare_to_floors=False`: when set (the
  zero-shot path) the **best floor** becomes the comparison bar and the hollow guard is
  skipped (beating a floor is the whole point). The default core/probe path is
  unchanged — a cell with only floors is still skipped. The strongest-bar selection is
  factored into `_strongest_by_predicate` so competitor-bar and floor-bar share one
  implementation. (The zero-shot arm has a single seed, so its paired test is
  under-powered → verdicts are conservatively `tie` unless run multi-seed; the row
  still carries the signed margin vs the floor.)
- **`noise_injection` seed is now in the cache key (`src/config.py`, `src/data.py`).**
  `apply_noise_injection` seeds the perturbation with `spec.get("seed", config.seed)`,
  but the window/embedding key folded in only the spec dict — so two configs differing
  ONLY in `config.seed` (same spec, no explicit spec seed) produced identical keys yet
  different perturbed data, silently reusing a stale cache (a violation of "cache keys
  are pure functions of Config," §1.2). New `Config.effective_noise_seed()` is the
  single resolution used by BOTH the perturbation and the key; `_window_key_fields`
  now adds `noise_seed` alongside `noise_injection` — **only when noise is set**, so
  every unperturbed key is byte-identical and `config.seed` stays absent from the
  no-noise key. An explicit `spec["seed"]` still pins a reproducible realization.
- **`ChronosEmbedder` refactored onto the tested base (`src/models/chronos.py`,
  `src/embeddings.py`).** The four v2 backbones isolate the GPU call in
  `_encode_batch`/`_load_pipeline` (the sole `# pragma: no cover`) and inherit the
  shared batching/pooling/loc-scale path, so they are CPU-tested; Chronos-2 alone kept
  a bespoke `embed_windows` with inline pooling and NO pragma, sitting at 29% coverage
  and unreachable under the "100% + single-pragma" gate (M0.1). It now extends
  `TSFMEmbedderBase` (`n_special_tokens = 2` for its REG+forecast tokens, `layout =
  "multivariate"`); only the two pragma'd backbone methods differ → **100% coverage**.
  The on-device pooling micro-optimization (§13) is retired in favor of the single
  host-side pooling reference — Stage A is one-time and cached, so the extra transfer
  is immaterial. The now-unused on-device twin `embeddings._pool_one_torch` (and its two
  tests) is deleted; `pool_window_embedding` is the single pooling reference for all
  five backbones.
- **The four backbone deps are declared (`requirements.txt`).** `momentfm`, `uni2ts`,
  `timesfm`, `granite-tsfm` were referenced by the new embedders and by
  `package_versions()` but never listed, so `pip install -r requirements.txt` left the
  M1 embedders un-importable and the Phase-1 spikes unrunnable. Added under a "v2 TSFM
  backbones (GPU; Stage A only)" block, conservatively pinned, imported only inside each
  `_load_pipeline` (never by the CPU tests). The M2+ libs (pycatch22, sksurv/lifelines,
  pyarrow) still arrive with their own milestones — `package_versions()` reports them
  `not-installed` until then, by design (§32).

## 41. Multi-seed zero-shot + backbone `_encode_batch` verification (real library APIs)
Two follow-ups to the M0/M1 review.

- **Zero-shot now runs over multiple seeds (`src/zeroshot.py`).** The arm is
  deterministic given its calibration set, so a single run is one lucky/unlucky draw of
  observed failures. `run_zeroshot` now sweeps `config.sweep_seeds` (default 5); each seed
  BOOTSTRAPS the calibration units (resample with replacement) before fitting the
  unsupervised health-index transform, the failure threshold, and both floors — so the
  reported seed-mean averages over draws and the win-rule's paired-seed test is no longer
  vacuous. `ZEROSHOT_KEYS` gained `seed`; rows stay `n_units=0` (the 0-target-failures
  endpoint); the health index still uses no RUL labels. Restartable per `(model, seed)`.

- **The four v2 backbone bodies were verified against each library's real source and
  corrected — all four had non-working API calls (`src/models/*.py`).** These
  `_encode_batch`/`_load_pipeline` methods are the sole `# pragma: no cover` boundary; CPU
  tests mock them, and the CI container has no GPU and no HuggingFace egress, so they were
  written from assumed APIs and never executed. Verifying each against the installed
  library (TTM, TimesFM signature-checked locally) and its GitHub source (MOMENT, Moirai):
  - **MOMENT** — `.embeddings` and `model_kwargs={"task_name":"embedding"}` were right, but
    MOMENT-1 hard-requires a FIXED `config.seq_len` (512) input with no auto-pad. Now pads
    each channel's most-recent cycles into a 512 buffer + `input_mask` and calls
    `embed(reduction="mean")` (verified against momentfm 0.1.4).
  - **Moirai-2** — `Moirai2Module` has **no `.encode()`**; its `forward()` consumes packed
    inputs and the reprs are internal. Rewritten to reproduce the encoder path
    (`scaler → in_proj → encoder`) per variate, the documented encoder-hidden-states
    fallback (RESEARCH_PLAN §11). Also: the `moirai2` submodule + `packed_causal_attention_mask`
    are NOT in any PyPI `uni2ts` release (only GitHub main) → `requirements.txt` now installs
    `uni2ts` from git.
  - **TimesFM 2.5** — `TimesFM_2p5_200M_torch` has **no `.embed()`**; per-patch hidden states
    are `output_embeddings` (index 1) of the underlying module's `forward(inputs, masks)`.
    Rewritten to patch (p=32) + mask + RevIN-normalize + read `output_embeddings`
    (signature-verified: `module.p`, the 4-tuple return, `torch_compile` kwarg).
  - **TTM** — `get_model` **requires `prediction_length`** and rejects sub-512 contexts
    without `force_return="zeropad"` (the old call passed neither → immediate raise). Fixed;
    inputs are zero-padded to `model.config.context_length`; `backbone_hidden_state`
    `(batch, n_variates, patches, d_model)` was the correct output field (signature-verified).
  - **`scripts/verify_backbones_colab.py`** (new) is the weight-level spike the container
    can't run: on a Colab GPU it loads each real model and runs `embed_windows` on synthetic
    contexts, asserting shape/finiteness/non-degeneracy and exiting non-zero on any failure.
    Final validation of these bodies (and the exact HF repo ids) is that spike, per
    RESEARCH_PLAN §9/§11 — a backbone that still fails is reported, not forced.

## 42. Per-model dependency isolation (the backbones can't share one environment)
Running the §41 Colab verification revealed that the four v2 backbones + Chronos-2 have
**mutually incompatible dependency pins** — no single environment can hold them, and the
combined `pip install -r requirements.txt` backtracks to `ResolutionTooDeep`. Proven on a
fresh Colab GPU runtime:
- **Moirai-2** (`uni2ts`) pins `torch<2.5`, so it uninstalls Colab's torch 2.10 and installs
  2.4.1 — which no longer matches the preinstalled `torchvision` (`operator
  torchvision::nms does not exist`), and that poisoned torch/torchvision then breaks
  Chronos-2 and TTM (their `transformers` import walks through torchvision).
- **MOMENT** (`momentfm`) hard-pins `numpy==1.25.2` (no Python-3.12 wheel → source build fails)
  and `huggingface-hub==0.24.0`.

Resolution — one isolated stack per backbone, one fresh runtime per backbone:
- **`requirements/` (new)**: `chronos.txt`, `ttm.txt`, `timesfm.txt`, `moirai.txt`,
  `moment.txt`, each a self-consistent stack, plus a `README.md` documenting every
  conflict. `moirai.txt` pins `torch==2.4.1` **and** the matching `torchvision==0.19.1`;
  `moment.txt` is installed `--no-deps` (its own pins are unbuildable). Pins are
  best-effort, finalized per model on a GPU.
- **Root `requirements.txt`** no longer lists the four v2 backbones (reverting the §40
  block that caused the resolver blow-up); it stays the installable **core + Chronos-2**
  for the CPU test suite and the Chronos campaign.
- **`notebooks/verify/<model>.ipynb` (new, 5)**: one thin notebook per model — clone,
  install only that model's `requirements/` file, run `verify_backbones_colab.py` for it.
  Each says to use a fresh runtime (the backbones must not share one). The dataset axis is
  deliberately NOT foldered: it carries no dependency variation, so it stays a runtime
  parameter, not a directory (avoids ~20 near-identical notebooks).
- **TimesFM repo-id fix**: the registry key `google/timesfm-2.5` 404'd on HuggingFace; the
  real weights are `google/timesfm-2.5-200m-pytorch` (verified in the timesfm source's
  `DEFAULT_REPO_ID`). Updated `EMBEDDERS` + the two test references. TimesFM's embedding
  body itself was reached and correct — only the id was wrong. `_embedding_key_fields`
  includes `model_name`, so only TimesFM's (never-built) cache key changes; FD001/Chronos
  keys are untouched.

## 43. Colab GPU verification round 1 — 3/5 pass; Moirai id + TTM torch/torchvision fixes
First real weight-level run of the §42 verify notebooks on a Colab GPU. **Chronos-2,
TimesFM 2.5, and MOMENT PASS** — finite, non-degenerate embeddings at the expected width
`F = n_variates·d_model` (Chronos 8·768=6144, TimesFM 8·1280=10240, MOMENT 8·1024=8192),
loc/scale `(N, n_variates, 2)` — confirming the pooling/aggregation/loc-scale contract end
to end for three backbones. Two failed and are fixed here:
- **Moirai-2 — wrong HF id** (same class of bug as TimesFM). `Salesforce/moirai-2` made
  `Moirai2Module.from_pretrained` build with an empty config (`__init__() missing 7
  required positional arguments`). The real id is **`Salesforce/moirai-2.0-R-small`**
  (uni2ts README loads exactly that). Updated `EMBEDDERS` + tests; the `_encode_batch`
  encoder path is unchanged (it was never reached before).
- **TTM — torch/torchvision ABI mismatch.** `from tsfm_public import get_model` pulls
  `from transformers import PreTrainedModel`, whose object-detection loss imports
  torchvision; Colab's stock torchvision (0.26, wants torch 2.11) mismatches the torch
  2.10 that granite-tsfm requires → `operator torchvision::nms does not exist`.
  `requirements/ttm.txt` now pins the matched pair **`torch==2.10.0` + `torchvision==0.25.0`**
  (torchvision 0.25.0 requires exactly torch 2.10.0, verified on PyPI; granite-tsfm accepts
  torch>=2.10,<2.11). The other three backbones' transformers-import paths never touch
  torchvision, which is why they passed.
Both are one-line-ish, ship in the same PR; re-verify TTM and Moirai on a fresh runtime.

## 44. Colab GPU verification round 2 — Moirai passes (4/5); TTM freq_token
Re-run of the §43 fixes on fresh runtimes:
- **Moirai-2 PASSES** with the corrected id — `emb=(4, 3072)` (8·384=3072 for the small
  model), finite, non-degenerate. This is the first real-weight execution of the
  source-verified `_encode_batch` encoder-packing path (scaler → in_proj → encoder per
  variate); it produces the canonical `(n_variates, patches, d_model)` correctly. **4/5**.
- **TTM — the torch/torchvision pin worked** (imports + loads cleanly now, revision
  `180-60-ft-l1-r2.1`), exposing the next layer: that r2.1 revision is
  frequency-prefix-tuned, so `forward` REQUIRES a `freq_token` (`Exception: Expecting
  freq_token in forward`). `models/ttm.py` now passes `freq_token = zeros(1)` (base/unknown
  frequency — we extract representations, not forecast a specific cadence; unused by
  non-ft variants; ft variants prepend one freq patch → patches+1, absorbed by the shared
  pooling). `# DECISION (uncited)`. Re-verify TTM on a fresh runtime.

## 45. catch22_gbm baseline + the C-MAPSS cross-TSFM Colab campaign (Stage A per model → Stage B once)
Wires the five GPU-verified backbones (§32–§44) into a runnable C-MAPSS campaign on Colab
and adds the last cheap foil the roster was missing. **The only `src/` change is the new
baseline**; everything else is notebook wiring around functions that already exist and pass
tests (`run_campaign`, `run_sweep`, `scoring.success_map`, `plots.*`, `horizon.run_earliness`).
No recorded result, cache key, or CSV schema changes; the FD001 keys stay byte-identical
(`windows_FD001_1da313c871251cec`) and `pytest -q` stays green.

- **`catch22_gbm` baseline (`src/baselines.py`, `requirements.txt`).** The hand-crafted-
  indicator foil (RESEARCH_PLAN §6, RQ-D: "do TSFMs make hand-crafted indicators
  obsolete?"). `catch22_features` computes the 22 canonical catch22 features
  (`pycatch22`, Lubba et al. 2019) **per channel per window** and concatenates them
  (`(N, 22·C)`); `Catch22GBMBaseline` feeds them to `lightgbm.LGBMRegressor` behind the
  **same `Baseline` interface** as `gbm` (`fit(train_w, train_y, val_w, val_y)` /
  `predict(test_w)`), registered in `BASELINES`. `pycatch22` is imported lazily inside
  `catch22_features` and, like the `lightgbm`/`sktime` baseline imports (NOT the GPU-only
  backbone loads), carries **no `# pragma: no cover`** — it is a CPU core dep exercised by
  the test, so the coverage policy's single sanctioned pragma boundary (§32) is untouched.
  `pycatch22` is added to `requirements.txt`'s **core** section (IMPLEMENTATION_PLAN §2
  lists it as core; tests use it). `run_sweep`'s DEFAULT baseline list is **unchanged**
  (recorded behaviour preserved) — `catch22_gbm` is opted in via `baseline_names` in the
  Stage-B notebook. A new test (`tests/test_smoke.py::test_catch22_gbm_baseline`) mirrors
  the gbm/minirocket test (`importorskip` both libs), asserts the `22·C` feature width, and
  exercises fit/predict on **both** the no-val and val (`eval_set`) branches (full line +
  branch coverage of the new code).

- **Per-model dependency isolation forces a Stage-A / Stage-B split (why two kinds of
  notebook).** The five backbones cannot share one environment (§42), and Stage B needs
  none of them once the embeddings are cached — so the split lands on the repo's existing
  Stage-A (embed → cache) / Stage-B (read cache → train heads + baselines) seam, with the
  **embedding cache on Google Drive as the hand-off**. Stage A and Stage B build the SAME
  canonical `Config` for every cache-key field (dataset / window / sensors / max_rul /
  model_name / pooling / tsfm_context_length / condition_norm) at the recorded §12 winner
  shape (`tsfm_context_length=256`, `pooling="mean"`; `head_features="emb+locscale"` is a
  Stage-B knob that does NOT change the cache, §9), so Stage B's `embedding_cache_key`
  matches the caches Stage A wrote.

- **Stage-A notebooks (`notebooks/campaign/{chronos,moment,timesfm,ttm,moirai}.ipynb`).**
  One per model, mirroring `notebooks/verify/<model>.ipynb`: clone the campaign branch →
  install ONLY that model's isolated stack (`requirements/<model>.txt`; MOMENT `--no-deps`)
  → mount Drive → build the canonical `Config` (`data_root="Data"` since C-MAPSS is
  committed, `cache_dir`/`results_dir` under a Drive folder, `model_name` = the exact
  `EMBEDDERS` registry key, e.g. `google/timesfm-2.5-200m-pytorch`,
  `Salesforce/moirai-2.0-R-small`) → `run_campaign(models=[that_model],
  datasets=["FD001".."FD004"], stages=["cache"], device="cuda")` (embeds + caches; the
  embedder auto-detects the GPU). A trailing cell also runs `build_horizon_cache` for the
  four datasets — the **GPU half of the horizon/earliness deliverable** (embed every test
  cycle), so Stage B (no backbone) finds the sidecar cache on Drive and only trains. Each
  notebook says "fresh GPU runtime, one model per runtime" and every step is restartable.

- **Stage-B notebook (`notebooks/campaign/score.ipynb`).** ONE core runtime: clone →
  `pip install -r requirements.txt` (core + Chronos-2 only; the four v2 backbones are NOT
  in it, and chronos-forecasting is never imported because the caches exist) → mount the
  same Drive folder → the SAME canonical `Config` →
  `run_campaign(models=[all five EMBEDDERS keys], datasets=["FD001".."FD004"],
  stages=["sweep","fairness","horizon","figures"],
  baseline_names=["predict_mean","gbm","minirocket","cnn","lstm","catch22_gbm"])` — reads
  the five caches, trains heads + baselines, writes per-combo `*_results_v2.csv` +
  data-scaling/horizon figures. Then the CROSS-TSFM deliverables reuse tested functions:
  `scoring.success_map` over `results_dir/*_results_v2.csv`
  (`cell_fields=("dataset","n_units")`) → `plots.plot_success_map` (the win/tie/loss/hollow
  map); the per-combo `*_results_v2.csv` concatenated into one frame →
  `plots.plot_data_scaling` (all five models on one curve per dataset/metric); and
  `horizon.run_earliness` over the concatenated `*_horizon_predictions.csv` →
  `plots.plot_earliness` / `plots.plot_cost_curve`. Baselines re-run per combo, so their
  rows repeat across combo CSVs — the concatenation **dedupes** them (one row per logical
  data point; the `<tag>_mlp` TSFM rows are unique per model), which matters especially for
  the cost curve's sums; the combined CSVs are named so the `*_results_v2.csv` /
  `*_horizon_predictions.csv` globs never re-pick them up.

- **Scope of this run.** RQ-Z zero-shot (`src/zeroshot.py`) and RQ-M representation-fairness
  (`sweep.run_representation_fairness`) use the TSFM's forecasting/embedding on GPU, so they
  belong in the per-model Stage-A notebooks, NOT Stage B; they are left out of this first
  core run (added later as optional Stage-A cells) so nothing blocks the core campaign. No
  result numbers or claims are written anywhere — no runs happen here (Task 2.5).

## 46. C-MAPSS cross-TSFM campaign — recorded results (July 2026 Colab runs)
The Tier-1 C-MAPSS campaign (§45's Stage-A × 5 + Stage-B) COMPLETED on Colab,
2026-07-24 → 2026-07-27. Every number below is read from the completed-run CSVs on
Drive (`pdm_tsfm/results/`): 20 per-combo `<FD00x>_<model>_results_v2.csv` (full
grid: 5 TSFMs × FD001–FD004 × unit grid × 5 seeds × {mse, corn}; 8 baselines ×
{native}; 2 340 deduped rows in `cross_model_data_scaling.csv`), per-combo horizon
CSVs, `earliness.csv` + `cost_curve.csv`, and 196 figures incl. the four
`cross_tsfm_success_map_*` heatmaps. No cells missing; no combo failed.

- **Sanity gate reproduced.** FD001 `chronos-2_mlp`, full data, mse: clipped RMSE
  **10.66** (5 seeds: 11.02/11.52/10.59/9.84/10.34) — matches the recorded §12
  winner (10.66 ± 0.51) from the pre-refactor pipeline. The §32–§45 build did not
  drift the recorded result.
- **Full-data leaders (seed-mean `nasa_clipped` | clipped RMSE).** FD001:
  Moirai-2 166 (Chronos-2 best RMSE 10.66); FD002: TimesFM 564 (TTM best RMSE
  10.96); FD003: TimesFM 220 | 11.39; FD004: TTM 684 | 12.39. Strongest
  competitor baseline at full data is `gbm_age` on all four (252/904/287/1020).
  **No single TSFM dominates**; Chronos-2 is top-1 on none of the four by NASA.
- **Success map (win-rule §36, `nasa_clipped`, paired-seed α=0.05, 130 TSFM
  cells):** **23 win / 103 tie / 4 loss / 0 hollow**. Per model (W–L): TimesFM
  8–0, Moirai-2 5–1, TTM 4–2, Chronos-2 3–0, MOMENT 3–1. Wins cluster on the
  multi-condition datasets (FD002: 13 of 23) and at n_units ≥ 25; every n=2 cell
  is a tie (5-seed pairing has no power there and nothing is usable at n=2 —
  best-TSFM NASA is within/above the predict-mean floor on FD004). The
  seed-MEAN of the best TSFM beats the best competitor baseline in 25/26
  (dataset × n_units) cells — all except FD004 n=2.
- **The 4 loss cells:** TTM FD003@50 (442 vs gbm 325), TTM & Moirai-2 FD004@10
  (5 893/5 971 vs minirocket 3 380), MOMENT FD004@249 (1 851 vs gbm_age 1 020).
  Scale note: tiny TTM is the FD004/FD002 full-data leader yet loses hardest in
  low-data cells — "does scale matter" is regime-dependent, not a yes/no.
- **CORN vs MSE (RQ-E, recorded).** At n=2, CORN's aggregate NASA is 0.3× MSE's
  (48k vs 164k); it rescues the catastrophic low-data blowups of MOMENT/TTM/
  Moirai (n≤5 NASA 2–6× better), while Chronos-2/TimesFM prefer MSE at n=5. On
  clipped RMSE, CORN ≤ MSE at every n (ratio 0.92–1.00). The §7 "ordinal holds
  where MSE explodes" claim replicates on C-MAPSS, but is model-dependent.
- **Earliness/cost two-sided layer (recorded nuance).** At full data the TSFM
  heads are dangerously-late slightly MORE OFTEN than gbm/lstm (e.g. FD001 frac
  late: gbm 0.052 vs TSFMs 0.070–0.084), but with smaller magnitudes — NASA
  (exponential in lateness) ranks TSFMs first while the LINEAR cost curve at
  late:early = 100:1 makes gbm/lstm cheapest on 3/4 datasets. Frequency vs
  magnitude of lateness diverge; the playbook must report both, per §8.
- **Tier-2 roster (per `probe_roster`, `nasa_clipped` cell-and-seed mean):**
  top-2 TSFMs = **TimesFM 2.5, Chronos-2**; top-2 foils = **gbm, minirocket**;
  best NN = **lstm**. Caveat (recorded, not yet acted on): the mean-over-cells
  rule is dominated by low-data blowup cells, so Moirai-2 ranks LAST among
  TSFMs despite the best FD001 full-data NASA; a probe that targets the
  high-data regime may want an explicit roster override.
- **Not yet run (unchanged scope, §45):** RQ-Z zero-shot, RQ-M common-
  representation fairness ablation, and the RQ-A/C/E(cap)/H factor probes —
  these are the remaining C-MAPSS chapters of Milestone 2.

## 47. Milestone-2 completion notebooks: 3-session split (`notebooks/campaign/milestone2/`)
Notebook-only wiring (no `src/` change) that runs the remaining C-MAPSS chapters (§46
"not yet run" list) on the functions that already exist and pass tests
(`run_factor_probe`, `run_representation_fairness`, `run_zeroshot`). Three notebooks =
three parallel Colab sessions, split by backbone because the model stacks cannot share
an environment (§42):

1. **`timesfm_probes.ipynb`** — TimesFM 2.5: RQ-A/C/E/H probes **with the shared
   baselines** (roster per §46 `probe_roster`: gbm + minirocket + lstm, + the
   predict_mean floor) + RQ-M fairness (TimesFM).
2. **`chronos_probes_zeroshot.ipynb`** — Chronos-2: the same probes **models-only**
   (baselines run once, in session 1; scoring globs both files) + RQ-M fairness
   (Chronos-2) + **RQ-Z** (`run_zeroshot`, FD001–FD004 — Chronos-2 only: it is the
   single backbone with a registered forecaster; the four wrappers stay the §46 gap).
3. **`fairness_moment_ttm_moirai.ipynb`** — RQ-M fairness for MOMENT/TTM/Moirai-2,
   MODEL-parameterized, one model per runtime cycle (3 cycles; ttm/moirai keep the
   §43/§44 restart-after-install caveat).

Recorded probe decisions (result-affecting, all passed as explicit Config overrides in
the notebooks): RQ-A `context ∈ {32, 64, 128, 192, 256}` on FD004 (the §5 anchor) +
FD001 (the §12 finer-grid caveat; 256 = campaign cache hit); RQ-C channel subsets on
FD001 `{all21, default14 (=FD001_NONCONSTANT_SENSORS, cache hit), top8, min4}` —
`all21` deliberately includes the FD001-constant channels (junk-channel tolerance);
RQ-E `max_rul ∈ {125 (cache hit), 200}` on FD001+FD004 with BOTH losses, cross-cap
comparison via the `*_unclipped` columns only (clipped metrics are per-cap); RQ-H on
FD001 (the §5 anchor) `{clean (cache hit), gaussian 30/20/10 dB, drift 1.0, dropout
0.1}`; RQ-M fairness anchors FD001 + FD004 (single- vs multi-condition), native arm =
campaign cache hit, common arm = one new pass per (model, dataset). Concurrency: every
session writes per-session CSVs (`probe_<factor>_<tag>.csv`,
`representation_fairness_<tag>.csv`) so no two Drive sessions ever append to one file;
the later scoring pass (a `score.ipynb` follow-up, core runtime) globs them together
with `cell_fields=('dataset','n_units','factor','level')`. All stages restartable; no
result numbers recorded here — no runs happen in this change (Task 2.5).

## 48. Milestone-2 notebooks, Colab run round 1: the head/baseline deps the Stage-A stacks lack
First live run of §47 failed at `heads.compute_loss` with
`ModuleNotFoundError: No module named 'coral_pytorch'` (RQ-E label-cap probe, `corn`
arm). **Root cause: a §47 design error, not a stack bug.** `requirements/<model>.txt` is
an *embedding-only* stack — the §45 Stage-A notebooks ran `stages=['cache']` and never
trained a head, so nothing there carries the head/baseline deps. §47's probes call
`run_factor_probe`, which builds the cache **and trains the head (and runs baselines) in
the same runtime**, so they need `coral-pytorch` (CORN) and `lightgbm` (gbm). Notebook-
only fix; no `src/` change.

- **Top-up install cell** added after the backbone install in both probe notebooks:
  `pip install --no-deps coral-pytorch` (+ `lightgbm` in the TimesFM session). **`--no-deps`
  is load-bearing** — coral-pytorch declares torch, and a plain install could re-resolve
  the pinned torch/torchvision that §43/§44 fixed. lightgbm has no torch/numpy pin, so it
  installs plainly. The cell **imports both and prints the torch version**, so a missing
  dep fails at setup rather than mid-probe.
- **`minirocket` DROPPED from the probe baseline roster** (TimesFM session now
  `['gbm', 'lstm', 'predict_mean']`), superseding §47's roster line. `DECISION
  (uncited):` it needs sktime + numba, whose numpy pins fight the backbone stacks, and it
  buys nothing at the probes' operating point — `run_factor_probe` runs at the FULL fleet
  (`n_units=None`), and in every full-fleet campaign cell (§46) the strongest baseline was
  `gbm`/`gbm_age`, never minirocket (which only led at n=10). `gbm` carries the win-rule
  bar; the hollow guard still gets `predict_mean`.
- **Session 3 (`fairness_moment_ttm_moirai.ipynb`) needs NO top-up**, now stated
  explicitly in the notebook: `run_representation_fairness` hardcodes the `mse` arm
  (`src/sweep.py`) and runs no baselines, so neither package is imported — which is what
  keeps Moirai-2's `torch==2.4.1` and TTM's `torch==2.10.0` pins untouched.
- **Recovery is free:** every probe is restartable, so the interrupted session resumes and
  skips completed levels after installing the dep.

## 49. RQ-Z round 1: `ChronosForecaster` called a Chronos-2 API that does not exist
Second live failure of the §47 notebooks, this one a **real `src/` bug**, not notebook
wiring: `run_zeroshot` raised `ValueError: not enough values to unpack (expected 2, got
1)` at `zeroshot.py::ChronosForecaster.forecast`. Same root cause as §41 ("written from
assumed APIs and never executed") — §41 verified the four *embedders* against their real
libraries but **not** the zero-shot forecaster, which sat behind `# pragma: no cover` and
so was never executed by any test either.

- **The bug.** The code unpacked `Chronos2Pipeline.predict(...)` into `(quantiles, mean)`.
  Verified against the library source: **`predict()` returns a single `list[torch.Tensor]`**,
  each entry `(n_variates, n_quantiles, prediction_length)` — unpacking a 1-element list
  into two names is exactly the observed error. The 2-tuple contract belongs to
  **`predict_quantiles()`**, which returns `(quantile_list, median_list)` with medians
  shaped `(n_variates, prediction_length)`.
- **The fix.** `ChronosForecaster` now calls **`predict_quantiles`** and takes the median
  as the point forecast — the right API for the threshold-crossing step, which needs a
  point trajectory, not a quantile fan.
- **Why it survived, and the structural fix.** §32's coverage policy says the pragma
  covers only "the single line where a lazy backbone library is first imported"; shape
  handling above it is meant to be mock-covered. This method violated that — the import,
  the call, AND the tensor/shape handling were all inside one pragma'd body, so no test
  could reach the conversion. Split accordingly: the backbone import + call moved to
  `_predict_median` (**pragma'd, GPU-only**), while the conversion is now the module-level
  **`point_forecast_from_median`** (**no pragma, fully tested**). It mirrors the embedder's
  known-good `detach().to("cpu").float().numpy()` chain (GPU + bfloat16 safe — the
  previous `np.asarray` on a CUDA tensor would have been the *next* failure) and **fails
  loud** on a short forecast per the §7 contract, instead of silently returning a
  truncated series.
- **Tests (4 new, `tests/test_zeroshot.py`).** GPU/bf16-style tensor conversion via a fake
  tensor carrying the `.detach()/.to()/.float()/.numpy()` chain; plain-array + truncation
  path; the fail-loud short-forecast branch; and `forecast()` delegating to a patched
  `_predict_median` so the whole conversion runs on CPU with no backbone.
  `src/zeroshot.py` is at **100% line + branch coverage**; `pytest tests/test_zeroshot.py`
  is green (14 tests).
- **Scope.** RQ-Z remains **Chronos-2 only** (the other four backbones still have no
  registered forecaster — the §46 gap is unchanged).

## 50. Batched embedding forward passes (MOMENT / TimesFM / Moirai / TTM) — Stage-A throughput
The first real Stage-A run on Colab (an L4) exposed a throughput bug: the four v2 backbones
embedded **one series at a time (batch size 1)** inside a nested per-(window × channel)
Python loop, so a full C-MAPSS pass was ~250k serialized forward passes and the GPU sat
~95% idle (MOMENT 1.6 GB / Moirai 0.3 GB / TimesFM 1.1 GB of 22.5 GB). Chronos-2 was fine —
it is multivariate-native and already batched whole windows via `Chronos2Pipeline.embed()`.
The verify spikes (§43/§44) only ran 4 windows, so the inefficiency was invisible.

**Fix — batch the forward passes; identical math, no cache/result change.** New shared
helpers on `TSFMEmbedderBase` (`src/models/base.py`, CPU-covered):
- `_grouped_forward(items, shape_key, forward_fn)` — groups the prepared per-series inputs
  by tensor shape (`shape_key`), sub-chunks each group to `embed_batch_size` (so GPU memory
  scales with the batch, not the dataset), runs **one** backbone call per chunk, and
  scatters outputs back to the original order. Because every item in a chunk is the same
  shape, the batched forward is exactly the batch-1 result stacked — no padding, no changed
  arithmetic.
- `_regroup_channels(flat, channels_per_window)` — restacks the window-major per-channel
  outputs into the per-window canonical `(n_variates, patches, d_model)` tensors.

Each `_encode_batch` (still the sole `# pragma: no cover` boundary) now builds its
per-series inputs, calls `_grouped_forward` with a batched `forward_fn`, and regroups:
MOMENT stacks every `(window, channel)` series (all share `seq_len=512`) into one
`embed()`; TimesFM/Moirai group series by patch count (few distinct values → a handful of
calls) — each Moirai element stays an independent single-variate sequence, so the packed
attention is unchanged; TTM (multivariate-native) stacks whole windows (uniform `(ctx, C)`)
into one forward. `embed_batch_size` now meaningfully bounds GPU memory (lower it if a
large model OOMs).

**Guardrails.** `tests/test_models.py` covers `_grouped_forward` (order preservation,
shape-homogeneous batching, `batch_size` sub-chunking) and `_regroup_channels`, plus an
end-to-end `_BatchedFakeEmbedder` asserting the embeddings are **byte-identical across
`embed_batch_size` ∈ {1, 2, 4, 100}** — the CPU guarantee behind the speedup. On real
weights, `scripts/verify_backbones_colab.py` now embeds 12 mixed-length windows and adds a
**batch-invariance check** (default batched path vs `batch_size=1`), so a re-run of the
`notebooks/verify/*.ipynb` confirms the batched shapes/scatter on GPU before a full campaign.
The pooled embeddings are unchanged, so **no cache key, CSV, or recorded result changes**
(FD001 stays `windows_FD001_1da313c871251cec`); `pytest -q` green.


**Recovered onto `main` (this commit).** The fix merged into `claude/c-mapss-colab-campaign-d5yut6` (PR #17) ~51 min AFTER that branch had already merged to `main` (PR #16), and the retarget its own description called for never happened — so it sat unmerged while §45/§46's whole campaign, and the first milestone-2 sessions, ran unbatched (observed: MOMENT at **1.9 windows/s**). Cherry-picked verbatim (code identical to PR #17); only this section is renumbered 46 -> 50 to avoid colliding with §46. Caches stay valid: embeddings are byte-identical across `embed_batch_size`, so no completed work is re-embedded.

## 51. Milestone-2 close-out: 100% coverage, the scoring notebook, and the §48 minirocket correction
Closes the Milestone-2 acceptance gate (IMPLEMENTATION_PLAN §5). Three parts: the coverage
gate reaches its recorded target, the deferred §47 scoring pass gets its notebook, and the
§48 audit trail is corrected to match what actually ran on Colab. No cache key, CSV schema,
or recorded result changes — the FD001 keys stay byte-identical
(`windows_FD001_1da313c871251cec`), and no numbers are written anywhere in the repo.

### (a) The coverage gate is met: `pytest -q --cov=src --cov-branch` → 100% line + branch
Was 85.3% with 154 tests; now 100% with 229, all CPU-only, no GPU, no downloads.
`.coveragerc` is **unchanged** (`fail_under = 100`), and the only `# pragma: no cover` in
`src/` is still the §32 lazy-backbone-import boundary — **no pragma was added**, and no
existing test was modified or renumbered. Six new test modules, in the established style
(`tests/synthetic.py` fixtures + mock embedders/forecasters):

- **`tests/test_plots_v1.py`** — the v1 Stage-C figures, which were the bulk of the gap
  (`src/plots.py` 44% → 100%). `plot_ablation`; `plot_horizon` for BOTH a saturating
  (`>= max_rul`) and a fully-closed bin arm; `plot_horizon_trajectories` (cap line, model
  filter, a model absent from one unit, the §20 seed-fallback notice, and every guard:
  mixed datasets, mixed label caps, an unavailable unit count, a header-only file);
  `plot_transfer` (zero-shot reference lines, a zero-shot-only file, an empty file);
  `plot_learning_curves` (per-loss panels, unparseable stems, the no-curves error); plus
  the style helpers and `plot_success_map`'s explicit `condition_field`. Same contract as
  `test_plots_v2.py`: Agg backend, tiny synthetic CSVs, `show=False`, `tmp_path` outputs,
  assert the files exist.
- **`tests/test_core_edges.py`** — the fail-loud guards and non-default branches of the
  core modules: `config` validation + `num_channels` + the unknown-dataset error; `data`
  windowing edges (every unit shorter than the window → correctly-shaped empties,
  `pad_short=False` dropping short test units, the varlen `units=` filter); `heads`
  (unknown-loss guards in all three entry points, a 4-layer head, `scale_targets=False`
  decoding for mse and quantile, an unknown `corn_decoding`); `train` (non-deterministic
  seeding, a torch whose `use_deterministic_algorithms` raises, the default-seed path);
  `baselines` (the abstract interface, the registry's unknown-name error, the
  no-validation NN training branch); `evaluate` (git state outside a repo, the
  other-model/other-loss skip in `paired_seed_ttest`, a missing results file, unfiltered
  `aggregate_data_scaling`, `archive_results_v1` idempotency, `load_learning_curve` on
  both the step and epoch axes); and `embeddings` (the `Embedder` protocol stubs,
  per-item and stacked-leading-axis loc/scale normalization, both fail-loud shape errors,
  the registry-resolved Stage-A path with its throughput print, the missing-cache error,
  and a partial npz).
- **`tests/test_dataset_edges.py`** — the loader guards a real download can trip.
  XJTU-SY: an empty bearing folder, a single-column snapshot, depth-1 descent giving up on
  an unrelated tree, a partially-downloaded condition set, no condition folder at all, a
  bearing too short to truncate, and a split that holds out every bearing. N-CMAPSS: a
  missing and an ambiguous per-dataset file, per-file `is_available`, a dev/test unit-id
  collision inside one file, the silent (`verbose=False`) parse and cache-hit paths, a
  test unit too short to truncate, and the DSALL member guards (no root, fewer than two
  files, a non-member name).
- **`tests/test_sweep_edges.py`** — the per-baseline window override (only the named
  baseline is re-windowed), the default baseline roster, the pre-loaded-cache entry point,
  ablation restart safety (a rerun may add cells but never recomputes one), the empty
  `select_best_ablation_cell` guard, `run_baseline_window_comparison` (restartable), the
  above-fleet unit-count skip in `run_fairness_baselines`, the `HeadFeatureBuilder`
  fit-before-transform contract and `output_dim` (asserted against the width `transform`
  actually produces), and transfer's multi-condition warning + oversized-shot skip.
- **`tests/test_horizon_campaign_edges.py`** — Stage A-H resolved through the model
  registry (with its throughput print), the missing-horizon-cache error, pre-loaded
  caches + the above-fleet skip, the §20 metrics/predictions out-of-sync guard, and the
  campaign running a partial stage list (`horizon` + `figures`, no sweep), merging a user
  `dataset_overrides` over the recorded defaults, and honouring the explicit `{}` opt-out
  (XJTU-SY runs at the base protocol instead of its recorded §30 defaults).
- **`tests/test_cache_keys.py`** — the FD001 **stable-key guard** invariant §1.2 always
  described but which no test pinned. It asserts the recorded window and embedding keys
  verbatim, that every v2 key field (`channel_aggregation`, `noise_injection`,
  `noise_seed`) is ABSENT from the key dicts at its default (and present the moment it is
  set), and that paths, `experiment_name`, Stage-B head knobs, and other families' split
  protocols never enter an FD001 key.

**Two unreachable defensive branches were made reachable** — the only `src/` changes in
this section. Both strictly improve the error a caller sees; neither changes a working
path, a cache key, or a result:
- `plots.plot_horizon_trajectories` guarded "no prediction rows for n_units=…" *after*
  deriving the available unit list from those same rows, so the guard could never fire —
  a header-only predictions file instead died inside `max()` with
  `max() arg is an empty sequence`. It now fails loud on the empty selection, naming the
  file and the remedy (§7: fail loud with the observed state).
- `embeddings.extract_loc_scale` reshaped each per-item entry to exactly
  `(n_variates, 2)`, which made its final shape check vacuous: an entry carrying the
  wrong variate count raised numpy's `cannot reshape array of size …` instead. Entries
  are now reshaped to `(-1, 2)`, so the module's own
  `loc/scale normalized to X, expected Y` message fires — the informative error the
  check was written for. Every previously-working input is byte-identical.

### (b) `notebooks/campaign/milestone2/score.ipynb` — the deferred scoring pass
The follow-up §47 promised, in the same folder as its three session notebooks. **Core
runtime only**: `pip install -r requirements.txt`, no backbone stack, no GPU — every input
is a cached CSV on Drive, so no TSFM is ever loaded (`chronos-forecasting` is not even
imported). Same setup shape as its siblings: clone the repo from GitHub → mount Drive →
build the canonical §12-winner `Config` (`tsfm_context_length=256`, `pooling='mean'`,
`head_features='emb+locscale'`), of which it uses only `results_dir` and the standard
`figures_dir()` / prefix helpers. It reuses tested `src/` functions unchanged — **no
`src/` change was needed to render any of it**.

- **Factor probes (RQ-A / RQ-C / RQ-E / RQ-H).** Globs `probe_<factor>_*.csv` for
  `factor ∈ {context, channels, label_cap, noise}` — the TimesFM and the Chronos-2 session
  files TOGETHER, which is exactly what the per-session file split (§47) was for — and
  scores each with `scoring.success_map` at
  `cell_fields=('dataset', 'n_units', 'factor', 'level')`, giving win / tie / loss / hollow
  per `(dataset, model, factor, level)` against the strongest competitor baseline in that
  cell with the paired-seed test (`nasa_clipped` primary, `predict_mean` driving the
  hollow guard). Emits **`probe_success_map.csv`** to the Drive results dir and one
  `plots.plot_success_map` heatmap per factor (faceted per dataset, `show=False`,
  `prefix='probe_<factor>_'`) to the Drive figures dir, plus a verdict tally per
  (factor, model, loss arm).
  **One loss arm at a time:** `win_verdict` keys cells WITHOUT `loss`, so the RQ-E
  `label_cap` probe — the one factor that ran both `mse` and `corn` — would otherwise
  collapse both arms onto a single `{seed: value}` map and silently score only whichever
  row was read last. Each TSFM loss arm is therefore scored separately (the baselines,
  always `native`, join every arm) and recorded in a `loss_arm` column; factors with a
  single arm are unaffected.
- **RQ-M fairness.** Reads every `representation_fairness_*.csv` present (chronos,
  timesfm, moment, ttm, moirai), dedupes on `(dataset, model, mode, seed, loss)`, and
  emits **`rq_m_fairness_summary.csv`** — tidy seed-mean, spread and seed count per
  `(dataset, model, mode, channel_aggregation, metric)` for both protocol metrics — plus
  `plots.plot_cross_tsfm` native-vs-common figures **per dataset**: FD001
  (single-condition) and FD004 (multi-condition) are different regimes, so pooling their
  bars would average across the very contrast the arm exists to show.
- **RQ-Z zero-shot.** Scores `zeroshot.csv` with
  `scoring.success_map(..., compare_to_floors=True)` in BOTH `nasa_clipped` and
  `rmse_clipped`, and annotates each row with the seed-mean of EACH floor
  (`predict_mean`, `cycle_reg`) and the signed margin against both — so "which floor was
  the tougher bar" is readable per dataset, not just which one the rule selected. Emits
  **`rq_z_summary.csv`** and one figure: the same `plot_success_map` renderer, re-labelled
  so the rows are the two protocol metrics and the columns the four datasets.
- **Degrades gracefully.** Every section inventories its inputs first and prints a notice
  naming exactly what is missing — including which fairness backbones are still to come
  from the remaining `fairness_moment_ttm_moirai.ipynb` (TTM / Moirai-2) runtime cycles —
  instead of raising, so the notebook is useful before the campaign is fully finished and
  can simply be re-run afterwards.
- Every artifact lands under `<DRIVE>/results/` and `<DRIVE>/results/figures/`; per-arm
  and per-dataset intermediates go in a `results/scoring_arms/` subfolder, and no derived
  file is named so that the `probe_*` / `representation_fairness_*` globs could re-pick it
  up. **Nothing is written into the repo** (invariant §1.6).

### (c) Correction to §48: `minirocket` was RETAINED in the probe runs, not dropped
§48 recorded minirocket as **DROPPED** from the probe baseline roster, leaving
`['gbm', 'lstm', 'predict_mean']`. The completed run contradicts that record: **every**
`probe_<factor>_timesfm.csv` on Drive carries `minirocket` rows at `loss='native'`
alongside gbm / lstm / predict_mean, for all four factors. The audit trail is corrected
here to match what actually ran — the shared probe baseline roster is
**`gbm + lstm + minirocket + predict_mean`**, i.e. §47's `probe_roster` foil pair (gbm,
minirocket) intact — and §48's roster line is superseded by this one. Nothing else in §48
changes: the `coral-pytorch` / `lightgbm` top-up install that section exists for was the
real fix and stands, as does the note that session 3 needs no top-up.
Scoring is unaffected by design: the win-rule takes the STRONGEST competitor per cell, so
an extra baseline can only make the bar tougher, never weaker, and `score.ipynb` simply
scores whichever roster the files contain.

`README.md`: the notebook layout block gains a `campaign/milestone2/` entry listing all
four notebooks, a new "Milestone-2 completion" subsection describes the three-GPU-session
+ one-core-scoring-pass split, and the coverage-gate paragraph drops its "reports below
100% by design" phasing note now that the gate is met.

## 52. Milestone 3 — XJTU `xjtu_feature_mode`: raw-vs-indicators (RQ-D)
The direct test of RESEARCH_PLAN RQ-D — *"do TSFMs make hand-crafted condition
indicators obsolete?"* — on the one dataset that ships raw waveforms (XJTU-SY: 32768
samples per axis per minute at 25.6 kHz). Until now `datasets/xjtu.py` emitted ONLY the
16 hand-crafted indicators, so the question could not be asked.

- **`xjtu_feature_mode ∈ {indicators, raw, raw+indicators}`** (`config.py`, default
  `indicators` = the historical behaviour). `raw` emits `2 · xjtu_raw_channels` channels
  (`h_raw_0…`, `v_raw_0…`), `raw+indicators` emits the raw block followed by the 16
  indicators. `Config.default_sensor_columns()` resolves the channel set from the mode,
  and `datasets/xjtu.xjtu_channel_columns` is its frame-side twin (same order, asserted
  in tests) — so switching arms is ONE field, not a hand-copied channel list.
- **Two documented reductions (`xjtu_raw_reduce`)**, because "raw" alone confounds two
  different collection choices:
  - `decimate` (default) — `xjtu_raw_channels` **evenly-spaced real samples**, first and
    last inclusive. Every emitted number is a reading that was actually taken, so this is
    subtractive in the strictest sense (RESEARCH_PLAN §1) and is exactly what a
    practitioner recording at the corresponding lower rate would hold.
  - `segment_rms` — RMS within each of that many contiguous equal segments, preserving
    the snapshot's FULL-RATE energy at coarser time resolution: the
    aggregation-coarsening arm, which is XJTU's RQ-G lever (RESEARCH_PLAN §5).
  Running both is what separates *"the TSFM lost because of the sampling RATE"* from
  *"…because of the REPRESENTATION"*. `# DECISION (uncited)` records both.
- **`xjtu_raw_channels = 16` per axis** (`# DECISION (uncited)`): it puts the raw arm's
  channel count (32) in the same order of magnitude as the indicator arm's (16), so the
  comparison is not confounded by a 1000×-wider input.
- **Fails loud, never fabricates.** `snapshot_raw` raises when a snapshot has fewer
  samples than the requested width instead of padding or repeating (invariant §7).
- **Cache keys.** The three fields join the window key **only when the mode is not
  `indicators`**, and only for the xjtu family — so the recorded XJTU key
  (`windows_XJTU-SY_97e96700cc2670b4`) and every C-MAPSS/N-CMAPSS key are byte-identical
  (asserted in `tests/test_cache_keys.py`).
- **Probe wiring.** `probes.CHANNEL_SET_FACTORS = ("feature_mode", "aggregation")`: these
  factors change which columns the LOADER emits, but `sensor_columns` is resolved eagerly
  in `__post_init__` and `replace` carries the resolved list forward — so each level also
  resets `sensor_columns=None` to re-resolve. Without this the probe would ask the loader
  for columns it no longer emits. An explicit `sensor_columns` in a level still wins.
- `_bearing_frame` now takes the `config` (it selects the channel block); the two
  existing edge tests were updated for the signature and nothing else.

## 53. Milestone 4 — N-CMAPSS aggregation granularity (RQ-G)
The "how finely must you sample, and how should sub-cycle data be aggregated?" chapter on
the dataset that has genuine sub-cycle data (1 Hz within each flight).

- **`ncmapss_agg_stride`** (default 1) sub-samples each flight's 1 Hz rows 1-in-N BEFORE
  the statistics are computed — stride 10 is a 0.1 Hz recorder. Striding is WITHIN each
  flight (`cumcount % stride == 0`), so row 0 of every flight is always kept and no
  flight can become empty.
- **`ncmapss_agg_stats`** (default `mean_std`) selects the per-cycle statistic set from
  `NCMAPSS_AGG_STAT_SETS`: `mean_std` (37 channels, the historical constant) or
  `mean_std_minmax_slope` (91). `ncmapss_feature_columns()` derives the names, and
  `NCMAPSS_FEATURE_COLUMNS` is now that function at the default — byte-identical to the
  old literal.
- **`cycle_len_s` stays the FULL 1 Hz row count even under a stride**
  (`# DECISION (uncited)`): flight duration is observable from the flight's start/end
  times no matter how fast the sensors were polled, so deriving it from the retained rows
  would confound the sampling-rate intervention with the silent loss of a duration
  covariate — two different collection choices.
- **Slope** is the least-squares slope against the within-flight second index, computed
  by the algebraic `cov(t,x)/var(t)` identity (no per-group Python loop over millions of
  rows) with the products frame built in float32 and **only when a slope is requested**,
  so the default path pays nothing. A group whose `t` has zero variance (a single
  retained row) gets slope 0.0 — the same convention `std` already uses for 1-row cycles.
- **The per-file aggregate cache is keyed by the knobs in its FILENAME**
  (`ncmapss_agg_<ds>_v1[_s<stride>][_<stats>].npz`), so every pre-§53 cache file keeps its
  exact name and stays valid while each knob combination gets its own coexisting
  aggregate. `NCMAPSS_AGG_VERSION` still guards changes to the aggregation LOGIC.
- Both fields join the window key **only when non-default**, ncmapss-only — `DS02` and
  `DSALL` keys are byte-identical.

## 54. Milestone 5 — the censoring machinery + the alarm target (shared by MetroPT/Backblaze)
The realistic industrial case RESEARCH_PLAN §4 calls for: a fleet that is **mostly
healthy**, with rare failures and many right-censored survivors. Forcing RUL regression
on such a fleet invents a failure date the data does not contain, so a second, censoring-
aware target is added alongside the RUL spine. Built once here, and consumed by the
MetroPT-3 loader below and by Backblaze (§56).

### (a) The label: `alarm_horizon` → `failure_within_horizon`
`data.add_alarm_label` turns "will this unit reach an intervention within H cycles?" into
a per-row 0/1/NaN label, where `r` is the row's time to the end of its unit's observed run:

| unit | condition | label | why |
|---|---|---|---|
| observed event | `r ≤ H` | **1** | the run really ends in an intervention |
| observed event | `r > H` | **0** | it does, but not yet |
| right-censored | `r > H` | **0** | provably survived the whole horizon — a genuine negative, so the survivor **does** contribute training signal |
| right-censored | `r ≤ H` | **NaN** | the horizon runs past the end of observation; whether it failed there is **unknowable** |

The NaN rows are **dropped** (`data.drop_unlabeled_rows`), never guessed. Labelling them 0
is the classic censoring bug: it teaches "healthy" from absence of evidence and inflates
precision. `data.EVENT_OBSERVED_COLUMN` is the flag a censored loader emits; every family
that omits it is treated as "all events observed", so nothing changes for the
run-to-failure datasets. Standard administrative-censoring treatment for a fixed-horizon
binary target.

- `alarm_horizon` is a **window-cache-key field** (it changes the labels AND drops rows) —
  added only when set, so every run-to-failure key is byte-identical.
- `Config` enforces **`alarm_horizon < max_rul`**: the label is read off the RUL target,
  which is clipped at `max_rul`, so a horizon at/after the clip point would be invisible.

### (b) The arm: `heads.ALARM_LOSS = "failure_within_horizon"`
One logit + `binary_cross_entropy_with_logits`. `heads.alarm_targets` derives the binary
target from the RUL tensor, so the whole pipeline keeps ONE target array end to end and
the head and the baselines can never disagree about the label. `heads.decode` returns a
**probability**, not a RUL — the single arm whose output is on a different scale — and
`heads.is_alarm_loss` is how callers distinguish it.

`train.train_head` early-stops on the validation **BCE** for this arm (an RMSE against RUL
labels would be meaningless); a new `history["val_score"]` carries the arm's
early-stopping criterion and equals `val_rmse` exactly for every regression arm, so every
recorded result is unchanged. `val_rmse` is `nan` on the alarm arm rather than a fake
number.

### (c) The metric: precision/recall at a lead time, never tabled against NASA
`evaluate.alarm_metrics` (precision / recall / F1 / specificity at a threshold, plus the
threshold-free AUROC and average precision and the Brier score),
`evaluate.alarm_lead_times` (the lead time actually bought on the events that were
CAUGHT — what turns a recall number into MetroPT's operational requirement), and
`evaluate.alarm_threshold_sweep` (the whole operating trade-off, the alarm arm's answer to
the cost curve — no single arbitrary cut point). Degenerate inputs report `nan` rather
than raising or scoring 0, which is routine under 1-in-23,500 imbalance.

`ALARM_METRIC_FIELDS` is deliberately **disjoint** from `METRIC_FIELDS`, and
`sweep.run_alarm_sweep` writes **`alarm_results.csv`, a different file** from
`results_v2.csv` — so alarm and RUL numbers can never be averaged into one table or
plotted on one axis. RESEARCH_PLAN §8's non-comparability rule is enforced structurally,
not by convention.

### (d) The competitors: alarm baselines
`baselines.AlarmBaseRateBaseline` (`alarm_base_rate`) predicts the training base rate —
AUROC 0.5 by construction, the alarm arm's "you learned nothing" floor, which is what
makes the hollow guard reachable. `alarm_gbm` and `alarm_catch22_gbm` are LightGBM
**classifiers** over the window statistics and the catch22 indicator bank respectively
(so RQ-D's "are indicators enough?" is asked of the alarm question too). A training draw
containing one class degrades to that constant rate with the same interface instead of
raising — the honest answer, and it keeps an unlucky low-data cell from killing a sweep.

### (e) The win-rule learns direction
Alarm and RQ-F metrics are **skill scores (higher is better)**; every RUL metric is an
error. `scoring.HIGHER_IS_BETTER_METRICS` + `metric_is_higher_better` resolve the
direction in ONE place, used by both the strongest-bar selection and the margin's sign, so
`margin > 0` still means "the TSFM is better" in either direction. Getting this wrong
would have silently inverted every alarm verdict. `FLOOR_MODEL_PREFERENCE` picks the first
floor PRESENT in a cell (`predict_mean` for RUL, `alarm_base_rate` for alarm), and the
hollow guard's comparison flips with the direction. `_probe` joins `TSFM_SUFFIXES` for the
RQ-F rows.

### (f) Campaign routing
`campaign.RUL_ONLY_STAGES = ("fairness", "horizon")` are **skipped with a notice** on a
censored fleet (`config.is_censored_dataset()`): `cycle_reg`/`gbm_age` regress RUL on
elapsed cycles and a censored survivor has no RUL to regress, and the horizon bins are RUL
bands the alarm arm does not predict. `sweep` routes to `run_alarm_sweep`, and `figures`
to `plots.plot_alarm_scaling`.

### (g) The MetroPT-3 loader (`src/datasets/metropt.py`) — the first censored dataset
Porto Metro APU telemetry (UCI 791; Veloso et al. 2022): one flat 1.5 M-row CSV of 15
signals, **no label column at all**, plus four documented air-leak events supplied
out-of-band. Every judgment call below is a `# DECISION (uncited):` in the module.

- **Cycles**: fixed `metropt_cycle_minutes` wall-clock bins, floored against the Unix
  epoch, so bin edges depend only on the knob and never on where the file starts. The
  irregular ~10 s stream is **never reindexed onto a fixed frequency**. Channels are
  `{analog}_mean`/`{analog}_std` (7 analog) then `{digital}_duty` (8 digital) = 22 — for
  a binary channel the mean IS the fraction of the bin it was active.
- **The invisible-gap defence**: ~17.6 % of wall-clock time is simply absent from the
  file with no NaN row and no sentinel, so a bin holding fewer than
  `metropt_min_samples_per_cycle` raw rows is **dropped, never aggregated** — a sparse
  bin's mean/std is a different quantity from a full bin's.
- **Units = intervention runs** (RESEARCH_PLAN §4, "the clock resets at each
  intervention"): run *k* is the period ending at the START of event *k*, so the four
  events cut the record into five runs and `unit_number` names the event it ends at.
  Rows falling INSIDE a failure window belong to no run and are dropped — the APU is
  already failing, and the window is the intervention itself. Bins are grouped by
  `(run, bin start)` so a bin straddling a boundary splits and each part faces the
  min-samples rule on its own.
- **Censoring**: `event_observed = 1` for a run ending at a documented event, `0` for the
  right-censored tail run. **A censored run can never be a test run** — its remaining
  life is unknown, so "RUL at truncation" does not exist; the error names the censored
  runs, lists the observed ones to pick instead, and says censored runs belong in TRAIN
  where they contribute genuine alarm-negative rows.
- **`time_cycles` counts SURVIVING bins**, renumbered 1..n per run (the canonical frame
  requires consecutive 1-based cycles). A wall-clock gap therefore COLLAPSES rather than
  leaving a hole, so **every MetroPT horizon/lead-time number is in aggregated cycles,
  not hours** (~82 % time coverage ⇒ a 100-cycle RUL is ~122 wall-clock hours at the
  60-minute default). Stated in the docstring so no reader converts them wrongly.
- **`fault_type`/`fault_severity` carry the event table's own strings verbatim** (and
  `'none'` for a censored run). No ordinal severity ladder is invented: all four real
  events are one type, and fabricating a ladder would manufacture exactly the signal the
  RQ-F probe exists to measure.
- **Fail-loud schema checks**, each naming expected AND observed: the byte-exact 17-name
  header (which catches a "helpful" `DV_eletric` → `DV_electric` correction and any
  reordering), an explicit `strftime` parse (never inferred, so a D/M-swapped fork cannot
  parse "silently right"), non-numeric signal columns, NaN cells (absent time is absent
  ROWS in this file, so a NaN cell means something else broke), and any digital value
  outside {0.0, 1.0}. The meaningless `Unnamed: 0` counter is never read (`usecols`),
  while the header check still proves it was present and first. Two copies of the data
  file raise rather than silently preferring one.
- **Parsed-frame cache** `metropt_agg_v1_c<minutes>m_n<min_samples>_e<8-hex event
  digest>.npz`, storing the untruncated aggregate so the split re-applies without
  re-parsing. The **event-table digest is in the filename** because the table is a code
  constant that cannot ride the `Config` cache key yet fully reshapes the aggregate.
- **Non-comparability**: published MetroPT numbers use the raw stream under each author's
  own labelling of the failure reports (e.g. Davari et al.'s 21 windows vs. the UCI
  table's 4). These hour-binned, intervention-run, censoring-aware numbers must never
  share a table with them.

## 55. Milestone 6 — UCI Hydraulic + the RQ-F adjustment-vs-replacement probe
The chapter RESEARCH_PLAN RQ-F exists for: *can a frozen TSFM embedding separate a fault
that needs an **adjustment** from one that needs a **replacement**, with few labels, and
does it beat hand-crafted indicators?* UCI 447 is the anchor because it is the only
dataset in the study that ships a **native graded severity annotation per cycle**.

### (a) The loader (`src/datasets/hydraulic.py`)
- **Cycles/channels**: one row of every sensor file IS one 60 s cycle, so the 17 sensors'
  intra-cycle samples are reduced to `hydraulic_agg_stats` statistics (34 channels at the
  `mean_std` default). This is the same cycle-aggregation device N-CMAPSS uses (§27) and
  it is what makes the three sampling rates (100/10/1 Hz) commensurable **without
  resampling anything**. `slope` is the least-squares slope against the intra-cycle time
  axis in SECONDS, so one slope means the same physical quantity at every rate.
- **Units = contiguous label BLOCKS** (maximal runs of the same severity 4-tuple), cut on
  the RAW cycle order BEFORE unstable rows are dropped, so a dropped settling cycle
  shortens a unit instead of splitting one physical run in two. This is the only
  leakage-safe segmentation available: the data is **not shuffled** — cooler condition
  has just THREE contiguous runs in the whole record, so any chronological split makes it
  perfectly separable.
- **Severity polarity, the fix that matters**: `severity_<component>` is the ordinal index
  into `HYDRAULIC_SEVERITY_ORDER`, so **0 = healthy and higher = worse for every
  component** — cooler/valve/accumulator ladders run DOWN in raw units while pump leakage
  runs UP. Assuming one global polarity would have inverted half the RQ-F labels.
- **The action taxonomy**: `action_<component>` ∈ {0 none, 1 adjust, 2 replace}, where
  `replace` is the component's WORST severity level and `adjust` is anything in between
  (`HYDRAULIC_ACTIONS`). This is the only mapping the dataset's own ladder supports
  without inventing thresholds, and it makes components with different ladder lengths
  comparable.
- **`setting_1/2/3 = 0.0`**: the rig has one operating point, and the severities are
  LABELS — putting them in a setting column would leak the RQ-F target into the features.
- **Stratified split**: a `hydraulic_test_fraction` of blocks, **stratified by cooler
  severity** (the coarsest, most leak-prone factor) and selected by deterministic
  systematic sampling at span midpoints — no RNG, spread across the valve/pump/accumulator
  factorial that cycles inside each cooler regime. Every stratum keeps ≥ 1 train block and
  yields ≥ 1 test block; a regime with a single eligible block stays wholly in TRAIN so no
  regime appears only in test. Only blocks with ≥ `window_size + 1` cycles are ELIGIBLE
  for test (they must survive truncation); shorter blocks are **not dropped**, they stay
  in train — which is what keeps the real dataset usable at sane window sizes.
- **Geometry validated up to one shared scale factor**: `documented_width == observed · k`
  for a single k common to all 17 files. A real download gives k = 1 (and must then hold
  exactly `HYDRAULIC_N_CYCLES = 2205` cycles); a uniformly down-scaled fixture gives
  k > 1 while still proving every rate ratio is intact; a per-file mismatch cannot produce
  a common k and raises. A non-finite reading is an error, never imputed.
- Sensor files are read and reduced **one at a time as float32** (556 MB of text, ~740 MB
  as float64), with the geometry checked before anything is concatenated. Parsed-frame
  cache `hydraulic_agg_v1_<stats>.npz` stores the UNFILTERED, UNSPLIT aggregate so
  `hydraulic_drop_unstable` / `hydraulic_test_fraction` re-apply without re-parsing.
- **The RUL arm is not the point and the module says so loudly.** This is a cyclic
  controlled-fault-injection rig, not a run-to-failure fleet: there is no degradation
  trend within a block. RUL is emitted so the pipeline runs uniformly; the deliverable is
  the RQ-F probe.

### (b) The probe (`src/taxonomy.py`)
`run_taxonomy_probe(config, label_column, ...)` — few-shot classification on FROZEN
embeddings, so it costs no new backbone work:
1. Re-window the secondary label out of `data.load_prepared` with the SAME functions the
   Stage-A cache was built from, so label *i* is window *i*. The alignment is **asserted
   against the cached unit ids**, never assumed — a cache built under a different
   windowing protocol raises instead of scoring mismatched rows.
2. For each `shots` value k, draw k labelled examples **per class** (seeded); a class with
   fewer contributes all it has, and the row records `n_labelled` — what the probe
   ACTUALLY trained on, not what was requested, so a scarce terminal fault is visible.
   Sampling by ROW is correct here: the question is how many labelled EVENTS an
   organization must annotate, and the split is already unit-disjoint.
3. Fit a light linear probe (impute → standardize → multinomial logistic regression) and
   score accuracy / macro-F1 / AUROC on the unit-disjoint test rows. **The standardizer is
   fit on the few labelled rows only** — peeking at the unlabelled pool would violate the
   premise of a few-shot deployment. NaNs are imputed because the catch22 bank legitimately
   emits them for degenerate channels (LightGBM eats them natively; a linear probe cannot),
   which would otherwise crash the indicator foil and silently drop it from the comparison.
4. Repeat per FEATURE SOURCE — `embedding` vs `catch22` vs `window_stats`. **The gap
   between the embedding line and the indicator line is the RQ-F answer.**

Degenerate cells report `nan` rather than raising: a single-class draw predicts that class
constantly, and an AUROC the probe cannot compute (a class absent from its few labels) is
`nan`. `plots.plot_taxonomy` renders the few-shot curve per label; `_probe` joins
`scoring.TSFM_SUFFIXES` so the rows score under the same win-rule as every other arm.

## 56. Milestone 7 — Backblaze Drive Stats: the censored fleet at real scale
The dataset RESEARCH_PLAN §3 calls "the ideal real-world C-MAPSS alternative": daily SMART
snapshots across a large multi-model drive fleet, mostly-healthy, right-censored, and big
enough to ask *"how many **failure events** must you observe before deploying?"* at a scale
no simulation provides. It consumes the §54 censoring machinery unchanged.

- **Every column is selected BY NAME.** The schema drifts across quarters (85 columns in
  2013/2014, 197 from Q3 2023 = 11 metadata + 186 SMART) and new SMART columns are
  **inserted in ascending attribute order, not appended** — positional indexing would
  silently read a different attribute per quarter. The metadata prefix is 5, 8 or 11
  columns wide (`vault_id`/`pod_id`/`is_legacy_format` arrived Q2 2023;
  `datacenter`/`cluster_id`/`pod_slot_num` Q3 2023); an unrecognized width raises. Day
  files are found by a recursive `**/????-??-??.csv` glob because the archives nest
  inconsistently and carry `__MACOSX/` + `.DS_Store` junk.
- **`failure` is a terminal marker, not a state** — it flags the LAST day a drive was
  operational, at most once per drive, always its final row. A drive that simply STOPS
  APPEARING with its last row `failure == 0` is **right-censored** (retired, migrated, or
  the record ended), NOT a failure. Conflating the two is the classic Drive Stats analysis
  bug; telling them apart is this milestone's entire point.
- **Units/cycles**: one "cycle" = one drive-day; one "unit" = one drive keyed by
  `(serial_number, model)` — serials are not globally unique forever, so the pair is the
  key, sorted and enumerated for a stable `unit_number`.
- **Scope control first** (RESEARCH_PLAN §11): SMART availability is *model-conditional*
  (a model populates ~17–22 of 93 attributes; `smart_187`/`188` are absent on several),
  so the fleet is restricted to `backblaze_models` before anything else — within one model
  the channel set is comparable across drives. `setting_1` is the model index so
  `condition_norm=True` normalizes per model, which matters because raw SMART counters are
  on wildly different scales across vendors.
- **Cleaning**: a row with `capacity_bytes < 0` (the `-1` sentinel) is dropped **whole** —
  Backblaze's own guidance is that such a row is unreliable, not just its capacity. Empty
  SMART cells → NaN → `0.0` (documented). `backblaze_min_days` drops drives with too
  little history to window.
- **Survivor subsampling that cannot lose a failure**:
  `backblaze_max_survivors_per_model` subsamples the CENSORED drives per model, seeded
  from `config.seed`, and **every FAILED drive is always kept**. At ~4.2e-5 failures per
  drive-day (~1 in 23,500) an unsubsampled fleet is almost entirely survivors, so Stage A
  would be dominated by drives that never fail.
- **Stratified split, guarded**: the test set holds out `backblaze_test_fraction` of
  DRIVES **stratified by (model, event_observed)**, so it can never come out
  all-survivors — a plain random split of a 1-in-23,500 fleet trivially contains zero
  failures and is then unscoreable. **A test set with no observed failure raises.**
- **Gaps (`# DECISION (uncited):`)**: collection misses days. A gap of at most
  `BACKBLAZE_MAX_GAP_DAYS` COLLAPSES (`time_cycles` counts observed days, so RUL is in
  observed drive-days — the same convention MetroPT uses for its dropped bins). A LONGER
  gap means the drive left the fleet and came back, so **only the final contiguous segment
  is kept**: the earlier segment is a different life and is not silently glued on.
- **RUL is a lower bound for censored drives and the module says so**: `rul_truth` is the
  number of observed days cut off by truncation, which is the true remaining life only for
  a FAILED test drive. That is exactly what `data.add_train_rul` documents and what
  `data.add_alarm_label` consumes — the unknowable rows are dropped there, never guessed.
  Run this dataset with `alarm_horizon` set: its RUL numbers are plumbing, the
  alarm/lead-time metric is the result.
- **`max_rul` is in observed DRIVE-DAYS**, not turbofan cycles — a per-experiment choice,
  flagged like XJTU's minutes (§22).
- Parsed-frame cache keyed by a scope digest (models × SMART set × date bounds × the
  filtering knobs) so the multi-GB parse happens once per scope; `pyarrow` is added to
  `requirements.txt` for it.
- **Non-comparability**: published Drive Stats numbers use wildly varying protocols —
  different model scopes, SMART subsets, horizons, usually-undocumented censoring
  treatment, and frequently a random drive-DAY split that leaks a drive across train and
  test. Nothing here is comparable to any of them.

## 57. Phase-B close-out: the review fixes, the coverage gate, and the run notebook
Milestones 3–7 were each adversarially reviewed after implementation. **Eight confirmed
defects were found and fixed**; every one is a case where the code would have worked on
the synthetic fixture and misbehaved on the real download, which is exactly what the
review pass exists to catch. Recorded here because each fix changes behaviour.

### (a) MetroPT-3 (§54g)
1. **Censoring was inferred from the run INDEX, not from the record** (high). `observed =
   run_id <= len(events)` silently converts a right-censored run into an observed failure
   with a **fabricated `rul_truth`** on any truncated mirror — and the moment a 5th event
   is appended to the table, which the module's own schema errors instruct. Fixed to gate
   observedness on the record actually REACHING the event (`starts[run-1] <= ts.max()`),
   which makes the existing censored-test-run guard fire correctly, and to **announce**
   every event beyond the end of the record rather than reclassify silently.
2. **The aggregate cache could not tell two CSVs apart** (medium). MetroPT has one dataset
   name, so the filename was fully determined by the knobs — two data roots, a
   re-download, or a swap between the three accepted file names all collided on one cache
   and the second load served the first file's readings. A `(name, size, mtime)` source
   digest now joins the filename, and the write is atomic (`os.replace`).
3. **The invisible-gap defence was not scale-invariant** (medium). It was an ABSOLUTE row
   count while `metropt_cycle_minutes` is this dataset's RQ-G sweep lever, making the
   data-quality filter ~140× stricter at 10-minute bins than at 1440-minute ones — so the
   aggregation-granularity comparison would have measured a coverage gradient. New
   `metropt_min_bin_coverage` (default 0.5) expresses it as a FRACTION of a bin at the
   documented `METROPT_NOMINAL_CADENCE_S`; the absolute floor remains as a guard, the
   effective threshold is the max of the two, and the loader now REPORTS how many bins it
   dropped (a silent filter that never fires reads exactly like one that works).

### (b) UCI Hydraulic (§55)
4. **The stratified split aliased against the nested factorial** (high). Valve has period 4
   in block index, so at `hydraulic_test_fraction = 0.25` — the most natural sweep value of
   a keyed field meant to be swept — every span midpoint landed on the same residue and the
   test set collapsed onto ONE valve level, silently making the RQ-F chapter unanswerable.
   Fixed by stratifying on the CROSS `(cooler, taxonomy component)`, adding a **fail-loud
   coverage guard** (both sides must see ≥ 2 levels of the target), and, when the cross is
   too sparse, falling back to the TARGET component alone — never to cooler alone, which is
   precisely the stratification that aliases. Skipped strata are now reported.
5. **`rul_truth` was a zero-variance target** (high). Uniform blocks truncated at a fixed
   fraction give every test unit the same remaining count, so the predict-the-mean floor
   scores a perfect 0.0 RMSE that no model can beat — and the campaign was routing those
   numbers into the same `results_v2.csv` and the same figures as C-MAPSS. The deeper
   reading is that **this rig has no failure events at all**: a block ends because the
   experimenter changed the set-point. So the loader now emits `event_observed = 0`
   everywhere (literally true), warns loudly when `rul_truth` is constant, and the new
   `config.CLASSIFICATION_DATASET_KINDS` / `is_classification_dataset()` routes the dataset
   to the **RQ-F taxonomy probe** as its campaign deliverable, skipping every
   time-to-event stage (`campaign.TIME_TO_EVENT_STAGES`).
6. **The cache was keyed by nothing identifying the data, and its staleness guard was
   self-referential** (medium): it compared the cached aggregate against the cached
   profile, so it could never fire, and pointing the loader at another hydraulic directory
   silently returned the previous dataset — bypassing the 2205-cycle truncated-download
   assertion. A `(name, size)` fingerprint of the 18 files now joins the filename.
   Also: the cold path returned float64 while the warm path returned a float32 round-trip,
   so a resume after Stage A trained on values differing from the cached embeddings' — both
   paths now return the stored precision.

### (c) Backblaze (§56)
7. **Gap segmentation ran on the days that survived cleaning** (high), so ≥ 3 consecutive
   `capacity_bytes = -1` rows — a hole the LOADER punches — were indistinguishable from the
   drive leaving the fleet and silently discarded its entire pre-hole history. `_read_day`
   now also returns the scoped rows it dropped, and segmentation runs on **presence**.
   Relatedly, the failure-semantics check ran over a drive's whole history *before*
   segmentation, hard-aborting the serial-reuse case the gap rule exists to support; it now
   validates the segment that is actually kept.
8. **The cache hashed only the scope knobs, never the corpus** (high): a grown directory
   (another quarter unzipped) silently served a stale aggregate. The in-scope day
   inventory — by file NAME, so the cache stays location-independent — now joins the digest.
   Plus: the cache write is atomic and a corrupt cache raises a message naming the file and
   the remedy instead of a bare `BadZipFile`; and duplicate / metadata-colliding entries in
   `backblaze_smart_columns` fail loud instead of dying inside the reader with a pandas
   "Length mismatch" (not silently de-duplicated — the emitted channel ORDER is the config's
   contract).

### (d) The gate
`pytest -q --cov=src --cov-branch` → **100% line + branch coverage of `src/`, 496 tests**,
CPU-only and download-free. `.coveragerc` is unchanged (`fail_under = 100`); the only
`# pragma: no cover` additions are lazy heavy-import lines (`pyarrow.csv`), the §32
boundary. Every recorded cache key is byte-identical — `windows_FD001_1da313c871251cec`,
`windows_FD002_3a594bbc827991fe`, `windows_XJTU-SY_97e96700cc2670b4`,
`windows_DS02_ba4dfa4567c86cba`, `windows_DSALL_ec6a375602a4b1c2` — because every new
field is family-scoped and/or conditional on being non-default (asserted in
`tests/test_cache_keys.py` and `tests/test_feature_modes.py`).

New test modules: `test_feature_modes.py` (M3/M4), `test_censoring.py` (the alarm arm end
to end), `test_taxonomy.py` (RQ-F), `test_metropt.py`, `test_hydraulic.py`,
`test_backblaze.py`, and `test_phase_b_integration.py` — the acceptance test that runs
every Phase-B chapter through the real registry, loading path, Stage-A cache and scoring
with only the GPU backbone mocked.

### (e) The run notebook
`notebooks/phase_b.ipynb` (notebook-only wiring, no `src/` change): one runtime for the
three real datasets, with the per-dataset protocol printed before anything runs, the alarm
chapter scored by the direction-aware win-rule, the RQ-F curve rendered, and the RQ-D /
RQ-G factor probes behind a `RUN_PROBES` flag. `requirements.txt` gains `pyarrow`;
scikit-survival / lifelines are deliberately NOT added — the censored arm as built uses a
fixed-horizon binary target (IMPLEMENTATION_PLAN §6.3's own design), so no survival library
is imported anywhere and an unused heavy dependency is worse than a missing one.

## 58. Notebook reorganisation: the milestone run-surface, per-dataset results folders, and the Backblaze downloader

Notebook-only change — **no `src/` line moved**, so no cache key, CSV schema, or recorded
result is touched. Motivation: the first live Milestone-3 attempt (one `phase_b.ipynb`
run) produced **zero artifacts**. Post-mortem on the notebook wiring found two structural
causes rather than a code bug: (a) `phase_b.ipynb` defaulted `DRIVE` to
`MyDrive/Predictive Maintenance LSTM` while every notebook that HAS produced results
(§45/§47) writes to `MyDrive/pdm_tsfm` — with nothing under the former path, every
dataset printed its skip-notice and the run ended clean and empty; (b) the run surface
was split per dataset FAMILY (`xjtu` / `ncmapss` / `phase_b`), but the §42 constraint
says the axis that actually partitions runtimes is the BACKBONE — a family notebook can
only ever run one model without an environment rebuild, so "milestone 3, all five
models" was 3 notebooks × 5 stacks = 15 hand-edited sessions waiting to go wrong.

### (a) The reorganisation
- `notebooks/campaign/` now has one folder per milestone: `milestone_1/` (the §45 Stage-A
  ×5 + `score.ipynb`), `milestone_2/` (the §47 notebooks, folder renamed from
  `milestone2/`), `milestone_3/` (new, below).
- `notebooks/archive/` holds the retired family notebooks `xjtu.ipynb`, `ncmapss.ipynb`,
  `phase_b.ipynb` verbatim (git history preserves them; the archive keeps them openable).
  `cmapss.ipynb` stays at `notebooks/` — C-MAPSS is complete, but its gated FD001
  deep-dives remain the only home of the ablation/transfer/raised-cap studies.

### (b) `milestone_3/` — five per-model notebooks, every non-C-MAPSS dataset
`chronos.ipynb` / `moment.ipynb` / `timesfm.ipynb` / `ttm.ipynb` / `moirai.ipynb`, one
backbone per GPU runtime (§42), each running `run_campaign` for its ONE model over
XJTU-SY · DS01–DS08c · DSALL · MetroPT-3 · Hydraulic · Backblaze with the recorded §12
winner shape and `DEFAULT_DATASET_OVERRIDES` untouched. Design decisions:
- **Per-dataset results folders.** Each dataset's campaign call runs at
  `results_dir=results/<dataset>/` (pure notebook wiring — `experiment_name` and every
  artifact filename are unchanged, so the `<dataset>_<model>_…` names stay
  self-identifying and cross-model scoring globs `results/*/*_results_v2.csv`). Figures
  land in `results/<dataset>/figures/`. The flat `results/` layout of §45/§47 is left
  as-is — this convention applies from Milestone 3 on.
- **A loud preflight cell** prints FOUND (with kind / censored / classification /
  channels / alarm-horizon and the applied overrides) or MISSING per dataset before
  anything runs — the failure mode of (a) is now visible in the first screen of output.
- **`DECISION (uncited):` the on-runtime baseline roster** is `predict_mean · gbm · cnn ·
  lstm · catch22_gbm`. Embedding and head-training share one runtime here, so the §48
  precedent applies: `minirocket` (sktime + numba) is dropped — its numpy pins fight the
  backbone stacks, and §46 recorded gbm/gbm_age as the strongest baseline in every
  full-fleet cell. `catch22_gbm` joins as the hand-crafted-indicator foil (RQ-D).
  Censored fleets keep the alarm sweep's own roster (`alarm_base_rate · alarm_gbm`).
- **Top-up installs** after the backbone stack (`coral-pytorch --no-deps`, `lightgbm`,
  `pycatch22`, `h5py`, `pyarrow`), assert-imported at setup (§48 pattern).
- **Per-dataset fault isolation:** one `run_campaign` call per dataset inside
  `try/except`, so a failure in one dataset (or a mid-DSALL disconnect) never kills the
  session; a summary cell reports ok / skipped_no_data / failed per dataset.
- **Readout cells** (guarded, read-only): full-fleet RUL snapshot per dataset, the
  direction-aware alarm win-rule + `plot_alarm_scaling` for MetroPT-3/Backblaze, and the
  RQ-F macro-F1 curves + `plot_taxonomy` for Hydraulic.
- **Gated `RUN_PROBES`:** RQ-D (XJTU feature_mode) + RQ-G (DS02 aggregation) for that
  backbone, baselines only in the chronos session (§47/§48 pattern), each session writing
  its own `probe_<factor>_<tag>.csv` under the dataset's results folder.
- One Drive root for everything: `DRIVE = MyDrive/pdm_tsfm` (where the §45/§47 caches and
  results already live) with a separate `DATA_ROOT` knob defaulting to `pdm_tsfm/Data`;
  the config cell and README warn about the retired notebooks' other default path.

### (c) `notebooks/backblaze_download.ipynb`
Stdlib-only (no pip, no GPU, no repo clone): streams the four 2024 quarterly zips from
Backblaze's public bucket to the runtime's EPHEMERAL disk, extracts **only** the
`YYYY-MM-DD.csv` members onto Drive under `Data/Backblaze/` (zip nesting kept — the §56
loader globs recursively), deletes each zip before the next quarter. Restartable at
quarter granularity (complete quarters skip; partial ones re-fetch and fill only
missing/short files, compared by size); guards against `__MACOSX`/non-day members and
zip-slip paths; a verify cell reports per-quarter completeness (91/91/92/92 days for
2024) + total GB, and the space math (~40 GB for the full year) is stated up front with
`QUARTERS` as the trim knob. Rationale: the manual download→local→Drive round-trip was
the blocking step for Milestone 7's data.

README updated (layout tree + the Run-on-Colab section now leads with `milestone_3/`).

## 59. Milestone-3 notebooks, Colab run round 1: backbone install parity with §45

First live run of the §58 notebooks failed in the MOMENT session with
`ModuleNotFoundError: No module named 'momentfm'` — raised at the FIRST embed call,
i.e. after the whole XJTU-SY parse had already run. **Root cause: §58 wrote one generic
install cell (`pip install -r requirements/<model>.txt`) for all five notebooks, but the
proven §45/§43 install lines are NOT uniform.** `requirements/moment.txt`'s own header
says momentfm must be installed `--no-deps` (it hard-pins `numpy==1.25.2`, which has no
py3.12 wheel — the plain resolve fails and the package is simply never installed), and
the TTM/Moirai installs swap torch/torchvision and need a session restart before
importing anything. Notebook-only fix, no `src/` change:

- **Install cells now mirror `milestone_1/` verbatim per model:** MOMENT gets
  `pip install --no-deps -r requirements/moment.txt`; TTM and Moirai get the §43
  "Runtime ▸ Restart session, then re-run from the top" note (the milestone-3 clone
  cell is re-run-safe, so re-running from the top is cheap), including the tell-tale
  `operator torchvision::nms does not exist` symptom and its meaning.
- **The top-up assert cell now imports the backbone itself** (`chronos` / `momentfm` /
  `timesfm` / `tsfm_public` / `uni2ts`) next to the §48 head/baseline asserts, and
  **asserts `torch.cuda.is_available()`** — so a botched backbone install or a
  CUDA-less torch fails at setup in seconds instead of after a multi-minute dataset
  parse (the exact session lost in this round), and a silent CPU-embedding run is
  impossible.

## Not implemented (deliberately out of Phase-1 scope, Task 2.6)
Experiment-tracking services; CLI frameworks. No result numbers, comparisons, or
conclusions are written anywhere (Task 2.5) — recorded winners (§12) come only from
completed runs.

*(N-CMAPSS moved OUT of this list — implemented in §27; see DATASET_EXPANSION_PLAN.md.
TimesFM/MOMENT/TTM/Moirai moved OUT — implemented in §34. MetroPT-3, UCI Hydraulic and
Backblaze moved OUT — implemented in §54–§56, closing Milestones 5–7.)*

Still deliberately out of scope after Phase B: a joint multi-task model (RUL +
failure-type + lead-time predicted together) — RESEARCH_PLAN §4 keeps that for a later
phase and the RQ-F probe stays few-shot on frozen embeddings (IMPLEMENTATION_PLAN §10);
perturbation of REAL sensor readings (`noise_injection` remains sim-only, guarded loud);
and raw sub-cycle/waveform deep modelling beyond the RQ-D downsampled-raw XJTU channels
and the RQ-G aggregation sweep — the pipeline stays cycle/window-level.

