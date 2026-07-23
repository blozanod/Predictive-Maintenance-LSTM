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

## 46. Batched embedding forward passes (MOMENT / TimesFM / Moirai / TTM) — Stage-A throughput
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

## Not implemented (deliberately out of Phase-1 scope, Task 2.6)
Experiment-tracking services; CLI frameworks. No result numbers, comparisons, or
conclusions are written anywhere (Task 2.5) — recorded winners (§12) come only from
completed runs.

*(N-CMAPSS moved OUT of this list — implemented in §27; see DATASET_EXPANSION_PLAN.md.
TimesFM/MOMENT/TTM/Moirai moved OUT — implemented in §34.)*

