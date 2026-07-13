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

## Not implemented (deliberately out of Phase-1 scope, Task 2.6)
FD002–FD004 & N-CMAPSS/bearings; TimesFM/MOMENT/TTM/Moirai (the `model_name`
string + `Embedder` protocol are the slot-in points); condition-wise normalization
for multi-condition datasets; experiment-tracking services; CLI frameworks. No
result numbers, comparisons, or conclusions are written anywhere (Task 2.5) — the
ablation winner and sanity-gate outcome (§12) are placeholders to fill after the
Colab run, not fabricated.
