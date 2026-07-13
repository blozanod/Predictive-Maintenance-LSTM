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

**Winner (to be recorded after the Colab run).** _Not filled in here: the ablation
requires the Chronos-2 GPU embedding pass, which does not run in the code-review
environment. No result numbers are invented (Task 2.5 rigor rule). After running
Stage A2 on the L4, record the winning cell and its seed-mean ± std clipped RMSE
here, with the one-line justification (e.g. "context 120 + emb+locscale, mean
pooling: −X.X RMSE vs emb-only; raw-fusion a further −Y.Y")._

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

## Not implemented (deliberately out of Phase-1 scope, Task 2.6)
FD002–FD004 & N-CMAPSS/bearings; TimesFM/MOMENT/TTM/Moirai (the `model_name`
string + `Embedder` protocol are the slot-in points); condition-wise normalization
for multi-condition datasets; experiment-tracking services; CLI frameworks. No
result numbers, comparisons, or conclusions are written anywhere (Task 2.5) — the
ablation winner and sanity-gate outcome (§12) are placeholders to fill after the
Colab run, not fabricated.
