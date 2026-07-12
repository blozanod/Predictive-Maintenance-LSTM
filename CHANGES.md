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
loc/scale), so the embedding path needs no cross-unit scaler and its embeddings are
**independent of the data fraction**. Consequently embeddings are computed once over
*all* FD001 train units (Stage A) and cached. The no-leakage rule (Task 2.4,
plan §6) is enforced where a scaler is actually fit — the **baselines**, whose
per-channel standardization is fit on the current fraction's *train* windows only
(`data.fit_channel_scaler`).

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

## 5. Test-label clipping is off by default
`clip_test_labels=False`: RMSE/MAE/NASA are computed against the **unclipped**
ground-truth RUL from `RUL_FDxxx.txt` (the original PHM08 target). Some
piecewise-linear papers clip test labels at `max_rul`; that is available via the
config flag. Training labels are always clipped at `max_rul=125` (Heimes 2008;
Li et al. 2018).

## 6. Data fractions expressed as engine-unit counts
The notebook's row fractions `[0.1 … 1.0]` are replaced by unit counts
`data_unit_counts=[2, 5, 10, 25, 50, 100]` (plan §6): subsampling is **by unit**,
seeded, with the sampled unit IDs saved to the run directory.

## 7. Pooling default = `last_patch`
`embed()` yields one embedding per patch; the plan requires deciding/ablating
last-patch vs mean vs flatten (§1). Default is `last_patch`
(`# DECISION (uncited)` in `embeddings.py`, since the plan does not mandate one);
`mean` and `flatten` are config options and change the cache key so each is cached
independently. Note the final patch position includes the 2 register/boundary
tokens `embed()` appends.

## 8. Reproducibility & provenance (added per Task 2.3)
Seeds are threaded through python/numpy/torch/CUDA/DataLoader (`train.set_seed`);
`torch.use_deterministic_algorithms(warn_only=True)` is enabled when
`deterministic=True`. Every sweep writes `run_metadata.json` (resolved config +
git commit/describe/dirty + package versions) beside its metrics CSV, and per-cell
sampled unit IDs to `units_n{N}_seed{S}.json`.

## Not implemented (deliberately out of Phase-1 scope, Task 2.6)
FD002–FD004 & N-CMAPSS/bearings; TimesFM/MOMENT/TTM/Moirai (the `model_name`
string + `Embedder` protocol are the slot-in points); condition-wise normalization
for multi-condition datasets; experiment-tracking services; CLI frameworks. No
result numbers, comparisons, or conclusions are written anywhere (Task 2.5).
