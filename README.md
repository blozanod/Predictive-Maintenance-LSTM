# Predictive Maintenance — Foundation Models vs. Specialized Models for RUL

Phase-1 pipeline (C-MAPSS FD001) for the study in `RESEARCH_PLAN.md`: frozen
Chronos-2 embeddings + an MLP head vs. from-scratch baselines, with the
infrastructure for the data-fraction × loss × seed sweeps that are the project's
centerpiece. **The plan is the source of truth**; deviations are logged in
`CHANGES.md`.

## Layout

```
src/
  config.py      Single Config dataclass: seeds, max_rul, window, tsfm_context_length,
                 head_features, pooling, unit-count grid, paths, model name, losses,
                 head/baseline hyperparams, and the versioned embedding-cache key.
                 Every result-affecting decision lives here, cited or tagged
                 "DECISION (uncited)".
  data.py        C-MAPSS load, RUL labels + clipping, unit-level train/val split,
                 by-unit seeded subsampling, fixed windowing (label at window end),
                 last-cycle test windows, VARIABLE-LENGTH TSFM contexts (aligned 1:1
                 to the fixed windows), baseline channel scaler.
  embeddings.py  Chronos2Pipeline.embed() wrapper: pooling (forecast_token/last_content/
                 mean/flatten, special tokens excluded from content poolings), on-GPU
                 batch pooling, per-window loc/scale capture, fp16-cached idempotent
                 disk cache. Injectable embedder (tests pass a mock).
  features.py    Head-feature assembly (emb / emb+locscale / emb+locscale+raw) with a
                 leakage-safe standardizer fit on the fraction's train split only.
  heads.py       2-layer MLP head; MSE / CORN (coral-pytorch) / quantile losses;
                 RUL<->bin mapping; expected-value & argmax ordinal decoding.
  baselines.py   predict-mean, GBM (lightgbm), MiniRocket+ridge (sktime), 1D-CNN,
                 LSTM. All consume the same cached raw windows; parallelized.
  train.py       Seeded on-GPU head training (tensor slicing, no DataLoader), early
                 stopping on val, per-step loss CSV.
  evaluate.py    RMSE / MAE / NASA score in BOTH protocols (clipped + unclipped); run
                 provenance; results-CSV (v2 schema) + curve helpers + v1 archiver.
  sweep.py       run_ablation (context × head_features + raw/pooling variants; picks
                 the winner), run_sweep (data-fraction × loss × seed at the winner),
                 run_baseline_window_comparison. Per-cell checkpointing +
                 completed-cell skipping. Never re-embeds.
tests/           CPU-only smoke tests (no GPU, no C-MAPSS download).
notebooks/
  colab_main.ipynb   Thin orchestrator: Setup → Stage A (embed once) → Stage A2
                     (ablation → winner) → Stage B (sweep at winner) → Stage C (plots).
```

## Run the tests (CPU, no download)

```bash
pip install -r requirements.txt
pytest -q
```

## Run on Colab (cell order)

Open `notebooks/colab_main.ipynb` and run cells top to bottom:

1. **Setup** — installs, mount Drive, add repo to `sys.path`, print GPU.
2. **Config** — point `data_dir`/`cache_dir`/`results_dir` at your Drive; override
   any ablation knob (`tsfm_context_length`, `head_features`, `pooling`, losses,
   embed batch/dtype).
3. **Stage A** — `build_embedding_cache(config)`: Chronos-2 `embed()` on GPU →
   Drive cache (fp16 embeddings + loc/scale + fixed windows). Idempotent; the only
   GPU-heavy stage (run once per embedding config).
4. **Stage A2** — `run_ablation(config)`: full-data MSE grid over context ×
   head_features (+ raw-fusion arm, pooling variants); `select_best_ablation_cell`
   picks the winner by clipped RMSE.
5. **Stage B** — `run_sweep(sweep_config)`: trains heads + baselines on the cache at
   the winning config → `results_v2.csv` (both protocols). Cheap, rerunnable,
   checkpointed after every cell.
6. **Stage C** — plots the data-scaling curve (headline, clipped + unclipped) and
   learning curves.

## Audit the uncited decisions

```bash
grep -rn --include='*.py' "DECISION (uncited):" src/
```
