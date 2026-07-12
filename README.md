# Predictive Maintenance — Foundation Models vs. Specialized Models for RUL

Phase-1 pipeline (C-MAPSS FD001) for the study in `RESEARCH_PLAN.md`: frozen
Chronos-2 embeddings + an MLP head vs. from-scratch baselines, with the
infrastructure for the data-fraction × loss × seed sweeps that are the project's
centerpiece. **The plan is the source of truth**; deviations are logged in
`CHANGES.md`.

## Layout

```
src/
  config.py      Single Config dataclass: seeds, max_rul, window, unit-count grid,
                 paths, model name, pooling, losses, head/baseline hyperparams,
                 and the embedding-cache key. Every result-affecting decision lives
                 here, cited or tagged "DECISION (uncited)".
  data.py        C-MAPSS load, RUL labels + clipping, unit-level train/val split,
                 by-unit seeded subsampling, windowing (label at window end),
                 last-cycle test windows, baseline channel scaler.
  embeddings.py  Chronos2Pipeline.embed() wrapper + pooling (last_patch/mean/flatten)
                 + idempotent disk cache. Injectable embedder (tests pass a mock).
  heads.py       2-layer MLP head; MSE / CORN (coral-pytorch) / quantile losses;
                 RUL<->bin mapping; expected-value & argmax ordinal decoding.
  baselines.py   predict-mean, GBM (lightgbm), MiniRocket+ridge (sktime), 1D-CNN,
                 LSTM. All consume the same cached raw windows.
  train.py       Seeded head training, early stopping on val, per-step loss CSV.
  evaluate.py    RMSE / MAE / NASA score; run provenance; results-CSV + curve helpers.
  sweep.py       data-fraction × loss × seed grid over cached embeddings/windows;
                 per-cell checkpointing + completed-cell skipping. Never re-embeds.
tests/           CPU-only smoke tests (no GPU, no C-MAPSS download).
notebooks/
  colab_main.ipynb   Thin orchestrator: Setup → Stage A (embed once) → Stage B
                     (sweeps) → Stage C (plots).
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
   any ablation knob (pooling, losses, embed batch/dtype).
3. **Stage A** — `build_embedding_cache(config)`: Chronos-2 `embed()` on GPU →
   Drive cache. Idempotent; the only GPU-heavy stage (run once per embedding config).
4. **Stage B** — `run_sweep(config)`: trains heads + baselines on the cache.
   Cheap, rerunnable, checkpointed after every cell.
5. **Stage C** — plots the data-scaling curve (headline) and learning curves.

## Audit the uncited decisions

```bash
grep -rn --include='*.py' "DECISION (uncited):" src/
```
