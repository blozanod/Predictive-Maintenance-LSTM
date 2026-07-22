# Predictive Maintenance — Foundation Models vs. Specialized Models for RUL

Phase-1 pipeline (C-MAPSS FD001) for the study in `RESEARCH_PLAN.md`: frozen
Chronos-2 embeddings + an MLP head vs. from-scratch baselines, with the
infrastructure for the data-fraction × loss × seed sweeps that are the project's
centerpiece. **The plan is the source of truth**; deviations are logged in
`CHANGES.md`.

## Layout

```
Data/            One root housing every raw dataset (config.data_root); only the
  CMAPSSData/    small C-MAPSS text files are committed. Drop XJTU-SY (Data/XJTU-SY/)
  XJTU-SY/       and N-CMAPSS (Data/N-CMAPSS/, the .h5 files flat) here — they are
  N-CMAPSS/      git-ignored. XJTU-SY also loads under its zip name
                 (XJTU-SY_Bearing_Datasets) or one nesting level down.
src/
  config.py      Single Config dataclass: seeds, max_rul, window, tsfm_context_length,
                 head_features, pooling, unit-count grid, paths (data_root +
                 experiment_name + result-path helpers), model name, losses,
                 head/baseline hyperparams, and the versioned embedding-cache key.
                 Every result-affecting decision lives here, cited or tagged
                 "DECISION (uncited)".
  datasets/      Raw loaders, one module per dataset family, behind a registry:
    cmapss.py    C-MAPSS FD001–FD004 (subdir CMAPSSData); xjtu.py XJTU-SY bearings
    xjtu.py      (subdir XJTU-SY); ncmapss.py N-CMAPSS DS01–DS08d + the combined
    ncmapss.py   DSALL fleet (subdir N-CMAPSS, .h5 → per-cycle aggregates, cached).
                 __init__.load_raw dispatches by config.dataset_kind();
                 base.resolve_data_dir maps data_root + subdir candidates (or a
                 data_dir override), tolerating alternate names + one nesting level.
  data.py        Preprocessing hub + the unified load_prepared entry point: RUL labels +
                 clipping, condition-wise normalization, unit-level train/val split,
                 by-unit seeded subsampling, fixed windowing (label at window end),
                 last-cycle test windows, VARIABLE-LENGTH TSFM contexts (aligned 1:1
                 to the fixed windows), baseline channel scaler. Re-exports the loaders.
  models/        Frozen-TSFM embedders, one module per model, behind a registry:
    chronos.py   ChronosEmbedder (amazon/chronos-2). __init__.make_embedder picks the
                 class for config.model_name — the slot-in point for MOMENT/TimesFM/TTM.
  embeddings.py  Model-agnostic embedding infra: pooling (forecast_token/last_content/
                 mean/flatten, special tokens excluded from content poolings), on-GPU
                 batch pooling, per-window loc/scale capture, fp16-cached idempotent
                 disk cache. Injectable embedder (tests pass a mock).
  features.py    Head-feature assembly (emb / emb+locscale / emb+locscale+raw) with a
                 leakage-safe standardizer fit on the fraction's train split only.
  heads.py       2-layer MLP head; MSE / CORN (coral-pytorch) / quantile losses;
                 RUL<->bin mapping; expected-value & argmax ordinal decoding.
  baselines.py   Specialized/from-scratch models: predict-mean, GBM (lightgbm),
                 MiniRocket+ridge (sktime), 1D-CNN, LSTM. Same cached raw windows.
  train.py       Seeded on-GPU head training (tensor slicing, no DataLoader), early
                 stopping on val, per-step loss CSV.
  evaluate.py    RMSE / MAE / NASA score in BOTH protocols (clipped + unclipped); run
                 provenance; results-CSV (v2 schema) + curve helpers + v1 archiver.
  sweep.py       run_ablation (context × head_features + raw/pooling variants; picks
                 the winner), run_sweep (data-fraction × loss × seed at the winner),
                 run_baseline_window_comparison. Per-cell checkpointing +
                 completed-cell skipping. Never re-embeds.
  horizon.py     Horizon-stratified evaluation; transfer.py cold-start transfer;
  transfer.py    plots.py Stage C figures. All result files are prefixed with
  plots.py       config.experiment_name (config.results_path / figures_dir helpers).
tests/           CPU-only smoke tests (no GPU, no C-MAPSS download).
notebooks/       One notebook per dataset family, each self-cloning the repo from GitHub
  cmapss.ipynb   and pointing at Drive for data/cache/results only — so they run in
  xjtu.ipynb     PARALLEL on separate Colab runtimes. Each: Setup (clone + mount) →
  ncmapss.ipynb  Config → Campaign (run_campaign restricted to that family). cmapss.ipynb
                 also carries the gated FD001 deep-dives (ablation → winner, sweep,
                 raised-cap, transfer, plots).
```

## Run the tests (CPU, no download)

```bash
pip install -r requirements.txt
pytest -q
```

### Coverage gate

```bash
pytest -q --cov=src --cov-branch     # .coveragerc pins fail_under=100 for src/
```

100% line + branch coverage of `src/` is the repo gate (invariant §8). Every heavy
backbone/dataset library is lazily imported inside a `_load_*` method the CPU tests never
reach; the **only** sanctioned `# pragma: no cover` is that single lazy-import line
(everything above it is covered by mocks — `tests/synthetic.py`). See `CHANGES.md` §32.
100% is the Milestone-2 acceptance gate; Milestone 0 stands up this tooling, so until
Milestones 1–2 finish covering every module the command reports below 100% by design.

## Run on Colab

There are **three notebooks, one per dataset family** — `notebooks/cmapss.ipynb`,
`notebooks/xjtu.ipynb`, `notebooks/ncmapss.ipynb` — so each family runs on its own Colab
runtime **in parallel** (CHANGES.md §33). On Drive you keep **only the notebooks and the
data**; each notebook **clones the code from GitHub** into Colab's ephemeral disk, so you
never mirror or re-upload the repo. Open one (or several at once) and hit **Run all**:

1. **Setup** — installs, mount Drive, `git clone` the public repo into `/content`, put that
   clone on `sys.path`, print GPU. The clone is re-run-safe (fast-forwards if present); set
   `REPO_BRANCH` to run a branch other than `main`.
2. **Config** — set `DRIVE` to the Drive folder holding your `Data/` (raw datasets live under
   one `data_root`: `Data/CMAPSSData`, `Data/XJTU-SY`, `Data/N-CMAPSS`); `cache/` and
   `results/` are written there too. Defaults are the recorded FD001 ablation winner
   (CHANGES.md §12).
3. **Campaign** — `run_campaign(config, datasets=…)` restricted to that family: C-MAPSS
   FD001–FD004 · XJTU-SY · N-CMAPSS DS01–DS08c + the combined DSALL fleet. Per combo it runs
   Stage A cache → data-scaling sweep → fairness arms → horizon eval → saved figures, each
   stage restartable. Per-dataset protocol choices come from
   `campaign.DEFAULT_DATASET_OVERRIDES` (CHANGES.md §30). Datasets not downloaded into
   `Data/` are skipped with a notice; every artifact is named `<dataset>_<model>_…`
   (e.g. `results/FD002_chronos-2_results_v2.csv`).
4. **Deep-dives** (in `cmapss.ipynb` only; optional — set `RUN_DEEP_DIVES = True` in its
   Config cell) — the single-dataset FD001 studies: the context/feature ablation, learning
   curves, the CORN-vs-MSE paired-significance table, the raised-label-cap arm (max_rul=200),
   and the FD001→FD003 cold-start transfer.

## Audit the uncited decisions

```bash
grep -rn --include='*.py' "DECISION (uncited):" src/
```
