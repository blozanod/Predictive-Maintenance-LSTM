# Predictive Maintenance — Foundation Models vs. Specialized Models for RUL

Phase-1 pipeline (C-MAPSS FD001) for the study in `RESEARCH_PLAN.md`: frozen
Chronos-2 embeddings + an MLP head vs. from-scratch baselines, with the
infrastructure for the data-fraction × loss × seed sweeps that are the project's
centerpiece. **The plan is the source of truth**; deviations are logged in
`CHANGES.md`.

## Layout

```
Data/            One root housing every raw dataset (config.data_root); only the
  CMAPSSData/    small C-MAPSS text files are committed. Everything else is git-ignored
  XJTU-SY/       — drop each download in its own subfolder:
  N-CMAPSS/        XJTU-SY/     the 3 condition folders (also accepts the zip's own
  MetroPT-3/                    name XJTU-SY_Bearing_Datasets, or one nesting level down)
  Hydraulic/       N-CMAPSS/    the .h5 files, flat
  Backblaze/       MetroPT-3/   MetroPT3(AirCompressor).csv (MetroPT3.csv also accepted)
                   Hydraulic/   the 17 sensor .txt files + profile.txt
                   Backblaze/   the daily YYYY-MM-DD.csv files (any nesting; fetched by
                                notebooks/backblaze_download.ipynb)
src/
  config.py      Single Config dataclass: seeds, max_rul, window, tsfm_context_length,
                 head_features, pooling, unit-count grid, paths (data_root +
                 experiment_name + result-path helpers), model name, losses,
                 head/baseline hyperparams, and the versioned embedding-cache key.
                 Every result-affecting decision lives here, cited or tagged
                 "DECISION (uncited)".
  datasets/      Raw loaders, one module per dataset family, behind a registry:
    cmapss.py    C-MAPSS FD001–FD004 (subdir CMAPSSData); xjtu.py XJTU-SY bearings
    xjtu.py      (subdir XJTU-SY) with the RQ-D raw-vs-indicators feature mode
    ncmapss.py   (CHANGES.md §52); ncmapss.py N-CMAPSS DS01–DS08c + the combined DSALL
    metropt.py   fleet (.h5 → per-cycle aggregates, cached) with the RQ-G aggregation
    hydraulic.py knobs (§53); metropt.py MetroPT-3 (real, CENSORED: intervention-run
    backblaze.py units + alarm target, §54); hydraulic.py UCI Hydraulic (real rig, the
                 RQ-F adjust-vs-replace anchor with native graded severity, §55);
                 backblaze.py Backblaze Drive Stats (real, censored, fleet-scale, §56).
                 __init__.load_raw dispatches by config.dataset_kind();
                 base.resolve_data_dir maps data_root + subdir candidates (or a
                 data_dir override), tolerating alternate names + one nesting level.
  data.py        Preprocessing hub + the unified load_prepared entry point: RUL labels +
                 clipping, condition-wise normalization, unit-level train/val split,
                 by-unit seeded subsampling, fixed windowing (label at window end),
                 last-cycle test windows, VARIABLE-LENGTH TSFM contexts (aligned 1:1
                 to the fixed windows), baseline channel scaler. Re-exports the loaders.
  models/        Frozen-TSFM embedders, one module per model, behind a registry.
    chronos.py   Five backbones (CHANGES.md §34): chronos.py (amazon/chronos-2,
    moirai.py    multivariate-native), moirai.py (Salesforce/moirai-2), moment.py
    moment.py    (AutonLab/MOMENT-1-large, univariate), timesfm.py (google/timesfm-2.5),
    timesfm.py   ttm.py (ibm-granite tiny channel-mixing). base.py holds the shared
    ttm.py       plain-patch TSFMEmbedderBase; the four new backbones only differ in
    ttm.py       their lazy backbone load/call. __init__.make_embedder picks the class
    base.py      for config.model_name.
  embeddings.py  Model-agnostic embedding infra: two-stage pooling (pool_patches over
                 the patch axis honoring forecast_token/last_content/mean/flatten with
                 an n_special_tokens layout knob, then aggregate_variates by
                 config.channel_aggregation — concat/mean, the RQ-M fairness knob),
                 on-GPU batch pooling, per-window loc/scale capture, fp16-cached
                 idempotent disk cache. Injectable embedder (tests pass a mock).
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
  horizon.py     Horizon-stratified evaluation + run_earliness (earliness histograms +
  transfer.py    cost curves, §37); transfer.py cold-start transfer.
  scoring.py     The win-rule (§36): strongest-baseline-per-cell, win/tie/loss/hollow
                 verdicts, and the success_map object plots.py renders.
  probes.py      Factor-probe harness (§38): run_factor_probe (channels/noise/feature_
                 mode/aggregation factor sweeps with the reduced roster) + probe_roster.
                 CHANNEL_SET_FACTORS re-resolve sensor_columns per level (§52).
  taxonomy.py    RQ-F few-shot adjust-vs-replace probe on FROZEN embeddings (§55):
                 k labels per class → linear probe, embedding vs catch22 vs window
                 stats; the gap between those curves is the RQ-F answer.
  zeroshot.py    Zero-shot health-index forecasting (RQ-Z, §39): no head, threshold
                 crossing → RUL, forecaster_factory seam.
  plots.py       Stage C figures + the v2 success-map / earliness / cost-curve /
                 cross-TSFM figures. All result files are prefixed with
                 config.experiment_name (config.results_path / figures_dir helpers).
tests/           CPU-only smoke tests (no GPU, no C-MAPSS download).
notebooks/       All Colab notebooks. Each clones the repo from GitHub into the runtime's
  cmapss.ipynb   ephemeral disk and uses Drive only for Data/ + cache/ + results/.
  backblaze_download.ipynb   cmapss.ipynb is the original C-MAPSS family notebook, kept
  verify/        for the gated FD001 deep-dives (ablation → winner, raised-cap, transfer).
  archive/       backblaze_download.ipynb streams the 2024 Backblaze Drive Stats zips and
  campaign/      extracts the daily CSVs straight onto Drive (§58). verify/ holds the
    milestone_1/ per-backbone weight-level GPU spikes (scripts/verify_backbones_colab.py).
    milestone_2/ archive/ holds the RETIRED family notebooks (xjtu, ncmapss, phase_b —
    milestone_3/ superseded by campaign/milestone_3/, CHANGES.md §58).
                 campaign/ is the run surface, one folder per milestone:
                 milestone_1/  the cross-TSFM C-MAPSS campaign (§45): Stage A per model
                               (chronos/moment/timesfm/ttm/moirai.ipynb — one GPU runtime
                               each, embed FD001–FD004 → cache to Drive) → Stage B
                               (score.ipynb — one core runtime: heads + baselines incl.
                               catch22_gbm, the success map, cross-model data-scaling,
                               and the earliness/cost figures).
                 milestone_2/  the remaining C-MAPSS chapters (§47, §51): three GPU
                               sessions (timesfm_probes.ipynb — RQ-A/C/E/H probes WITH
                               the shared baselines + RQ-M; chronos_probes_zeroshot.ipynb
                               — the probes models-only + RQ-M + RQ-Z;
                               fairness_moment_ttm_moirai.ipynb — RQ-M for the other
                               three) + score.ipynb (core runtime: probe success map,
                               RQ-M fairness summary, RQ-Z table + figures).
                 milestone_3/  the dataset scale-up (§58): FIVE per-model notebooks
                               (chronos/moment/timesfm/ttm/moirai.ipynb), each running
                               its ONE backbone over every non-C-MAPSS dataset —
                               XJTU-SY · N-CMAPSS DS01–DS08c + DSALL · MetroPT-3 ·
                               Hydraulic · Backblaze — writing per-dataset
                               results/<dataset>/ folders.
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

100% line + branch coverage of `src/` is the repo gate (invariant §8) and is **met** as of
the Milestone-2 close-out (`CHANGES.md` §51). Every heavy backbone/dataset library is
lazily imported inside a `_load_*` method the CPU tests never reach; the **only**
sanctioned `# pragma: no cover` is that single lazy-import line (everything above it is
covered by mocks — `tests/synthetic.py`). See `CHANGES.md` §32. The suite is CPU-only and
download-free, so the gate runs anywhere `pip install -r requirements.txt` succeeds.

## Run on Colab

Everything runs from Colab notebooks that **clone this repo from GitHub** into the
runtime's ephemeral disk and use Drive only for what must persist: `Data/`, `cache/`,
`results/` — one Drive root, default `MyDrive/pdm_tsfm`. Milestones 1–2 (all five
backbones on C-MAPSS) are **complete**; their notebooks stay for reproduction. The
active run surface is **`notebooks/campaign/milestone_3/`**.

### Milestone 3 — every non-C-MAPSS dataset (`notebooks/campaign/milestone_3/`, CHANGES.md §58)

**Five notebooks, one backbone per GPU runtime** — `chronos` / `moment` / `timesfm` /
`ttm` / `moirai` `.ipynb` — because the five stacks are mutually incompatible
(`requirements/README.md`). Each notebook runs `run_campaign` for its ONE frozen backbone
over XJTU-SY · N-CMAPSS DS01–DS08c + the combined DSALL fleet · MetroPT-3 · UCI
Hydraulic · Backblaze (C-MAPSS is done — milestones 1–2). Open one (or several, on
separate runtimes) and **Run all**:

- **Per-dataset results folders.** Each dataset's `run_campaign` call gets
  `results_dir=results/<dataset>/`, so every artifact lands in its dataset's own folder
  (figures under `results/<dataset>/figures/`). Filenames keep the `<dataset>_<model>_…`
  prefix, so cross-model scoring globs `results/*/*_results_v2.csv`.
- **Automatic routing** (CHANGES.md §54–§56). Run-to-failure datasets get the RUL
  data-scaling sweep + horizon eval. The censored fleets (MetroPT-3, Backblaze) get the
  binary **alarm** sweep instead — `*_alarm_results.csv`, a *different file* because
  precision/recall/AUROC + lead time share no scale with RMSE/NASA and must never be
  tabled together; the RUL-only `fairness`/`horizon` stages are skipped with a notice.
  Hydraulic (a fault-injection rig with **no failure events at all**, so its RUL is
  degenerate by construction) gets the **RQ-F taxonomy probe** → `*_taxonomy.csv`.
  Per-dataset protocol comes from `campaign.DEFAULT_DATASET_OVERRIDES`; each unit/horizon
  is in that dataset's own units (MetroPT: binned hours; Hydraulic: 60 s rig cycles;
  Backblaze: drive-days).
- **A preflight cell** prints FOUND/missing per dataset *before* anything runs — a
  dataset absent from `Data/` is skipped with a notice, never an error.
- **Restartable everywhere.** Re-running skips cached embeddings and completed sweep
  cells, so the big datasets (N-CMAPSS, DSALL, Backblaze) can be finished across several
  sessions; trim the notebook's `DATASETS` list to slice work deliberately.
- **On-runtime baseline roster** (`DECISION`, §58): `predict_mean · gbm · cnn · lstm ·
  catch22_gbm`. `minirocket` is dropped on backbone runtimes (its sktime/numba pins fight
  the backbone stacks — the §48 precedent); censored fleets use the alarm sweep's own
  roster automatically.
- **Gated probes.** `RUN_PROBES = True` adds RQ-D (XJTU raw-vs-indicators) and RQ-G
  (N-CMAPSS aggregation) for that backbone; baselines run once, in the chronos session
  (the §47/§48 pattern), and every session writes its own `probe_<factor>_<tag>.csv`.

**Getting Backblaze onto Drive:** run `notebooks/backblaze_download.ipynb` first. It
streams the 2024 quarterly zips onto the runtime's ephemeral disk and extracts **only the
daily `YYYY-MM-DD.csv` files** into `Data/Backblaze/` (no zip touches Drive, no local
round-trip). The full 2024 year is ≈ 40 GB on Drive; trim its `QUARTERS` list to take
less. Restartable: complete quarters are skipped, partial ones are filled in.

**FD001 deep-dives** (optional, `notebooks/cmapss.ipynb`, `RUN_DEEP_DIVES = True`): the
context/feature ablation, learning curves, CORN-vs-MSE paired significance, the
raised-label-cap arm (max_rul=200), and the FD001→FD003 cold-start transfer.

### Milestone 1 — cross-TSFM C-MAPSS campaign (`notebooks/campaign/milestone_1/`, CHANGES.md §45) — complete

The five backbones have mutually incompatible dependencies (CHANGES.md §42), so the
cross-TSFM comparison splits along the embedding cache — a clean Drive hand-off — into
per-model **Stage A** and a single **Stage B**:

1. **Stage A — one GPU runtime per model.** Open `notebooks/campaign/milestone_1/<model>.ipynb`
   (`chronos`, `moment`, `timesfm`, `ttm`, `moirai`) on a fresh GPU runtime, install only
   that model's `requirements/<model>.txt`, and run it: it embeds FD001–FD004 with that one
   frozen TSFM and writes the `.npz` caches to Drive (`run_campaign(..., stages=["cache"])`
   plus the horizon caches). Restartable; run all five (in parallel on separate runtimes).
2. **Stage B — one core runtime.** Open `notebooks/campaign/milestone_1/score.ipynb`,
   `pip install -r requirements.txt` (core only, no backbones — the embeddings are cached),
   point `DRIVE` at the same folder, and run it: `run_campaign(..., models=[all five],
   stages=["sweep","fairness","horizon","figures"], baseline_names=[…,"catch22_gbm"])` reads
   the caches and trains heads + baselines, then it assembles the cross-TSFM **success map**,
   the combined cross-model **data-scaling** curves, and the **earliness / cost** figures.

Both stages build the **same canonical `Config`** for every cache-key field (the recorded
§12 winner shape: `tsfm_context_length=256`, `pooling="mean"`), so Stage B finds exactly the
caches Stage A wrote.

### Milestone 2 — completion notebooks (`notebooks/campaign/milestone_2/`, CHANGES.md §47/§51) — complete

The remaining C-MAPSS chapters — the factor probes (RQ-A history, RQ-C channels, RQ-E label
cap, RQ-H sim-only noise), the RQ-M common-representation fairness ablation, and the RQ-Z
zero-shot arm — split into **three GPU sessions** (one backbone per runtime, since the stacks
cannot share an environment) plus **one core-runtime scoring pass**:

1. `timesfm_probes.ipynb` — TimesFM 2.5: every factor probe **with the shared baselines** +
   RQ-M fairness.
2. `chronos_probes_zeroshot.ipynb` — Chronos-2: the same probes **models-only** (the baselines
   ran once, in session 1) + RQ-M fairness + **RQ-Z** (`run_zeroshot`, FD001–FD004).
3. `fairness_moment_ttm_moirai.ipynb` — RQ-M fairness for MOMENT / TTM / Moirai-2, one model
   per runtime cycle.
4. `score.ipynb` — **one core runtime, no backbone** (`pip install -r requirements.txt`; every
   input is a cached CSV). It globs the per-session probe CSVs together, scores them with
   `scoring.success_map` at `cell_fields=("dataset", "n_units", "factor", "level")`, and writes
   the probe success map + heatmaps, the RQ-M native-vs-common summary + figures, and the RQ-Z
   floors table + figure to Drive. Every section degrades gracefully (a clear notice, not a
   crash) when a session's CSVs are not on Drive yet.

## Audit the uncited decisions

```bash
grep -rn --include='*.py' "DECISION (uncited):" src/
```
