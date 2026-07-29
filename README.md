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
                   Backblaze/   the daily YYYY-MM-DD.csv files (any nesting)
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
notebooks/       One notebook per dataset family, each self-cloning the repo from GitHub
  cmapss.ipynb   and pointing at Drive for data/cache/results only — so they run in
  xjtu.ipynb     PARALLEL on separate Colab runtimes. Each: Setup (clone + mount) →
  ncmapss.ipynb  Config → Campaign (run_campaign restricted to that family). cmapss.ipynb
  phase_b.ipynb  also carries the gated FD001 deep-dives (ablation → winner, sweep,
                 raised-cap, transfer, plots). phase_b.ipynb runs the three REAL
                 industrial datasets (MetroPT-3 · Hydraulic · Backblaze) and scores the
                 two chapters they produce instead of a RUL curve: the censored
                 alarm/lead-time metric and the RQ-F few-shot taxonomy probe (§54–§56).
  verify/        One notebook per backbone: install its isolated requirements/<model>.txt
                 and run the weight-level GPU spike (scripts/verify_backbones_colab.py).
  campaign/      The cross-TSFM C-MAPSS campaign (CHANGES.md §45): Stage A per model
                 (chronos/moment/timesfm/ttm/moirai.ipynb — one GPU runtime each, embed →
                 cache to Drive) → Stage B (score.ipynb — one core runtime, read the five
                 caches, train heads + baselines incl. catch22_gbm, emit the success map,
                 cross-model data-scaling, and earliness/cost figures).
    milestone2/  The remaining C-MAPSS chapters (CHANGES.md §47, §51) — three parallel GPU
                 sessions plus one core-runtime scoring pass:
                 timesfm_probes.ipynb (RQ-A/C/E/H factor probes WITH the shared baselines
                 + RQ-M), chronos_probes_zeroshot.ipynb (the same probes models-only +
                 RQ-M + RQ-Z zero-shot), fairness_moment_ttm_moirai.ipynb (RQ-M for
                 MOMENT/TTM/Moirai-2, one model per runtime cycle), and score.ipynb —
                 the deferred scoring pass: it globs the per-session CSVs
                 (probe_<factor>_{chronos,timesfm}.csv, representation_fairness_*.csv,
                 zeroshot.csv), applies the win-rule, and writes the probe success map,
                 the RQ-M fairness summary and the RQ-Z table + their figures to Drive.
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
   FD001–FD004 · XJTU-SY · N-CMAPSS DS01–DS08c + the combined DSALL fleet · MetroPT-3 ·
   Hydraulic · Backblaze. Per combo it runs Stage A cache → data-scaling sweep → fairness
   arms → horizon eval → saved figures, each stage restartable. Per-dataset protocol
   choices come from `campaign.DEFAULT_DATASET_OVERRIDES` (CHANGES.md §30, §54–§56).
   Datasets not downloaded into `Data/` are skipped with a notice; every artifact is named
   `<dataset>_<model>_…` (e.g. `results/FD002_chronos-2_results_v2.csv`).

   **Censored fleets take a different route.** MetroPT-3 and Backblaze are mostly-healthy
   fleets with right-censored survivors, so `config.is_censored_dataset()` sends them to
   the binary **alarm** sweep (“will this unit need an intervention within
   `alarm_horizon` cycles?”) writing **`alarm_results.csv`** — a *different file* from
   `results_v2.csv`, because the alarm metrics (precision/recall/AUROC + lead time) share
   no scale with the RUL ones and must never be tabled together. The RUL-only `fairness`
   and `horizon` stages are skipped with a printed notice (CHANGES.md §54).
4. **Deep-dives** (in `cmapss.ipynb` only; optional — set `RUN_DEEP_DIVES = True` in its
   Config cell) — the single-dataset FD001 studies: the context/feature ablation, learning
   curves, the CORN-vs-MSE paired-significance table, the raised-label-cap arm (max_rul=200),
   and the FD001→FD003 cold-start transfer.

### Phase B — the three real industrial datasets (`notebooks/phase_b.ipynb`, CHANGES.md §54–§56)

`notebooks/phase_b.ipynb` runs MetroPT-3, UCI Hydraulic and Backblaze on one runtime.
**They do not all produce a RUL curve, by design** — `run_campaign` routes each to the arm
its physics supports:

| dataset | what it is | arm the campaign runs | headline artifact |
|---|---|---|---|
| **MetroPT-3** (UCI 791) | real metro-train APU; 4 documented air-leak events + a right-censored tail | binary **alarm** sweep | `*_alarm_results.csv` + alarm-scaling figures |
| **Backblaze** Drive Stats | real drive fleet; ~1 in 23,500 drive-days fails, most drives censored | binary **alarm** sweep | `*_alarm_results.csv` + alarm-scaling figures |
| **UCI Hydraulic** (447) | real rig with **no failure events at all** (faults are injected and held) | **RQ-F taxonomy probe** | `*_taxonomy.csv` + the few-shot curve |

Two consequences worth knowing before reading any number:

- **Alarm metrics are never tabled against RUL ones.** Precision/recall/AUROC + lead time
  share no scale with RMSE/NASA, so they go to a *different CSV* and the win-rule scores
  them in the reversed direction (they are skill scores, not errors). The RUL-only
  `fairness` and `horizon` stages are skipped with a printed notice.
- **Hydraulic's RUL is degenerate by construction** — its label blocks are uniformly sized,
  so `rul_truth` comes out constant and the predict-the-mean floor scores a perfect 0.0.
  The loader warns, and the campaign runs the RQ-F probe for it instead.

Each unit means something different per dataset (MetroPT: an intervention run, in binned
*hours*; Hydraulic: a constant-fault label block, in 60 s rig cycles; Backblaze: one drive,
in observed *drive-days*), so read every horizon and lead time in that dataset's own units.

### Cross-TSFM C-MAPSS campaign (`notebooks/campaign/`, CHANGES.md §45)

The five backbones have mutually incompatible dependencies (CHANGES.md §42), so the
cross-TSFM comparison splits along the embedding cache — a clean Drive hand-off — into
per-model **Stage A** and a single **Stage B**:

1. **Stage A — one GPU runtime per model.** Open `notebooks/campaign/<model>.ipynb`
   (`chronos`, `moment`, `timesfm`, `ttm`, `moirai`) on a fresh GPU runtime, install only
   that model's `requirements/<model>.txt`, and run it: it embeds FD001–FD004 with that one
   frozen TSFM and writes the `.npz` caches to Drive (`run_campaign(..., stages=["cache"])`
   plus the horizon caches). Restartable; run all five (in parallel on separate runtimes).
2. **Stage B — one core runtime.** Open `notebooks/campaign/score.ipynb`, `pip install -r
   requirements.txt` (core only, no backbones — the embeddings are cached), point `DRIVE` at
   the same folder, and run it: `run_campaign(..., models=[all five],
   stages=["sweep","fairness","horizon","figures"], baseline_names=[…,"catch22_gbm"])` reads
   the caches and trains heads + baselines, then it assembles the cross-TSFM **success map**,
   the combined cross-model **data-scaling** curves, and the **earliness / cost** figures.

Both stages build the **same canonical `Config`** for every cache-key field (the recorded
§12 winner shape: `tsfm_context_length=256`, `pooling="mean"`), so Stage B finds exactly the
caches Stage A wrote.

### Milestone-2 completion (`notebooks/campaign/milestone2/`, CHANGES.md §47/§51)

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
