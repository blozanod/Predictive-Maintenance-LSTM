# Implementation Plan: Protocol v2 — native-resolution N-CMAPSS, XJTU de-artifacting, and the eval-correctness arm

**Status:** ready to implement · **Prepared:** July 2026 · **Target branch:** a fresh feature
branch off `main` (this plan document itself lives on `claude/dsall-results-review-jq681w`;
merge or cherry-pick it first, then branch).

This is a self-contained work order for an implementing agent. Read first, in this order:
`RESEARCH_PLAN.md` (protocol), `CHANGES.md` (audit trail, currently through §31),
`DATASET_EXPANSION_PLAN.md` (house style for loader work), and — **critically** —
`PAPER_COMPLETION_PLAN.md` on branch `claude/sweep-results-review-3ff8va` (the paper work
order; its Task 2 is absorbed into this plan as Task A, and a **parallel agent** is executing
its Task 4 (MetroPT-3) and Task 5 (RQ3) on a separate branch — see §9 Coordination).

The repo contract, inherited by every task below:
- every result-affecting choice is a `Config` field or a `# DECISION (uncited):` tag;
- every deviation gets the next free numbered `CHANGES.md` section (next free is **§32**;
  see §9 for numbering coordination with the parallel branch);
- every stage is restartable (per-cell checkpointing, completed-cell skipping);
- no numeric result or claim is ever written into the repo from anything but a completed run;
- cache keys are pure functions of config; recorded FD001 winners, existing cache keys, and
  CSV schemas are untouchable (`RESULTS_SCHEMA_VERSION` stays 2 — new work is new rows);
- `pytest -q` (CPU-only, no downloads) stays green after every task.

**Division of labor.** The agent writes all code and runs the CPU acceptance checks. GPU work
runs on the user's Colab (L4, high-RAM, ample budget → default to **5 seeds** and full grids
everywhere; no compute-abort criteria needed). Tasks are tagged **[agent]** / **[user-on-Colab]**.

---

## 0. Pre-registration — write this into CHANGES *before* any GPU run

This plan changes evaluation protocols on the two dataset families where the TSFM currently
looks worst. That is legitimate **only** under the following framing, which must be recorded
in the first CHANGES section this branch adds, verbatim in substance:

> The cycle-aggregation protocol for N-CMAPSS (§27) and the indicator-trend protocol for
> XJTU (§22) were deliberate information bottlenecks adopted to reuse one cycle-level
> pipeline. The fixed 0.6-of-life truncation additionally degenerates the last-cycle eval
> (predict_mean near-optimal by construction — see RESULTS_CONTEXT / the July 2026 analysis).
> Protocol v2 adds (a) a corrected truncation protocol and (b) a native-resolution
> sub-cycle arm on DS02, to test a **pre-registered, falsifiable hypothesis**:
> *H: the cycle-aggregation bottleneck, not the frozen TSFM, explains the contested
> N-CMAPSS result.* Both outcomes are reportable: H holds → the TSFM boundary sharpens to
> "needs native-resolution signals"; H fails → a genuine negative result on realistic
> fleets. **Both protocol arms (cycle-level and sub-cycle) are reported in the paper
> regardless of outcome; neither replaces the other.** Sub-cycle numbers may be described
> as "consistent with the published DS02 range" (protocol-family comparability) but are
> never tabled against published results (channel sets, windows, downsampling differ).

Any implementer or writer who finds themselves deleting or hiding the cycle-level arm because
the sub-cycle arm looks better is violating the plan. The paper's thesis is a *boundary*, and
the cycle-level results are half of the boundary.

---

## 1. Task list (each row = one commit; `pytest -q` after each)

| # | Task | Tag | Size |
|---|------|-----|------|
| A | Randomized-truncation test protocol (N-CMAPSS + XJTU + DSALL) | agent, then user | M |
| B | Native-resolution DS02 arm (`DS02SC`): sub-cycle loader + labels + sweep wiring | agent, then user | **L** |
| C | XJTU light fix: raised-cap arm + FPT-stratified evaluation | agent, then user | M |
| D | Near-failure probe: `last_content` pooling arm + CORN decode ablation | agent, then user | M |
| E | DSALL provenance assertion + scale-resolution analysis | agent (+user check) | S |
| F | `notebooks/colab_protocol_v2.ipynb` — the separate runnable branch-of-experiment notebook | agent, then user | M |

A is a strict prerequisite for the *last-cycle* numbers of B and C (they share the truncation
machinery), so do A first. B/C/D are then independent. F is assembled last but should be
grown incrementally as each task lands.

---

## 2. Task A — randomized-truncation test protocol  [agent codes, user runs]

Absorbed from `PAPER_COMPLETION_PLAN.md` Task 2; that spec is adopted **as written**, summary:

- New config fields: `test_truncation_mode: str = "fixed"` (choices `{"fixed","random"}`) and
  `test_truncation_range: tuple = (0.4, 0.9)`. Defaults keep every recorded run and cache key
  byte-identical (assert FD001's window/embedding keys unchanged; existing stable-key test
  stays green).
- `mode="random"`: per-test-unit fraction drawn from a seeded RNG keyed by
  `(dataset, unit_id, seed_base)` — deterministic, reproducible, varied across units. Guards
  and error messages identical to the fixed path. `rul_truth[unit] = n - keep`.
- Both fields join `_window_key_fields()` **only** for ncmapss/xjtu/DSALL (the conditional-key
  pattern of §27); C-MAPSS keys must not move.
- `test_truncation_mode` joins the restart keys of `results_v2` / horizon cells (the same way
  `max_rul` was added in §18) so fixed-mode and random-mode rows coexist in the same CSVs.
  Fixed-mode CSV rows are **kept**, never deleted.
- Tests: (a) fixed mode reproduces current truncation exactly; (b) random mode yields
  per-unit-varied `keep` with strictly higher test-RUL std than fixed on the synthetic fleet;
  (c) same seed → identical draws; (d) FD001 keys unchanged under both modes.

**[user-on-Colab]** Re-run the campaign for the 9 N-CMAPSS files + XJTU-SY + DSALL with
`test_truncation_mode="random"` (re-keys those caches → one fresh Stage A each; trivial on L4).

**Definition of done:** random-mode rows exist for every non-C-MAPSS dataset and the analysis
shows `predict_mean` is no longer near-optimal (the protocol discriminates again).

**Additionally (the "horizon-primary" decision, record in the same CHANGES section):** for
N-CMAPSS/XJTU/DSALL, the **all-cycles horizon eval is the primary metric** in every analysis
script and figure; last-cycle numbers (now under random truncation) are secondary. Rationale:
continuous monitoring is the deployment reality, and the horizon eval has 88–617 predictions
per cell versus 3–39. This is a reporting decision, not a code change, but it must be recorded
so figures and the paper draft follow it consistently.

---

## 3. Task B — native-resolution DS02 arm (`DS02SC`)  [agent codes, user runs]

**Scope decision (user, July 2026): DS02 only.** DS02 is the community benchmark file
(published sub-cycle baselines sit in the ~6–10 RMSE range), and it is the one file where
Chronos-CORN currently loses even on the honest horizon eval (built-in dev→test distribution
shift). One file is the sharpest hypothesis test per GPU-hour; other files stay cycle-level.

### 3.1 Dataset registration

- New dataset name **`DS02SC`** (SC = sub-cycle), new `dataset_kind() == "ncmapss_sub"`,
  new module `src/datasets/ncmapss_sub.py`, registered in `DATASET_LOADERS` /
  `DATASET_FAMILIES` exactly like §27 (the registry-drift test then covers it — keep green).
  A separate kind (not a flag on `ncmapss`) keeps every cache key, override, and results row
  unambiguous, and keeps the §27 loader untouched.
- Reuses the same on-disk file (`Data/N-CMAPSS/N-CMAPSS_DS02*.h5`) and the same h5 reading
  discipline as §27 (only `W_*`, `X_s_*`, `A_*` + name arrays; float32; `X_v`/`T`/`Y` are
  oracles and must never be read).

### 3.2 Canonical frame (the crux — every choice below is a `DECISION` tag + CHANGES)

- **Downsample 1 Hz → 0.1 Hz by mean-pooling each non-overlapping 10 s block** (partial
  trailing blocks dropped). 10× downsampling matches the common practice line in the DS02
  literature (Arias Chao et al. baselines and successors); mean-pooling (not decimation)
  avoids aliasing. Record as a cited-adjacent DECISION naming the baseline paper.
- One canonical row = one 0.1 Hz timestep. Columns: `unit_number` = engine unit;
  `time_cycles` = **consecutive downsampled-timestep index per unit** (1-based);
  `setting_1` = `Fc` (constant per unit), `setting_2` = flight-cycle number of the source
  flight (informational), `setting_3` = 0.0; sensor channels = the **18 raw channels**
  (4 `W` + 14 `X_s`), file order, no aggregation. `condition_norm` auto-OFF (continuous
  conditions; W channels are inputs).
- **Labels: per-row RUL in FLIGHT CYCLES, constant within a flight** (`rul = n_cycles_unit −
  flight_cycle_of_row`), the community convention. This breaks the repo's implicit
  "RUL = max(time_cycles) − time_cycles" derivation, so: the canonical frame for this kind
  carries an explicit `rul` column, and `data.add_train_rul` (or its call site) is extended
  to **respect a preexisting `rul` column when the dataset kind provides one**, guarded so
  the C-MAPSS/XJTU/ncmapss paths are bit-identical (test this). `max_rul=125` stays and is
  inactive (DS02 lives are 59–89 cycles) — same recorded observation as §27.
- **Windows:** `window_size = 50` rows (≈ 8.3 min of flight) for all models — the head, GBM
  stats, MiniRocket, CNN, LSTM all see the same 50×18 window; `tsfm_context_length = 256`
  rows (truncate-to-available-history, never padded), mirroring the recorded C-MAPSS winner
  shape and its already-analyzed information-set asymmetry (§12 caveats; the `gbm_age` /
  Task-3a fairness arms answer it).
- **Window subsampling:** full stride 1 gives ~5·10⁵ dev windows — fine on L4 for Stage A,
  but add `subcycle_window_stride: int = 1` to Config (in the window key for this kind only)
  so the user can thin to stride 2–5 if Drive space or head-training time ever bites.
  Default 1.
- **Parsed-frame cache** mirroring §3.3 of `DATASET_EXPANSION_PLAN.md`:
  `ncmapss_sub_agg_DS02_v{NCMAPSS_SUB_AGG_VERSION}.npz`, version constant bumped on any
  logic change, config-independent by construction.

### 3.3 Split & evaluation protocol

- Train = dev units (2, 5, 10, 16, 18, 20), full run-to-failure. Test = units 11, 14, 15 —
  the file's deliberate distribution shift is preserved, never resplit.
- **Primary eval = all-cycles horizon over the FULL, untruncated test trajectories** —
  this *is* the community protocol (test units are run-to-failure; every cycle is scored).
  The horizon cache design (§16) already embeds every test cycle; verify it works under the
  new kind and per-row labels.
- **Secondary eval = last-cycle under `test_truncation_mode="random"`** (Task A machinery;
  the fixed mode is pointless here and need not run). This keeps one
  cross-dataset-comparable last-cycle number without the degeneracy.
- Unit-count grid: auto-append rule (§29) gives {2, 5, 6}; 5 seeds; losses {MSE, CORN};
  all five baselines + both floors + `cycle_reg`/`gbm_age` fairness arms on the same windows.
- **Comparability warning (verbatim in CHANGES, §27-style):** DS02SC numbers are
  protocol-family comparable to published DS02 work (1 Hz-derived windows, full-trajectory
  eval) but not table-comparable (different channel sets, window lengths, downsampling).
  Allowed language: "consistent with the published range". Forbidden: same-table comparison.

### 3.4 Fixture + tests

`tests/synthetic.py::write_synthetic_ncmapss` already writes 1 Hz-shaped h5; reuse it.
Tests: (1) frame contract (row = 0.1 Hz step; 18 channels; consecutive `time_cycles`;
`rul` constant within flight, decreasing across flights, hits 0 in last flight);
(2) downsampling correctness vs manual numpy mean-pool (atol 1e-5); (3) `rul`-column respect
guard leaves an FD001 and a DS02 (cycle-level) frame bit-identical; (4) stride field keys the
window cache for DS02SC but not FD001; (5) end-to-end smoke `load_prepared →
build_embedding_cache(MockEmbedder) → run_sweep → run_horizon_eval` on the synthetic file.

### 3.5 What answers the hypothesis  [user-on-Colab, then agent analysis]

Run the full DS02SC grid; the analysis script (extend the existing scaling/horizon scripts)
prints, side by side: DS02 (cycle-level, horizon, random-truncation) vs DS02SC (sub-cycle,
horizon) for chronos-mse, chronos-corn, and best-of-baselines, per n_units. The pre-registered
readout: **does Chronos-2's deficit on DS02 close (or invert) at native resolution while the
baselines move less?** State the outcome in the analysis deliverable either way.

---

## 4. Task C — XJTU light fix  [agent codes, user runs]

**Scope decision (user, July 2026): light fix.** Keep the 16 indicator channels and the
§22 framing. No raw-waveform modeling, no leave-one-bearing-out CV, no %-life targets
(explicit non-goals, §8). The current "sensing-modality limit" finding is the paper's XJTU
paragraph; this task removes the two artifacts a reviewer could use to dismiss it.

1. **Raised-cap arm.** The recorded `max_rul=125` *minutes* saturates 92.5 % of test rows
   (the aggregate then measures cap-holding). Add a second recorded arm
   `max_rul=1000` for XJTU only (under random truncation, per-unit RUL = n·(1−frac) tops out
   ≈ 909 min on the default test bearings, so 1000 leaves the target effectively uncapped
   while keeping CORN's bin construction finite — with K=25 bins that is 40-min bins; keep K
   fixed and record it). `max_rul` already re-keys caches (§18) and already distinguishes
   result rows — **no schema work**, this is a notebook/campaign override plus a CHANGES
   DECISION recording why 1000. XJTU Stage A is ~8.6k windows — re-embedding is trivial.
   The 125 arm is kept and still reported (it is the "capped-planning-horizon" view).
2. **Random truncation** comes free from Task A (`test_truncation_mode="random"` on the
   xjtu path; the field is shared).
3. **FPT-stratified evaluation (analysis-side only — no training change).** Compute per test
   bearing a First-Prediction-Time: the first minute where horizontal-axis RMS exceeds
   `mean + 3·std` of that bearing's first 30 % of life, sustained for 3 consecutive minutes;
   if never triggered, FPT = end of life (record the census). This is the standard
   flat-then-cliff concession in the bearing-RUL literature (cite the XJTU-SY dataset paper's
   FPT usage; tag `DECISION` with the threshold specifics). Implement in the horizon analysis
   script: report each metric pre-FPT and post-FPT, alongside the existing distance-to-failure
   bands. **The paper's lead-time claims for bearings come from post-FPT rows only**; pre-FPT
   rows exist to show *why* (the signal is flat — no model can know).
   Tests: synthetic bearing with a step change → FPT lands at the step; flat synthetic → FPT
   = end of life and is reported as "no onset detected", never silently dropped.

**Definition of done:** XJTU horizon rows exist for {cap 125, cap 1000} × random truncation,
and the analysis emits the per-band + pre/post-FPT table. Expected (not promised) outcome:
the sensing-modality-limit story survives de-artifacting — if it *doesn't*, that is a finding;
report it.

---

## 5. Task D — near-failure probe: pooling + CORN decode  [agent codes, user runs]

Motivation (from the July 2026 DSALL horizon analysis; re-derive before quoting): Chronos-2
is the best arm trajectory-wide on DSALL but the *worst* near failure — 0–25-cycle bin bias
+15 to +17 (over-predicting remaining life at end-of-life, the NASA-score catastrophic
direction) while LSTM sits at +2.5. Two mechanistic hypotheses, both cheaply testable:

1. **Mean-pooling dilutes the endgame** (the terminal cycles are a vanishing fraction of the
   pooled context). Test: run the `pooling="last_content"` arm (already a supported ablation
   value, §12) through the **horizon eval** on DSALL and DS02SC. Pooling is in the embedding
   cache key, so this is one extra Stage A per dataset (cheap on L4) plus head training.
   Readout: 0–25-bin bias and RMSE, mean vs last_content, both losses.
2. **CORN's expectation decode pulls toward mid-bins.** The head decodes RUL as the expected
   value over bin probabilities; near the boundary this is structurally biased inward. Add a
   decode variant emitted as an *additional results row* with `loss="corn-argmax"` (median /
   argmax-over-bins decode — pick argmax-of-cumulative-threshold, i.e. the CORN-native rank
   decode, and record it). Same trained head, same cached logits path — decode-only, so it
   costs nothing but a small change in `train.predict_head` + the sweep emit. Rows keyed by
   the new loss value collide with nothing. (This also discharges the RESEARCH_PLAN §8
   "expected-value decoding ablation" debt.)
3. **Trajectory plots.** Extend `plot_horizon_trajectories` usage in the notebook to overlay
   chronos-mean / chronos-last_content / lstm on the same test units' final 50 cycles —
   the "when does each model notice the terminal downturn" figure. Plot code only.

Tests: decode variant is deterministic given a fixed head; `corn-argmax` rows appear with
correct restart keys; last_content horizon cells key distinctly from mean.

**Definition of done:** a small table (arm × {0–25, 25–50, 50–75, all} × {bias, RMSE}) and
the trajectory figure; a one-paragraph verdict on which hypothesis (if either) explains the
near-failure over-prediction.

---

## 6. Task E — DSALL provenance + scale-resolution analysis  [agent; user confirms]

The DSALL run exists (July 2026: 60 train units / 39 test units across the 9-file pin of
CHANGES §31). Two small pieces:

1. **[agent]** Add a fail-loud provenance assertion: when `dsall_datasets` is pinned, the
   DSALL loader already raises on missing members — additionally print and write into the
   run-metadata JSON the resolved member list *and* the resulting train/test unit counts, and
   assert the pin has ≥2 members. Test with two synthetic files. **[user]** Confirm the
   existing DSALL run's metadata JSON shows the 9-member pin and 60/39 units; if the metadata
   predates this logging, the check is re-run cheaply from caches in the notebook.
2. **[agent]** Analysis deliverable (extends the PAPER_COMPLETION_PLAN Task 1 question): from
   `results_v2` + horizon CSVs, the scale-resolution readout — *the MSE head's low-fleet
   collapse (19/45 seed-cells at 6–9 units) versus its behavior on DSALL as n_units grows
   2→60.* The July analysis indicates the collapse resolves by ~25 units (MSE ≈ CORN at 50–60)
   — i.e. the clean "CORN removes the minimum-fleet-size requirement" RQ1×RQ2 result — but
   this must be **re-derived from the CSVs by the script**, never hardcoded (repo contract).

---

## 7. Task F — `notebooks/colab_protocol_v2.ipynb`  [agent builds, user runs]

The user-facing deliverable: **one new notebook, separate from `colab_main.ipynb`**, which
stays the frozen driver of the recorded protocol-v1 runs (do not edit it in this branch).

- Structure mirrors `colab_main.ipynb` (Drive mount, pip cell, config cell, staged campaign
  cells, figure cells), built surgically as a new JSON file — never by regenerating the old
  notebook.
- Config cell sets the protocol-v2 overrides in one visible place:
  `test_truncation_mode="random"` for ncmapss/xjtu/DSALL; the `DS02SC` combo; the XJTU
  `max_rul ∈ {125, 1000}` arms; the `pooling ∈ {mean, last_content}` probe arms; the
  `corn-argmax` decode rows. Everything else inherits `DEFAULT_DATASET_OVERRIDES`.
- Cells in run order: (1) Task A reruns (9 × N-CMAPSS + XJTU + DSALL, random truncation);
  (2) DS02SC full grid; (3) XJTU raised-cap arm; (4) pooling/decode probe on DSALL + DS02SC;
  (5) Task E provenance check; (6) analysis + figures (side-by-side DS02 vs DS02SC scaling,
  XJTU pre/post-FPT bands, near-failure probe table, trajectory overlays). Every stage is
  restartable; a Run-all after interruption must skip completed cells (existing checkpointing
  handles this — verify the new restart keys participate).
- All results append to the same `results/` CSVs on Drive, distinguished by the new key
  fields/dataset names — protocol-v1 rows are never modified or deleted.
- Optional final cell (cheap, reuses caches): the `emb+locscale+age` head arm from
  PAPER_COMPLETION_PLAN Task 3a on FD002 + DS02SC + DSALL — **include only if the parallel
  branch has not already landed it** (check before coding; see §9).

---

## 8. Explicit non-goals

- No raw-waveform XJTU modeling, no leave-one-bearing-out CV, no %-life targets (user
  decision: light fix only).
- No sub-cycle arm for any file other than DS02 (user decision), and no sub-cycle DSALL.
- No MetroPT-3, no second TSFM — a **parallel agent** owns PAPER_COMPLETION_PLAN Tasks 4–5.
- No edits to `colab_main.ipynb`, recorded winners, existing cache keys, or CSV schemas.
- No deletion of any fixed-truncation or cap-125 result rows — v1 and v2 protocol rows
  coexist and are both reported.
- No numeric results written into the repo; every number in §3–§6 above is orientation from
  the July 2026 analysis session and must be re-derived from CSVs.

## 9. Coordination with the parallel branch (MetroPT / RQ3)

- **CHANGES numbering:** claim sections by PR order, not by plan order. Take the next free
  number at merge time and renumber your sections if the parallel branch lands first (§32+ is
  a starting assumption, not a reservation). Never edit the other branch's sections.
- **Config-field collisions:** Task A's `test_truncation_mode` may also be wanted by the
  MetroPT loader eventually; the field is dataset-agnostic by design (conditional key
  membership). If the parallel agent needs it before this branch merges, coordinate rather
  than fork the field.
- The `emb+locscale+age` arm (Task 3a) is speced in PAPER_COMPLETION_PLAN; whichever branch
  lands it first owns it — the notebook cell in §7 is conditional on it not existing yet.

## 10. Acceptance checklist

1. **[agent]** `pip install -r requirements.txt && pytest -q` green, CPU-only, no downloads,
   after every task commit.
2. **[agent]** `grep -rn "DECISION (uncited):" src/` shows every judgment call from §2–§5.
3. **[agent]** Stable-key regression: FD001 `window_cache_key()` / `embedding_cache_key()`
   identical to `main` under default config AND under `test_truncation_mode="random"` (the
   field must not leak into C-MAPSS keys).
4. **[agent]** Campaign dry-run on the repo as-is (only C-MAPSS data): DS02SC and all v2 arms
   report `skipped_no_data` cleanly; nothing crashes.
5. **[user-on-Colab]** Run-all `colab_protocol_v2.ipynb` on the L4: (a) Task A random-mode
   rows appear for all 11 non-C-MAPSS combos; (b) DS02SC Stage A completes and the grid
   fills at 5 seeds; (c) XJTU cap-1000 rows appear; (d) probe table + figures render;
   (e) interrupt-and-rerun skips completed cells.
6. **[user]** Confirm DSALL metadata shows the 9-member pin; report all new CSVs back for the
   analysis deliverables (§3.5, §4, §5, §6.2).
