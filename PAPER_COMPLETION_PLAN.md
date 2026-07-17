# Implementation Plan: complete the experiment for the paper

**Status:** ready to execute · **Prepared:** July 2026 · **Target branch:** a fresh feature
branch off `main` (do NOT reuse `claude/sweep-results-review-3ff8va`; that branch holds the
review + figures work and any PR from it is separate).

This is a self-contained instruction set for an implementing agent (a new Opus session). It
assumes you have read `RESEARCH_PLAN.md` (the research protocol), `CHANGES.md` (every recorded
protocol decision — the repo's audit trail, currently through §30), and
`DATASET_EXPANSION_PLAN.md` (the prior work order, for house style). **Read all three first.**

The repo's contract, which every task below inherits:
- every result-affecting choice is a `Config` field or a `# DECISION (uncited):` tag;
- every deviation gets the next numbered `CHANGES.md` section (**next free number is §31**);
- every stage is restartable (per-cell checkpointing, completed-cell skipping);
- **no numeric result or claim is ever written into the repo from anything but a completed
  run** — this plan quotes results from a July 2026 analysis session as *starting context*,
  not as repo content; they get re-derived from CSVs, never hardcoded;
- cache keys are pure functions of config; do not change recorded FD001 winners, existing
  cache keys, or CSV schemas (`RESULTS_SCHEMA_VERSION` stays 2 — new work is new *rows* and
  new *columns appended fail-loud*, not a schema break);
- `pytest -q` (CPU-only, no downloads) stays green after every task.

**Division of labor.** You (the agent) do all the *code* — loaders, the protocol fix,
analysis scripts, the writeup. The *GPU sweeps* run on Colab and are the **user's** action
(they have the Drive with the datasets and the T4). Each task below is tagged **[agent]** or
**[user-on-Colab]**. Mirror `DATASET_EXPANSION_PLAN.md §8`: you run the CPU acceptance checks;
the user runs the Colab sweeps and reports the CSVs back.

---

## 0. The paper, and where the evidence already stands

**Thesis.** *"Data efficiency and loss choice for frozen-TSFM RUL — when does the foundation
model win, and which loss survives sparse failures?"* Contributions: RQ1 (data-efficiency
curves / crossover) and RQ2 (ordinal vs MSE in the low-data regime) are the core; RQ4
(cross-domain) needs a real-telemetry leg; RQ3 (cross-TSFM) is either completed with one more
model or explicitly descoped.

**What the completed runs already establish** (from the analysis session; re-derive all of
these from `results/*_results_v2.csv` + `results/*_horizon*.csv` before quoting — they are
context to orient you, not facts to copy):
- **C-MAPSS, RQ1: no crossover.** Frozen Chronos-2 + small head leads the *best-of-five*
  from-scratch baseline at every fleet size on all four FD datasets (won ~25/26 cells);
  matches at 5 units what from-scratch needs 25–50 to reach (~5–10× data efficiency). Full
  data (seed-mean clipped RMSE, CORN): FD001 ≈ 10.8, FD002 ≈ 11.5, FD003 ≈ 11.3, FD004 ≈ 11.7
  — within ~1 RMSE of the published Chronos-2 adapter (10.32) with no fine-tuning.
- **RQ2: CORN stabilizes the low-data regime.** On the 9 per-file N-CMAPSS fleets (6–9 units),
  the MSE head diverged or collapsed to constant output in **19/45** seed-cells (9 fully
  constant); the CORN head on the *same cached embeddings* failed **2/45** (median RMSE
  10.5 vs 20.4). CORN ≤ MSE on all four C-MAPSS datasets; paired-by-seed t-test significant on
  FD003 (p≈0.006), FD004 (p≈0.004), and N-CMAPSS DS02/DS06.
- **Age confound, partially answered.** `gbm_age` (GBM given the elapsed-cycles feature the
  TSFM's variable-length context implicitly carries) closes only **24–33 %** of the
  Chronos-vs-GBM gap on C-MAPSS; the `cycle_reg` floor is far behind (32–41 RMSE). So on
  C-MAPSS ~70 % of the advantage is representation, not age. **This is not yet nailed shut on
  N-CMAPSS/XJTU, and there is no age-in-head arm yet** (Task 3).
- **Two known artifacts that block honest N-CMAPSS/XJTU claims** (Tasks 1–2):
  1. **DS08d file is corrupt** on Drive (truncated ~32 bytes); DSALL fails because the campaign
     pins DS08d as a member. DSALL — the ~60+-unit high-data N-CMAPSS arm — therefore **never
     ran**, and it is the single most informative missing experiment for RQ1.
  2. **The last-cycle protocol is degenerate on N-CMAPSS/XJTU.** Fixed 0.6-of-life truncation
     on a small homogeneous fleet compresses every test unit's RUL into a narrow band centered
     near the training mean, so `predict_mean` is near-optimal *by construction* over only
     3–6 test predictions per cell. The `results_v2` last-cycle numbers there are not
     trustworthy; the all-cycles **horizon** eval is (88–352 predictions/cell). Any RQ1/RQ4
     claim on these datasets must come from the horizon eval or a fixed protocol, never the
     raw `results_v2` last-cycle cells.

**The writing risk to design against** (stated so every task serves it): do not write a
"representations win everywhere" story. The honest, and stronger, story is a *boundary*: the
TSFM advantage is large and age-independent on long, smooth, multivariate C-MAPSS trajectories;
it is **loss-dependent and contested** on sparse realistic fleets; and it degrades under
distribution shift (DS02) and cliff-shaped degradation (XJTU). Every task below either sharpens
that boundary or removes an artifact that would let a reviewer dismiss it.

---

## 1. Task list (priority order; each is one commit; run `pytest -q` after each)

| # | Task | Tag | Blocks the paper? | Size |
|---|------|-----|-------------------|------|
| 1 | Fix DS08d + run DSALL (the RQ1 high-data arm) | user + agent | **Yes — core RQ1** | S (agent) |
| 2 | Randomized-truncation protocol for N-CMAPSS/XJTU test units | agent + user | **Yes — correctness** | M |
| 3 | Age confound: add age-in-head arm + write the fairness analysis | agent + user | **Yes — RQ1/RQ2 defense** | M |
| 4 | Real-telemetry dataset loader (MetroPT-3) for RQ4 | agent + user | **Yes — RQ4 gap** | L |
| 5 | RQ3 decision: add ONE TSFM *or* descope in writing | user-choice | Yes (either arm) | L or S |
| 6 | Deep-dive runs: significance table, transfer, raised-cap | user-on-Colab | No — supporting | S (already coded) |
| 7 | Results-aggregation + paper-figure notebook (reuse deck notebook) | agent | No — write-up | M |

Tasks 1–3 are the critical path and are mutually independent — do them first, in any order.
Task 4 is the biggest single lift and overlaps 100 % with the internship (same MetroPT data).
Task 5 is a decision you (the user) must make before the agent spends effort. Tasks 6–7 are
run-and-write.

---

## 2. Task 1 — Run DSALL (DS08d excluded)  [user, then agent]

**Resolved as of CHANGES §31:** DS08d (~2.9 GB) truncates on download and is not reliably
obtainable, so the DSALL default pin now excludes it (9 members: DS01–DS07 + DS08a + DS08c).
DSALL no longer fails on the missing file. **Do not block on DS08d** — it is optional; if a
verified full copy appears later, add `"DS08d"` back to the DSALL pin in `campaign.py` (a
deliberate new-key run, not a silent change).

**[user-on-Colab]** Re-run the campaign cell — every stage is restartable, so only DSALL
executes; it reuses the nine existing per-file aggregate caches (§27), so it is cheap. Report
back: `DSALL_chronos-2_results_v2.csv` and both horizon CSVs. (The per-file DS08d combo simply
reports `skipped_no_data` and needs no action.)

**[agent]** Nothing to code unless the DSALL member-pinning needs adjustment (it should not —
`DEFAULT_DATASET_OVERRIDES` already pins all 10 members, §30). **Analysis deliverable:** extend
the age/scaling analysis script (Task 3) to include DSALL and answer the load-bearing question:
*as the N-CMAPSS fleet grows from 6–9 units to ~60+, does the MSE collapse disappear and does
the CORN advantage shrink?* Two outcomes, both publishable — state which occurred:
- MSE collapse resolves at scale → "the TSFM head needs a minimum fleet size that from-scratch
  models don't; CORN removes that requirement" (a clean RQ1×RQ2 result).
- It persists → "domain mismatch, not sample size" (a different, also-useful RQ4 result).

**Definition of done:** DSALL rows present at multiple `n_units` in `results_v2` + horizon; the
scaling script prints the DSALL curve for chronos/mse, chronos/corn, and best baseline.

---

## 3. Task 2 — Randomized-truncation test protocol  [agent codes, user runs]

**Why (record verbatim in CHANGES §31 or §32):** the current fixed
`ncmapss_test_truncation=0.6` / `xjtu_test_truncation=0.6` truncates every test unit at the same
fraction of its own life. On a small homogeneous fleet this compresses test-RUL variance so far
that `predict_mean` is near-optimal by construction, which makes every model look equally
(un)skilled and destroys the discriminative power of the last-cycle protocol — the exact
opposite of what C-MAPSS's own varied `RUL_FDxxx.txt` provides. This is a measurement artifact,
not a model result, and it currently prevents any honest RQ1/RQ4 statement on N-CMAPSS/XJTU.

**Change (both loaders — `datasets/ncmapss.py`, `datasets/xjtu.py`):**
- Add config field `test_truncation_mode: str = "fixed"` (choices `{"fixed", "random"}`,
  default `"fixed"` so **all recorded runs and cache keys are byte-identical** — assert FD001's
  key is unchanged, keep the existing stable-key test green). Add
  `test_truncation_range: tuple = (0.4, 0.9)` used only when mode is `"random"`.
- When `mode == "random"`: each test unit is truncated at a fraction drawn from a **seeded**
  RNG keyed by `(dataset, unit_id, config.seed_base)` — deterministic and reproducible, varied
  across units. Guard `keep = max(window_size, floor(frac*n))`, `1 <= keep < n`, same error
  message as the fixed path. `rul_truth[unit] = n - keep`.
- Both new fields join `_window_key_fields()` **only when they affect the dataset**
  (ncmapss/xjtu/DSALL), same conditional-key pattern the existing truncation fields use
  (§27) — C-MAPSS keys must not move.
- The mode is a **DECISION (uncited)** with a CHANGES section; the fixed-mode runs are KEPT
  (they are what the last-cycle literature-style protocol used) and the random-mode runs are a
  **new arm** that shares the CSVs (the `test_truncation_mode` value distinguishes rows — add
  it to the relevant restart keys so cells don't collide, exactly as `max_rul` was added in
  §18).

**Tests:** (a) `mode="fixed"` reproduces current truncation exactly; (b) `mode="random"` yields
per-unit-varied `keep` with higher test-RUL std than fixed on the synthetic fleet; (c) same seed
→ identical truncation (determinism); (d) FD001 window/embedding keys unchanged in both modes.

**[user-on-Colab]** After the code lands: re-run the campaign for N-CMAPSS + XJTU-SY + DSALL with
`test_truncation_mode="random"` (this re-keys those caches → one fresh Stage A per dataset;
C-MAPSS untouched). Report the new `results_v2` + horizon CSVs. **Do not delete the fixed-mode
CSVs** — the paper reports both and notes the artifact explicitly.

**Definition of done:** random-mode rows exist for every non-CMAPSS dataset; the analysis script
shows `predict_mean` is no longer near-optimal (its RMSE rises relative to the trained models),
i.e. the protocol now discriminates.

---

## 4. Task 3 — Nail the age confound shut  [agent codes, user runs, agent writes]

The C-MAPSS half is already answered (`gbm_age` closes 24–33 %). Two gaps remain.

**4a. Add an age feature to the TSFM head (the symmetric fairness arm).** Today only the
baselines get an explicit elapsed-cycles feature (`gbm_age`, §19); the TSFM head gets age only
*implicitly* through context length. Add the symmetric arm so the comparison is airtight:
- Extend `features.HeadFeatureBuilder` (or add a `head_features` value
  `emb+locscale+age`) that appends the window's **last real `time_cycles`** value as one
  standardized column, fit on the fraction's train rows only (same leakage rule as loc/scale,
  §9). One `# DECISION (uncited):` + a CHANGES note. Wire it as an *optional extra sweep arm*,
  not a change to the recorded winner (`emb+locscale` stays the default).
- **The falsifiable check:** if `emb+locscale+age` ≈ `emb+locscale` (age adds ~nothing to the
  head) **and** `gbm_age` still trails Chronos, then the TSFM advantage is representational, not
  age — the reviewer's first attack is dead. If age materially helps the head, report that
  honestly; it weakens (does not kill) the representation story and reframes RQ1 partly as
  "TSFMs make engine-age legible."

**4b. Extend the age analysis to N-CMAPSS/XJTU/DSALL.** The gap-closed-by-age computation exists
for C-MAPSS; run it on the realistic datasets once Tasks 1–2 land (so it uses the non-degenerate
protocol). Write the fairness subsection from the resulting table.

**[user-on-Colab]** Run the `emb+locscale+age` arm at the full grid on FD002 + one N-CMAPSS file
+ DSALL (cheap — head-only, reuses Stage A caches). Report CSVs.

**Definition of done:** a single fairness table (dataset × {chronos-best, gbm, gbm_age,
chronos+age, cycle_reg}) with the gap-closed-by-age %, and a one-paragraph verdict per dataset
family.

---

## 5. Task 4 — Real-telemetry dataset for RQ4 (MetroPT-3)  [agent codes, user runs]

**Why MetroPT-3 over Azure PdM (record the choice).** RESEARCH_PLAN §3 lists both. Azure PdM is
a *synthetic tutorial* dataset with classification-shaped labels and no RUL leaderboard — a weak
"realistic" leg. **MetroPT-3** (Porto metro air-production unit; Veloso et al. 2022, Nature Sci
Data) is real industrial multivariate telemetry (analog pressures/currents/temperatures +
digital signals, 1 Hz) with logged failure events — genuine run-to-failure-style data, the exact
shape of the internship's press problem. Use MetroPT-3 as the RQ4 real-telemetry leg. (If the
user prefers Azure PdM as a *second, weaker* realism probe, add it after MetroPT-3, clearly
labeled exploratory — do not let it be the only real-data point.)

**Implementation — mirror `datasets/ncmapss.py` exactly (it is the closest precedent):**
- New module `src/datasets/metropt.py`, registered in `datasets/__init__.py`
  (`DATASET_LOADERS["metropt"]`, `DATASET_FAMILIES["metropt"]`); `Config.dataset_kind()` maps
  the dataset name(s) → `"metropt"`; `DEFAULT_SENSOR_COLUMNS["metropt"]` = the analog channel
  list. The registry-drift test then covers it automatically — keep it green.
- **Cycle definition (DECISION uncited, the crux — document it).** MetroPT is continuous 1 Hz,
  not cycle-structured. Choose a canonical "cycle" = a fixed time bin (e.g. 10-min or 1-hour
  window), aggregated to per-bin stats (mean+std of each analog channel + duty-cycle fraction of
  each digital signal), exactly the N-CMAPSS aggregation pattern (§27). This keeps the whole
  downstream pipeline unchanged. Parse once → cached aggregate frame
  (`metropt_agg_*_v{VERSION}.npz`, versioned like `NCMAPSS_AGG_VERSION`).
- **Labels from events (DECISION uncited).** Derive RUL as time-to-next-failure-event from the
  published failure report timestamps; units = inter-failure run segments; pre-register the
  labeling rule in the module docstring + CHANGES before running any model (RESEARCH_PLAN §8).
  `condition_norm` resolves auto-OFF (no discrete operating grid) unless a regime variable is
  identified.
- **Non-comparability warning** (verbatim in CHANGES, like §27): our binning/labeling is a
  repo-specific protocol; numbers are for same-protocol cross-model comparison (RQ1/RQ4), never
  to be tabled against MetroPT anomaly-detection papers.
- Synthetic fixture + tests mirroring the N-CMAPSS test block (canonical-frame contract,
  aggregation correctness, label derivation, cache versioning, end-to-end smoke with
  `MockEmbedder` → `run_sweep`).

**[user-on-Colab]** Drop MetroPT-3 under `Data/MetroPT/`, run the campaign combo. Report CSVs.

**Definition of done:** MetroPT rows in `results_v2` + horizon; it appears in the data-scaling
and horizon figures automatically (the campaign + faceted plots already handle new datasets).

---

## 6. Task 5 — RQ3 decision: add one TSFM or descope  [user decides first]

**This is a decision the user must make before the agent invests effort.** Present it as
written; do not default silently.

- **Option A — add MOMENT** (`AutonLab/MOMENT-1-large`). Scientifically the sharpest: it is the
  only candidate pretrained for *general representation* (masked reconstruction) rather than
  forecasting, so "does forecasting-pretraining or representation-pretraining give better RUL
  embeddings?" becomes answerable. Highest integration effort (new `src/models/moment.py`, one
  `EMBEDDERS` entry — the registry is the documented slot-in point; must expose
  `embed_windows(contexts) -> (embeddings, loc_scale)` and `describe()`). Re-run Stage A per
  dataset for the new backbone (GPU cost = one embedding pass per dataset).
- **Option B — add TTM** (IBM Tiny Time Mixers, ~1–5 M params). Cheapest integration and yields
  a clean "does scale matter, or just pretraining?" result — "TTM within X % of Chronos-2 at
  1/100 the size" is a quotable finding (RESEARCH_PLAN §4, §6 fairness).
- **Option C — descope RQ3.** Perfectly legitimate: keep RQ1/RQ2/RQ4 as the paper, and in the
  writeup change RQ3 from a question to explicit future work ("we anchor on Chronos-2; cross-TSFM
  comparison is left to future work, and the `src/models/` registry is the slot-in point"). Do
  **not** leave RQ3 written as a question with no second model — that is the one unacceptable
  state.

**Agent guidance:** if the user picks A or B, follow the `models/__init__.py` contract exactly,
add the model to `EMBEDDERS`, add a CPU smoke test with a mock, and hand the Stage-A re-embed to
the user. If C, edit `RESEARCH_PLAN.md`/the paper outline to descope and stop.

---

## 7. Task 6 — Deep-dive runs (already coded; just run)  [user-on-Colab]

These need no new code — they exist and are gated behind `RUN_DEEP_DIVES` in the notebook. Run
after Tasks 1–4 so they use the fixed protocol:
- **Paired significance table** — `evaluate.paired_seed_ttest` per (dataset, n_units, bin);
  report CORN−MSE with p-values as descriptive support (5 seeds is low-powered — never p alone).
- **Cold-start transfer** — `transfer.run_transfer_eval`, default FD001↔FD003; report zero-shot
  / target-only / source+target. Optionally add a MetroPT→(other) pair once Task 4 lands.
- **Raised label cap** — the `max_rul=200` arm (§18) for horizon honesty; keeps the 125-cap
  runs untouched and adds the 125–200 bins that test early detectability.

**Definition of done:** `transfer.csv`, the paired-test table, and the max_rul=200 horizon rows
exist; the agent folds them into the figure notebook (Task 7).

---

## 8. Task 7 — Aggregation + paper figures  [agent]

Reuse and extend `notebooks/deck_figures.ipynb` (already on the review branch) — it reads only
metric CSVs, computes all annotations at run time, and writes PNG+PDF. Add paper-specific panels:
the multi-dataset data-scaling grid (RQ1), the CORN-vs-MSE stability census across the fixed AND
random protocols (RQ2), the fairness table (Task 3), and the RQ4 real-telemetry curve. Keep the
"no hardcoded numbers" property. **Write no prose conclusions in the repo** until the runs that
back them are complete (repo contract); the figures + a `results/README` pointer are the
deliverable, and the paper draft itself lives outside the repo until numbers are final.

---

## 9. Acceptance checklist (agent runs 1–4; user runs 5–7)

1. `pip install -r requirements.txt && pytest -q` — green, CPU-only, no downloads, after each task.
2. `grep -rn "DECISION (uncited):" src/` — every new judgment call (random-truncation mode,
   age-in-head, MetroPT cycle/label choices, any new TSFM) is tagged.
3. Stable-key check: `Config(dataset="FD001")` `window_cache_key()`/`embedding_cache_key()`
   **identical to `main`** before and after every task (existing test — keep it passing).
4. New CHANGES sections appended from §31 onward; no existing section renumbered/edited.
5. **[user]** DS08d re-downloaded, DSALL + random-protocol + age-in-head + MetroPT sweeps run on
   Colab; CSVs returned.
6. **[user]** Spot-check one MetroPT unit's derived frame + label against the raw event log.
7. **[user + agent]** Re-derive every headline number in §0 from the returned CSVs; confirm the
   boundary story (C-MAPSS win age-independent; realistic data contested/loss-dependent) holds
   under the *fixed* protocol before it goes in the paper.

## 10. Explicit non-goals

- No writing of paper prose or numeric claims into the repo from incomplete runs.
- No more than one added TSFM (Task 5) — this is not the cross-TSFM survey paper.
- No raw 1 Hz / sub-cycle modeling of MetroPT or N-CMAPSS; no raw XJTU waveforms (cycle-aggregate
  framing only, consistent with the existing loaders).
- No changes to recorded C-MAPSS winners, cache keys, or CSV schemas (append columns fail-loud;
  never repurpose one).
- Do not commit any dataset files.
