# Implementation Plan: N-CMAPSS + XJTU-SY — full multi-dataset experiment sweep

**Status:** ready to implement · **Prepared:** July 2026 · **Target branch:** a fresh feature branch off `main`

This document is a self-contained instruction set for an implementing agent. It assumes you
have read `RESEARCH_PLAN.md` (the research protocol) and `CHANGES.md` (every recorded
protocol decision — the repo's audit trail). **Read both first.** The repo's contract:
every result-affecting choice is a `Config` field or a `# DECISION (uncited):` tag, every
deviation gets a numbered `CHANGES.md` section, every stage is restartable, and no numeric
result/claim is ever written into the repo from anything but a completed run.

**Goal.** After this work, `run_campaign(config)` (already implemented, `src/campaign.py`)
sweeps **FD001–FD004 × XJTU-SY × N-CMAPSS DS01–DS08d (+ a combined DSALL)** × every
registered TSFM, running cache → sweep → fairness → horizon → figures per combo, restartable,
with per-dataset protocol defaults that need no hand-editing. The only user action is
dropping the raw datasets under `Data/`.

---

## 0. Ground truth about the two datasets (so you don't need the PDFs)

### 0.1 N-CMAPSS (NASA "Turbofan Engine Degradation Simulation Data Set 2")

- One `.h5` file per sub-dataset. Shipped filenames (the suffix is a generation id —
  **glob, never hardcode full names**): `N-CMAPSS_DS01-005.h5`, `N-CMAPSS_DS02-006.h5`,
  `N-CMAPSS_DS03-012.h5`, `N-CMAPSS_DS04.h5`, `N-CMAPSS_DS05.h5`, `N-CMAPSS_DS06.h5`,
  `N-CMAPSS_DS07.h5`, `N-CMAPSS_DS08a-009.h5`, `N-CMAPSS_DS08c-008.h5`,
  `N-CMAPSS_DS08d-010.h5`. Verify against what's on disk; treat the on-disk glob as truth.
- HDF5 keys per file: `W_dev, X_s_dev, X_v_dev, T_dev, Y_dev, A_dev` and the same with
  `_test`, plus name arrays `W_var, X_s_var, X_v_var, T_var, A_var` (byte strings —
  decode via `np.array(h5[k], dtype='U20')`).
  - `A` (auxiliary): columns `unit, cycle, Fc, hs` — unit id, flight-cycle number,
    flight class (1=1–3h, 2=3–5h, 3=5–7h flights), health state.
  - `W` (flight conditions, 4 cols): `alt, Mach, TRA, T2`. Continuous, not discrete.
  - `X_s` (measured sensors, **14** cols): T24, T30, T48, T50, P15, P2, P21, P24, Ps30,
    P40, P50, Nf, Nc, Wf (order may differ — **always read `X_s_var` from the file**).
  - `X_v` (virtual sensors), `T` (health-parameter ground truth θ), `Y` (per-row RUL):
    **do not load** — `X_v`/`T` are simulation oracles, and RUL is re-derived from cycle
    counts by `data.add_train_rul` exactly as for C-MAPSS.
- Sampling is **1 Hz within each flight**; one flight = one cycle; flights are thousands
  of rows. DS02: 5.3M dev rows (6 units: 2,5,10,16,18,20) + 1.2M test rows (units 11,14,15),
  end-of-life 59–89 cycles. Other files are the same shape with different unit counts and
  failure modes. Dev/test unit ids never collide within a file (assert anyway).
- Test units are provided **run-to-failure** (RUL hits 0 at the last row) — there is no
  `RUL_*.txt`. The community evaluates over the whole test trajectory; our pipeline's
  standard protocol is predict-at-last-observed-cycle. §3.4 resolves this via truncation
  (same device as XJTU-SY, CHANGES.md §22).

### 0.2 XJTU-SY bearings

- 15 bearings, 3 condition folders, 5 bearings each. **Folder names as shipped:**
  `35Hz12kN` (2100 rpm, 12 kN), `37.5Hz11kN` (2250 rpm, 11 kN), **`40Hz10kN`**
  (2400 rpm, **10 kN**). Inside: `Bearing1_1/ … Bearing3_5/`, each holding `1.csv … N.csv`,
  one CSV per minute, 32768 rows × 2 columns (horizontal, vertical accel @ 25.6 kHz).
- Lifetimes (file counts): 1_1:123, 1_2:161, 1_3:158, 1_4:122, 1_5:52, 2_1:491, 2_2:161,
  2_3:533, 2_4:42, 2_5:339, 3_1:2538, 3_2:2496, 3_3:371, 3_4:1515, 3_5:114.
- **The existing loader is broken for condition 3**: `XJTU_CONDITIONS` in
  `src/datasets/xjtu.py` maps `"40Hz12kN": (2, 40.0, 12.0)` — wrong folder name **and**
  wrong force. Effect: condition 3 silently never loads, and since the default
  `xjtu_test_bearings` includes `Bearing3_4/3_5`, `load_xjtu` raises "not on disk" —
  XJTU-SY has never actually run. Fix in §2.

### 0.3 Where the user drops the data (document this in README + notebook)

```
Data/
  CMAPSSData/            (committed)
  N-CMAPSS/              N-CMAPSS_DS02-006.h5, ... (all .h5 files, flat)
  XJTU-SY/               35Hz12kN/, 37.5Hz11kN/, 40Hz10kN/  (each with Bearing folders)
```
The user's upload may be named `XJTU-SY_Bearing_Datasets` and/or have one extra nesting
level from the zip — §2.2 makes the loader tolerant of both. `.gitignore` already ignores
everything under `Data/` except `CMAPSSData/` — no change needed there.

---

## 1. Task list (do in this order; each is one commit)

| # | Task | Files touched | Size |
|---|------|---------------|------|
| 1 | Fix XJTU condition-3 folder/force + unmatched-folder guard | `src/datasets/xjtu.py`, tests, `CHANGES.md` | S |
| 2 | Subdir candidates + one-level-nested discovery | `src/datasets/base.py`, `xjtu.py`, `cmapss.py`, `__init__.py`, tests | S |
| 3 | N-CMAPSS loader (`src/datasets/ncmapss.py`) + config/registry wiring + synthetic-h5 test fixture | new module, `config.py`, `datasets/__init__.py`, `tests/synthetic.py`, `tests/test_datasets.py`, `requirements.txt`, `CHANGES.md` | **L** |
| 4 | DSALL combined N-CMAPSS dataset | `ncmapss.py`, tests, `CHANGES.md` | M |
| 5 | Auto-append full-fleet cell to the unit-count grid | `src/sweep.py` (`run_sweep`, `run_fairness_baselines`), tests, `CHANGES.md` | S |
| 6 | `DEFAULT_DATASET_OVERRIDES` in campaign + notebook update | `src/campaign.py`, `notebooks/colab_main.ipynb`, `README.md`, `CHANGES.md` | M |
| 7 | Docs pass: README layout, CHANGES sections, run instructions | `README.md`, `CHANGES.md` | S |

Run `pytest -q` after every task (all existing 48+ tests must stay green — they are CPU-only
and need no downloads). Do not renumber or edit existing CHANGES sections; append new ones
(next free number is §25).

---

## 2. Task 1–2: XJTU-SY fixes

### 2.1 Condition table fix (Task 1)

In `src/datasets/xjtu.py`:

```python
XJTU_CONDITIONS = {
    "35Hz12kN":   (0, 35.0, 12.0),
    "37.5Hz11kN": (1, 37.5, 11.0),
    "40Hz10kN":   (2, 40.0, 10.0),   # was "40Hz12kN"/12.0 — wrong name AND force
}
```

Add a guard in `load_xjtu`: after scanning, any directory under the XJTU root that is not a
key of `XJTU_CONDITIONS` (and looks like a condition folder, e.g. matches `r"^[\d.]+Hz\d+kN$"`)
must raise a `ValueError` naming the unmatched folders and the expected keys — a renamed or
new condition folder must never be silently skipped. Non-matching stray dirs (e.g. `__MACOSX`)
are ignored.

Cache note for `CHANGES.md`: the fix changes XJTU data content (condition 3 appears,
`setting_3` becomes 10.0) without changing any cache-key field. This is safe **because no
valid XJTU cache can exist** — the old loader raised on the default test bearings — but say
so explicitly, and instruct deleting any `cache/emb_XJTU-SY_*.npz` built with a hacked
config.

Tests: `tests/synthetic.py::write_synthetic_xjtu` already builds folders from
`XJTU_CONDITIONS`, so it self-heals; add (a) a regression test asserting
`XJTU_CONDITIONS["40Hz10kN"] == (2, 40.0, 10.0)` and that `"40Hz12kN"` is absent, and
(b) a test that an unexpected `45Hz9kN` folder raises with both names in the message.

### 2.2 Tolerant directory resolution (Task 2)

Real-world layouts to accept without user surgery:

1. `Data/XJTU-SY/35Hz12kN/...` (documented layout)
2. `Data/XJTU-SY_Bearing_Datasets/35Hz12kN/...` (the zip's own name — the user's upload)
3. `Data/XJTU-SY_Bearing_Datasets/XJTU-SY_Bearing_Datasets/35Hz12kN/...` (zip-in-folder)

Implementation:
- `base.resolve_data_dir(config, subdir)` accepts `str | tuple[str, ...]`: for a tuple, return
  the first `config.data_root / candidate` that exists, else `data_root / candidates[0]`
  (so error messages name the documented layout). `config.data_dir` override still wins
  verbatim. Each family declares `SUBDIR = ("XJTU-SY", "XJTU-SY_Bearing_Datasets")`,
  `("N-CMAPSS",)`, `("CMAPSSData",)`; the `DATASET_LOADERS` registry entries carry the tuple.
- In `load_xjtu` (and `is_available`): if no condition folder exists directly under the
  resolved root, scan the root's **immediate** subdirectories for one containing condition
  folders and descend one level (print a one-line notice). Depth-1 only — no recursive walk.
- **Not part of any cache key** (paths never are — CHANGES.md §23).

Tests: synthetic XJTU written under the alternate name and under one nesting level both load.

---

## 3. Task 3: N-CMAPSS loader — `src/datasets/ncmapss.py`

The pipeline is cycle-level (RUL in cycles, windows of `window_size` cycles, Chronos-2
contexts = per-cycle multivariate series). N-CMAPSS is 1 Hz *within* flights. The adaptation
— mirroring the XJTU indicator-trend design (CHANGES.md §22) — is:
**one canonical-frame row = one flight cycle, channels = per-cycle summary statistics.**
All of the following protocol choices carry `# DECISION (uncited):` tags and one collective
`CHANGES.md` section; there is no community-standard *cycle-level* N-CMAPSS protocol.

> **Comparability warning (record it verbatim in CHANGES.md):** published N-CMAPSS RMSEs
> (e.g. Arias Chao et al. baselines, DS02 ≈ 6–10 RUL-RMSE) are computed on 1 Hz sub-cycle
> windows over full test trajectories. Our cycle-aggregated, truncation-protocol numbers are
> **not comparable to them** and must never be placed in the same table. The dataset's role
> here is same-protocol cross-model comparison (RQ1/RQ4), exactly like XJTU-SY.

### 3.1 Canonical frame mapping

| canonical column | N-CMAPSS source |
|---|---|
| `unit_number` | `A[:, unit]` (int; dev+test ids are disjoint within a file — assert, raise if not) |
| `time_cycles` | `A[:, cycle]` (int, 1-based flight index) |
| `setting_1` | `Fc` (flight class 1/2/3 — constant per unit) |
| `setting_2`, `setting_3` | 0.0 (unused; keeps `condition_keys` well-defined) |
| sensor channels | per-cycle stats of `W` (4 vars) and `X_s` (14 vars), §3.2 |

`condition_norm` resolves **OFF** for N-CMAPSS (flight conditions are continuous, not a
discrete grid; the aggregates already carry the condition information as channels). Because
`setting_1 = Fc`, a user *can* flip `condition_norm=True` to get per-flight-class
normalization — mention this in the module docstring, default stays auto-OFF. Update
`Config.effective_condition_norm()` only if you add ncmapss to the auto-ON list — **don't**.

### 3.2 Per-cycle aggregation

- For each `(unit, cycle)` group: `mean` and `std` of each of the 18 raw channels
  (4 `W` + 14 `X_s`), plus **one** extra channel `cycle_len_s` = number of 1 Hz rows in the
  cycle (observable flight duration; `DECISION (uncited)` — it is deployment-legitimate,
  and the age-confound fairness arms of CHANGES.md §19 already cover "cheap covariate"
  critiques). Total **37 channels**.
- Column names: `f"{var}_mean"`, `f"{var}_std"`, `"cycle_len_s"`, `W` vars first then
  `X_s` vars, in the order read from `*_var`.
- Add to `src/config.py`:
  ```python
  NCMAPSS_W_VARS  = ("alt", "Mach", "TRA", "T2")
  NCMAPSS_XS_VARS = ("T24", "T30", "T48", "T50", "P15", "P2", "P21", "P24",
                     "Ps30", "P40", "P50", "Nf", "Nc", "Wf")
  NCMAPSS_FEATURE_COLUMNS = [f"{v}_{s}" for v in NCMAPSS_W_VARS + NCMAPSS_XS_VARS
                             for s in ("mean", "std")] + ["cycle_len_s"]
  DEFAULT_SENSOR_COLUMNS["ncmapss"] = list(NCMAPSS_FEATURE_COLUMNS)
  ```
  The loader must **assert** that the decoded `W_var`/`X_s_var` from the file equal the
  config constants **as sets** and use the file's order for reading; on mismatch raise with
  both lists printed (then fix the constants — do not silently adapt). This is the same
  fail-loud pattern as the registry-drift test.
- Memory discipline: open with `h5py`, read **only** `W_*`, `X_s_*`, `A_*` and the three
  `*_var` name arrays, cast to `float32` on read (`np.asarray(h5[k], dtype=np.float32)`).
  Largest file ≈ 6.5M rows × 18 cols ≈ 500 MB — fine on Colab. Aggregate with a single
  pandas `groupby(["unit","cycle"], sort=True).agg(["mean","std"])` (or a numpy
  `np.add.reduceat` pass if you prefer; pandas is fine at this size). `std` with one-row
  cycles → fill NaN with 0.0.

### 3.3 Parsed-frame cache (critical for cost)

Parsing 1–3 GB of h5 per dataset per run is minutes of Drive I/O; the aggregated frame is
only ~10²–10³ rows. Cache it:

- Path: `Path(config.cache_dir) / f"ncmapss_agg_{config.dataset}_v{NCMAPSS_AGG_VERSION}.npz"`
  with `NCMAPSS_AGG_VERSION = 1` a module constant — **bump it whenever aggregation logic
  changes** (it plays the role `CACHE_SCHEMA_VERSION` plays for embeddings).
- Contents: train values + test values (float32), column list, unit/cycle arrays, and the
  **full-length** per-test-unit cycle counts (so truncation can be re-applied from config
  without re-parsing — see §3.4: the cache stores *untruncated* frames).
- The aggregation has **no config knobs**, so the cache is config-independent by
  construction (aside from the version constant). Idempotent: load if present, else build
  and save. Print a one-liner either way (`[ncmapss] parsed DS02: 9 units, 726 cycles …` /
  `[ncmapss] loaded cached aggregate …`).

### 3.4 Split & test protocol

- Train = the file's `*_dev` units, full run-to-failure. Test = the file's `*_test` units —
  this preserves the dataset's deliberate distribution shift (DS02 test units 14/15 fly
  short/low routes unseen in dev; keep that property, do not resplit).
- Test units are truncated at `config.ncmapss_test_truncation` (new field, default **0.6**)
  of their life, `keep = max(window_size, floor(n * frac))`, guarded `1 <= keep < n`;
  `rul_truth[unit] = n - keep` (cycles). Mirror the XJTU code path (CHANGES.md §22)
  exactly, including the error message when a unit is too short.
- New `Config` fields, added to `_window_key_fields()` **only when
  `dataset_kind() == "ncmapss"`** (same pattern as the xjtu fields — C-MAPSS/XJTU keys must
  not change; there's a test asserting FD001's key is stable, keep it passing):
  ```python
  ncmapss_test_truncation: float = 0.6
  ```
- `max_rul` stays 125 by default and is **effectively inactive** (N-CMAPSS end-of-life is
  59–100 cycles, so the piecewise cap never binds → the target is plain linear RUL, which
  matches N-CMAPSS community practice). Record this observation in CHANGES.md so nobody
  "fixes" it later.

### 3.5 Registry & config wiring

- `Config.dataset_kind()`: names starting with `"DS"` → `"ncmapss"` (check before the
  `"FD"` branch order doesn't matter; keep the error message listing all three families).
- `src/datasets/ncmapss.py` declares
  `DATASETS = ("DS01","DS02","DS03","DS04","DS05","DS06","DS07","DS08a","DS08c","DS08d","DSALL")`,
  `NCMAPSS_SUBDIR = ("N-CMAPSS",)`, `is_available(config)` = glob
  `N-CMAPSS_{config.dataset}*.h5` non-empty (for DSALL: any `N-CMAPSS_DS*.h5`), and
  `load_ncmapss(config)` returning the canonical `(df_train, df_test, rul_truth)` triple.
  File resolution: `sorted(root.glob(f"N-CMAPSS_{config.dataset}*.h5"))`; zero matches →
  `FileNotFoundError` naming the glob; **more than one match → raise** (ambiguous, e.g.
  DS08 prefix collisions are prevented by the exact `DS08a` names, but guard anyway).
- Register in `datasets/__init__.py` (`DATASET_LOADERS["ncmapss"]`,
  `DATASET_FAMILIES["ncmapss"]`). The existing registry-drift test
  (`test_dataset_kind_and_registry_never_drift`) then covers the new family automatically —
  verify it passes, don't weaken it.
- `requirements.txt`: add `h5py>=3.10` under core (tests write synthetic h5, so it cannot
  be GPU-section-only).

### 3.6 Synthetic fixture + tests

`tests/synthetic.py::write_synthetic_ncmapss(dir, dataset="DS02", n_dev_units=3,
n_test_units=2, cycles_per_unit=(10, 14), rows_per_cycle=(15, 25), seed=0)`:
writes `N-CMAPSS_DS02-000.h5` with `W_dev/X_s_dev/A_dev`, `*_test`, and `W_var/X_s_var/A_var`
(byte strings), plus `X_v_*`/`T_*`/`Y_*` arrays (full row count, random — rows are few) so
the fixture matches the real key set even though the loader must never read them. Sensor
values: give each unit a per-channel drift toward failure so heads/baselines can learn
something.

Tests (in `tests/test_datasets.py`, mirroring the XJTU block):
1. Canonical contract: one row per (unit, cycle); columns exactly
   `INDEX + SETTING + NCMAPSS_FEATURE_COLUMNS`; `time_cycles` consecutive from 1;
   `setting_1` constant per unit ∈ {1,2,3}.
2. Aggregates correct: pick one (unit, cycle), compare `alt_mean`/`Wf_std`/`cycle_len_s`
   against manual numpy on the raw fixture arrays (atol 1e-5).
3. Split protocol: train units == dev units (full length); each test unit truncated to
   `max(window_size, floor(0.6 n))`; `rul_truth == n - keep`; changing
   `ncmapss_test_truncation` changes `window_cache_key()` for a DS02 config but NOT for an
   FD001 config.
4. Var-name mismatch: fixture written with a renamed sensor → loader raises listing both sets.
5. Aggregate cache: second `load_ncmapss` call does not reopen the h5 (monkeypatch
   `h5py.File` to raise after first build, or check mtime), and a bumped
   `NCMAPSS_AGG_VERSION` rebuilds.
6. End-to-end smoke: `load_prepared` → `build_embedding_cache` with `MockEmbedder` →
   `run_sweep` on the synthetic DS02 with tiny config (copy the XJTU smoke test; this is
   the test that catches shape/key drift everywhere downstream).
7. Registry: `Config(dataset="DS03").dataset_kind() == "ncmapss"`; `DEFAULT_SENSOR_COLUMNS`
   resolution gives the 37 channels; campaign test (`tests/test_campaign.py`) extended so
   the synthetic campaign covers an ncmapss dataset too.

---

## 4. Task 4: DSALL — the combined N-CMAPSS fleet (the RQ1 high-data arm)

Motivation (record in CHANGES.md): **per-file N-CMAPSS is a low-unit dataset** (6–9 dev
units) — by-unit it sits at the *low* end of the data-efficiency sweep, not the high end
RESEARCH_PLAN §3 hoped for. The high-data arm is the union: all files together ≈ 100+ units
with heterogeneous failure modes and flight classes — a realistic mixed fleet.

- `dataset="DSALL"`: loader iterates every `N-CMAPSS_DS*.h5` **present on disk** (sorted),
  loads each via the same per-file aggregate caches (§3.3 — so DSALL costs nothing extra
  after the per-file parses), renumbers units as `file_index * 1000 + unit` (collision-proof,
  reversible, recorded in the module docstring), keeps each file's dev/test roles, and
  concatenates. `rul_truth` reindexed accordingly.
- `is_available("DSALL")` = at least **two** DS files present (a 1-file DSALL is just that
  file — raise/skip with a message otherwise).
- DSALL cells in results CSVs are keyed `dataset="DSALL"` — no schema change (the `dataset`
  column already exists in every restart key since CHANGES.md §21).
- A DSALL built from a different file subset silently means a different dataset, so the
  member list must be deterministic and in the cache key. Config cannot read disk (cache
  keys are pure functions of config — repo invariant), so: add a config field
  `dsall_datasets: Optional[list] = None`. When set, the loader loads exactly those members
  and **raises** if any is missing on disk, and the sorted list joins
  `_window_key_fields()` (only when `dataset == "DSALL"`). When None, the loader takes
  whatever is on disk — convenience for exploration only, keyed as `"auto"`; recorded runs
  should always set it, which the campaign default does (§6.1). Print the resolved member
  list at load; the run-metadata JSON captures it via the resolved config.
- Tests: two synthetic DS files → DSALL has union of units, no id collisions, per-file
  truncation preserved, `dsall_datasets=["DS02"]` raises (too few / missing member).

---

## 5. Task 5: unit-count grid — auto-append the full fleet

Today `run_sweep`/`run_fairness_baselines` **skip** any `n_units > available`
(`sweep.py:185-187`), so with the default grid `[2,5,10,25,50,100]`:
XJTU-SY (9 train bearings) runs only {2,5} and **never a full-data cell**; DS02 (6 dev
units) runs only {2,5}. Fix in both functions:

```python
available = len(all_units)
counts = sorted({n for n in config.data_unit_counts if n < available} | {available})
```

- FD001–FD004 (100 train units): grid unchanged → **every existing restart key and recorded
  result stays valid** (state this in CHANGES.md).
- XJTU-SY → {2,5,9}; DS02 → {2,5,6}; DSALL → {2,5,10,25,50,~N}.
- Also apply to `run_horizon_eval`'s `n_units_list` handling if it filters the same way
  (check `horizon.py`; the campaign passes `None` = all units, so it may already be fine —
  verify, don't assume).
- Test: 4-unit synthetic config with `data_unit_counts=[2, 50]` produces exactly cells
  {2, 4} in `results_v2.csv`.

---

## 6. Task 6: campaign defaults + notebook

### 6.1 `DEFAULT_DATASET_OVERRIDES` (module constant in `src/campaign.py`)

`run_campaign(dataset_overrides=None)` currently means "no overrides". Change the default to
a new module constant (explicit `{}` still opts out; merging user overrides over defaults —
user wins per dataset **per key**):

```python
DEFAULT_DATASET_OVERRIDES = {
    # XJTU-SY "cycles" are MINUTES (CHANGES.md §22): protocol chosen deliberately here,
    # recorded once, instead of every notebook re-deciding it.
    # DECISION (uncited): max_rul=125 min keeps the piecewise-target convention uniform
    # with C-MAPSS (the unit differs; bearings degrade over 42 min-42 h). window_size=30
    # minutes, tsfm_context_length=256 (the recorded §12 winner shape).
    "XJTU-SY": {"max_rul": 125, "window_size": 30, "tsfm_context_length": 256},
    # N-CMAPSS: defaults are already sane (max_rul cap inactive, §3.4); pin DSALL's
    # member list for deterministic cache keys once all files are downloaded:
    "DSALL": {"dsall_datasets": ["DS01","DS02","DS03","DS04","DS05","DS06","DS07",
                                  "DS08a","DS08c","DS08d"]},
}
```

Keep the existing behavior that `sensor_columns` always resolves to the dataset default
inside the campaign. Print the resolved override per combo (one line) for provenance.

### 6.2 Notebook (`notebooks/colab_main.ipynb`)

- Config cell: document the three `Data/` subfolders (§0.3) and that N-CMAPSS h5 files go
  flat under `Data/N-CMAPSS/`. Note the first N-CMAPSS pass parses h5 → per-file aggregate
  cache in `cache_dir` (minutes each, once).
- Campaign cell: no code change beyond picking up `DEFAULT_DATASET_OVERRIDES`
  automatically; add a markdown note that the campaign now covers
  FD001–FD004 + XJTU-SY + DS01…DS08d + DSALL and skips whatever isn't downloaded.
- Keep the deep-dive section untouched.
- Edit the `.ipynb` **surgically** (json — match the existing cell structure; do not
  regenerate the notebook or touch unrelated cells' outputs/metadata).

### 6.3 Expected compute (sanity for the implementer, not a promise)

Cycle-level framing keeps everything cheap: N-CMAPSS ≈ 40–70 windows/unit × ≤10 units per
file → Stage A per (file × TSFM) is seconds-to-minutes on a T4; XJTU-SY ≈ 8.6k windows of
16 channels — similar. The dominant one-time cost is h5 parsing (§3.3). If Chronos-2 OOMs
on 37-variate × 256-cycle batches, lower `embed_batch_size` — leave a note in the notebook
config cell, don't change the default.

---

## 7. Documentation (Tasks 1–7, `CHANGES.md` sections to append)

- **§25** XJTU condition-3 fix (folder + force), unmatched-folder guard, cache-safety
  argument (§2.1 above).
- **§26** Tolerant data-dir resolution: subdir candidates + depth-1 nesting; not in cache keys.
- **§27** N-CMAPSS loader: the full decision record — cycle aggregation (37 channels incl.
  `cycle_len_s`), W+X_s only (X_v/T/Y excluded as oracles), dev/test split preserved,
  truncation protocol + `ncmapss_test_truncation` in the window key (ncmapss-only),
  condition_norm auto-OFF (Fc grouping available manually), inactive max_rul note, the
  **non-comparability warning** (§3 intro), aggregate cache + `NCMAPSS_AGG_VERSION`.
- **§28** DSALL: role (RQ1 high-data arm), unit renumbering, `dsall_datasets` determinism rule.
- **§29** Unit-count grid auto-appends the full fleet; FD00x grids unchanged.
- **§30** `DEFAULT_DATASET_OVERRIDES` + notebook data-layout instructions.

`README.md`: extend the layout block (`datasets/ncmapss.py`), the Data/ tree (§0.3), and the
campaign description (now 16 datasets incl. DSALL). Keep the "plan is the source of truth"
framing.

---

## 8. Acceptance checklist (the implementer runs all of it; the user runs the last two)

1. `pip install -r requirements.txt && pytest -q` — everything green, CPU-only, no downloads.
2. `grep -rn "DECISION (uncited):" src/` — every new judgment call listed in §3–§6 appears.
3. New-cache-key blast radius check: instantiate `Config(dataset="FD001")` before/after the
   branch — `window_cache_key()` and `embedding_cache_key()` **identical to main** (write
   this as a quick throwaway script, or better: it's already covered by the existing
   stable-key test — confirm).
4. `python -c "from src.campaign import run_campaign; from src.config import CONFIG; run_campaign(CONFIG)"`
   on the repo as-is (only C-MAPSS data present) — FD001–FD004 run/skip correctly, XJTU-SY
   and all DS* report `skipped_no_data` with the documented `Data/` path in the message.
5. **User, on Colab with data in Drive:** Run-all in `colab_main.ipynb`; confirm
   (a) `[ncmapss] parsed …` lines then cached reruns, (b) per-combo CSVs named
   `results/<dataset>_chronos-2_results_v2.csv` appear for XJTU-SY and each DS file,
   (c) data-scaling figures facet per dataset with the full-fleet cell present.
6. **User:** spot-check one N-CMAPSS unit's frame (`load_prepared`) against the exploration
   notebook's Table-5 cycle counts (e.g. DS02 unit 2 → 75 cycles).

## 9. Explicit non-goals (do not do these)

- No new TSFMs (MOMENT/TimesFM/TTM) — separate task; the registry is the slot-in point.
- No raw 1 Hz / sub-cycle N-CMAPSS modeling and no raw XJTU waveform modeling.
- No changes to recorded FD001 winners, existing cache keys, or CSV schemas
  (`RESULTS_SCHEMA_VERSION` stays 2 — new datasets are new *rows*, not new columns).
- No result numbers/claims written anywhere; no experiment-tracking services; no CLI.
- Do not commit any dataset files.
