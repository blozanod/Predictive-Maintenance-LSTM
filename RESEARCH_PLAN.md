# Research Plan: When Do Time-Series Foundation Models Work for RUL — and How Should You Collect Data So They Do?

**Author:** Bernardo Lozano · **Date:** July 2026 · **Status:** Draft v2 (repositioned)

---

## 0. What changed from v1, and why

v1 asked *"does Chronos-2 + a head work for RUL?"* and made **data-efficiency curves** the centerpiece. Two June–July 2026 papers (§11) answered the base question (yes) and partly scooped the framing. v2 keeps the same pipeline and the data-efficiency axis but **repositions the whole study around a bigger, un-scooped, practitioner-facing question:**

> **When do time-series foundation models (TSFMs) win for Remaining-Useful-Life prediction, when do they lose, and — concretely — how should an organization collect and label data so that they win?**

The deliverable is no longer a benchmark number. It is a **validated playbook**: by the end we want to be able to tell someone standing up an RUL data-collection program *exactly* how to do it — what to record and keep, what indicators (if any) to compute, how many failure events they must observe before a model is trustworthy, how to label those events, and how to treat the difference between faults that need an *adjustment* versus faults that need a *part replacement*. Every chapter of that playbook is backed by a controlled experiment.

This is easier to justify than v1: it is genuinely novel (nobody has published the *when/why* map across simulated **and** real industrial data with a family of TSFMs), it is decision-relevant to real deployments, and it degrades gracefully — even partial results are useful findings rather than a single headline that must clear a 1-RMSE-point bar.

**What stays exactly as built (the repo is the right pipeline):** frozen-TSFM embeddings → cached → MLP head, vs. from-scratch and cheap-feature baselines; by-unit data-fraction sweeps; both-protocol metrics; condition-wise normalization; horizon-stratified and cold-start-transfer evaluation; the restartable multi-dataset `run_campaign`. See §10 for the built-vs-new inventory.

---

## 1. Study design: interventional first, observational second

We answer "when do TSFMs win" by **deliberately manipulating the conditions** and measuring the effect, not by eyeballing correlations across datasets (which are confounded on a dozen axes at once). The design has two tiers:

- **Tier 1 — Core comparison (observational backbone).** Every dataset × every model × the data-volume sweep × loss arms × seeds, at each dataset's ablation-winner configuration. This is the cross-dataset, cross-model map and is mostly already built.
- **Tier 2 — Factor probes (interventional).** For each playbook factor (§5), we *intervene* on a small set of **anchor datasets** with a reduced model roster (top-2 TSFMs + top-2 cheap foils + best from-scratch NN) so the design stays tractable while still isolating cause. Each probe answers one playbook chapter.

**Intervention rules (what we are allowed to do to the data).** The raw sensor readings are sacrosanct. Interventions are:

- **Additive** — compute new indicator/derived columns from the raw signal (never mutating it). Always allowed.
- **Subtractive / collection-choice** — drop channels, downsample the sampling rate, coarsen the per-cycle aggregation. These are exactly the choices a practitioner makes at install time and never alter the value of a reading we keep. Allowed.
- **Perturbative** — inject synthetic noise/drift into raw readings to map the noise-tolerance frontier. **Allowed on the *simulated* datasets only** (C-MAPSS, N-CMAPSS), where the signal is unrealistically clean and controlled noise makes it *more* lifelike. **Never applied to the real datasets**, and never mixed into a real-data headline number. On real data, the noise axis is covered observationally (the three real datasets have genuinely different natural noise) plus the subtractive interventions.

---

## 2. Research questions (organized as playbook chapters)

Each RQ is simultaneously a chapter of the playbook and an interventional factor.

- **RQ-A — History:** How much history (context length) does a TSFM need to see per unit, and where do returns saturate?
- **RQ-B — How many failures before deploying (primary quantitative axis):** How does win/loss scale with the number of **failure events** observed? Where is the crossover vs. from-scratch models, and does it move across domains? Includes the **zero-failure endpoint** (RQ-Z).
- **RQ-C — What to record & keep:** Which sensor channels matter? Can a TSFM tolerate dropping channels a practitioner would rather not instrument?
- **RQ-D — What to compute (the sharpest TSFM question):** Do TSFMs make hand-crafted condition indicators (RMS, kurtosis, crest factor, …) obsolete? We feed **raw channels vs. classic indicators vs. indicators-only** and measure whether the foundation model still needs feature engineering.
- **RQ-E — How to label:** Piecewise vs. linear RUL, where degradation first becomes observable (the `max_rul` cap), and how to label under **censoring** (mostly-healthy fleets). Ordinal vs. MSE lives here and in RQ-B.
- **RQ-F — Adjustment vs. replacement:** Can a frozen TSFM embedding separate *minor / self-correcting* faults (needs an adjustment) from *terminal* faults (needs a replacement) with few labels, and does it beat hand-crafted indicators at it?
- **RQ-G — Sampling rate / aggregation:** How finely must you sample, and how should sub-cycle data be aggregated?
- **RQ-H — Noise tolerance (sim-only):** How much sensor noise/drift can a TSFM absorb before it loses to a specialized model?
- **RQ-M — Which TSFM, and does multivariate-native modeling matter?** Across five TSFMs spanning multivariate-native, univariate, and tiny-pretrained families: which wins, and is joint cross-channel reasoning necessary or is channel-independent modeling enough?
- **RQ-Z — Zero-shot (0 failures):** Does TSFM zero-shot forecasting of a health index give usable RUL with **no training and no observed failures** — the strongest possible answer to RQ-B?

---

## 3. Datasets

**Everything already tested stays.** Three real industrial datasets are added to move the study from "high-fidelity simulation + one bench rig" toward genuine industrial conditions (real sensor noise, real setups, real alarms, censoring).

| Dataset | Realism | Role | Task shape | Status |
|---|---|---|---|---|
| **C-MAPSS FD001–FD004** | Simulated | Saturated benchmark; comparability anchor; **sim-only noise probe** (RQ-H) | Run-to-failure RUL | Built |
| **N-CMAPSS DS01–DS08c + DSALL** | Simulated (realistic flight profiles) | High-data arm (DSALL); many channels (RQ-C); sub-cycle aggregation (RQ-G); low-data ordinal robustness | Run-to-failure RUL | Built |
| **XJTU-SY bearings** | Real (bench rig) | Non-C-MAPSS physics; raw 25.6 kHz available → **raw-vs-indicators** (RQ-D) | Run-to-failure RUL | Built |
| **MetroPT-3** (Porto Metro air compressor) | **Real industrial** | Alarms & lead-time; real analog+digital noise; fault types (RQ-F); censoring | Adapted (failure reports → time-to-intervention) | **New** |
| **UCI Condition Monitoring of Hydraulic Systems** | **Real rig** | **Adjustment-vs-replacement** (native graded component severity); multi-component | Adapted (per-cycle severity) | **New** |
| **Backblaze drive fleet** | **Real industrial, fleet-scale** | The ideal real-world C-MAPSS alternative; **censoring** flagship; "how many *failures* before deploying" at scale | Censored (RUL-to-failure + right-censored survivors) | **New** |

**Dropped:** Azure Predictive Maintenance (synthetic tutorial dataset) — redundant now that three genuinely real industrial datasets are in.

**Real-dataset notes.**
- **MetroPT-3** — 15 analog + digital signals at 1 Hz over 6 months of a real metro APU; unlabeled but with company failure reports (air-leak events) and an explicit "detect ≥ 2 h before non-operational" requirement. RUL is derived relative to each documented intervention (§4).
- **UCI Hydraulic** — real test rig, 60 s constant-load cycles, ~17 sensors (pressures at 100 Hz, flows, temperatures, vibration, efficiency). Four components (cooler, valve, pump, accumulator) each carry **graded severity labels per cycle** — this *is* the adjustment→replacement gradient, already annotated. Its cyclic controlled-fault structure makes it stronger for RQ-F than for smooth RUL trends; used accordingly.
- **Backblaze** — daily SMART snapshots across a large multi-model drive fleet; failure is flagged on a drive's last day. Restrict to a few high-volume drive models to control heterogeneity, treat non-failed drives as **right-censored**, and expect heavy class imbalance. Requires a censoring-aware protocol (§4, §6) — real engineering, not a drop-in loader. Numbers are never compared across the sub-cycle-window published literature (same non-comparability discipline the repo already applies to XJTU/N-CMAPSS).

---

## 4. Prediction target & label model

**The spine is RUL regression; the clock resets at each maintenance intervention.** "End of life" is redefined as the **next intervention** (repair *or* replace), so each real dataset yields RUL-to-intervention and every unit's series is segmented into runs. This preserves every existing result and comparison and keeps a single clean metric spine.

Layered on top:
- **Adjustment vs. replacement** is a **secondary label** that (i) segments each series into runs and (ii) seeds the RQ-F few-shot study: a classification probe on frozen embeddings asking whether the embedding separates minor/adjustment events from terminal/replacement events with few labels, and whether it beats hand-crafted indicators. Anchored on UCI Hydraulic (native graded severity: e.g. valve 100→73 %, pump leakage none→severe) and MetroPT (distinct fault types). Reuses the embedding cache — no new backbone work.
- **Censoring.** Run-to-failure datasets have none; the real fleets (Backblaze, MetroPT) are mostly-healthy with rare, right-censored failures — the realistic case. The censoring chapter (RQ-E/RQ-B) studies how to learn from a fleet that is mostly still alive, how many *failure events* (not units) are needed, and how censored survivors should enter training. This revives the ordinal/censored-data motivation (LSTM-OR, §11) with a real home.

If the RQ-F side study proves rich, it graduates toward joint multi-task prediction (RUL + failure-type + time-to-alarm) in a later phase — but we do **not** bet the plan on multi-task upfront.

---

## 5. Playbook factors → interventions → anchor datasets

Tier-2 probes run the reduced roster (top-2 TSFMs + top-2 cheap foils + best NN) unless noted.

| Chapter / factor | Intervention type | Anchor datasets |
|---|---|---|
| **RQ-A** History / context length | additive (vary context) | FD004, N-CMAPSS DS02, MetroPT |
| **RQ-B** How many failures before deploying | subtractive (subsample by event) | all (core grid); Backblaze, MetroPT (censored) |
| **RQ-C** What to record & keep (channel selection) | subtractive (drop channels) | N-CMAPSS, Backblaze (SMART availability), Hydraulic |
| **RQ-D** What to compute (raw vs. indicators) | additive (indicator columns) | XJTU (raw 25.6 kHz), MetroPT |
| **RQ-E** How to label (piecewise/linear/ordinal/censored) | additive/label protocol | all (ordinal cheap on cached heads); Backblaze, MetroPT (censoring) |
| **RQ-F** Adjustment vs. replacement | secondary label + few-shot probe | UCI Hydraulic, MetroPT |
| **RQ-G** Sampling rate / aggregation | subtractive (downsample/coarsen) | N-CMAPSS (1 Hz→cycle), XJTU (per-minute), MetroPT (1 Hz) |
| **RQ-H** Noise tolerance | **perturbative — SIM ONLY** | FD001, N-CMAPSS DS02 |

---

## 6. Models

Five TSFMs spanning three families, so "which TSFM" and "does multivariate-native modeling matter" are both answerable (the multivariate-native category has ≥ 2 members, so no conclusion rests on a category-of-one).

| Tier | Models | Role |
|---|---|---|
| **Multivariate-native TSFM** | **Chronos-2** (amazon/chronos-2, ~120M, group attention, official `.embed()`), **Moirai-2** (Salesforce, any-variate) | the "reason across sensors jointly" category |
| **Univariate TSFM** | **MOMENT** (CMU, 385M, masked reconstruction), **TimesFM 2.5** (Google, decoder-only) | channel-independent contrast → answers *does multivariate matter?* |
| **Tiny pretrained foil** | **TTM** (IBM Tiny Time Mixers, ~1–5M) | answers *does scale matter?* |
| **Cheap non-DL foils** (first-class) | **MiniRocket + ridge**, **catch22 + GBM** | answers *do you even need a TSFM / are hand-crafted indicators enough?* |
| **From-scratch NN** | **1D-CNN**, **LSTM** (+ floors: predict-mean, cycle-count linear regression) | the specialized bar to beat |

**Cross-TSFM fairness (RQ-M).** Each model is used **the way a practitioner actually would** — Chronos-2 and Moirai-2 as multivariate groups; MOMENT and TimesFM as per-channel embed → concatenate — with head capacity held fixed and input-dim / parameter / wall-clock counts reported. Then **one controlled ablation** puts all five on a common mean-pooled representation to prove the ranking is not an aggregation artifact. The head, pooling machinery, loc/scale-fusion, and cache infra are shared across backbones (the repo's `src/models/` registry is the slot-in point; only Chronos-2 is registered today).

**Heads (identical across all TSFMs for fairness):** 2-layer MLP, hidden ~256, dropout; linear-vs-2-layer ablated on the anchor only.

---

## 7. Loss functions

- **MSE** — the comparability spine; run always.
- **Ordinal (CORN)** — kept as a **parallel arm across the board**, not exiled to one chapter, because (a) it earned its keep empirically (in low-data N-CMAPSS, MSE error exploded while CORN held in MSE's range) and (b) running both losses on the TSFM head is **nearly free** — heads train in seconds on cached embeddings, so the shared Stage-A embedding pass dominates cost. Ordinal's low-data robustness is an explicit headline finding (RQ-B) and its natural fit for censored labels is central to RQ-E. (Doubling only costs anything for retrained from-scratch NNs, which stay native regressors.)
- **Quantile / pinball** — optional third arm for calibrated intervals where maintenance decisions care about lower quantiles.

---

## 8. Metrics & the definition of "win"

**Primary win metric: the NASA / PHM08 asymmetric score** — late predictions (the dangerous direction) are punished hardest. **RMSE-clipped** is reported alongside as the literature-comparability metric (both-protocol reporting is already built).

**"Too early is also bad."** Excessive earliness wastes useful life and triggers premature maintenance, so it is measured, not ignored:
- a **cost curve swept over a range of early-cost : late-cost ratios** (no single arbitrary ratio), and
- **earliness histograms** — % *dangerously late* vs. % *wastefully early* — reusing the repo's horizon `bias` machinery.

**Win rule (per condition cell).** A TSFM **wins** iff its **seed-mean** primary metric beats the **strongest baseline in that same cell** (the toughest bar — best-per-cell, not a fixed reference) by a margin that survives the repo's existing **paired-seed t-test** (arms share each seed's sampled units and split, so pairing is valid). Otherwise the cell is a **tie** (within noise) or a **loss**. An **absolute-floor guard** flags cells where *everything* fails — a TSFM "win" there is hollow and does not count as a success condition.

**Censored datasets (Backblaze, MetroPT)** cannot use NASA score cleanly. They get a **chapter-specific metric**: precision/recall at a fixed alarm lead-time (and/or a cost curve), reported separately and never tabled against the RUL-datasets' NASA scores.

---

## 9. Experimental protocol & phasing

**Protocol invariants (already enforced in code):** subsample **by unit/event**, never by row; fit scalers and any data-driven selection on the training split of each fraction only; split by unit everywhere; condition-wise normalization for multi-condition datasets; ≥ 5 seeds per cell (resample *which* units in low-data cells); log wall-clock and parameter counts; every result-affecting choice is a `Config` field or a `# DECISION (uncited):` tag; every deviation gets a numbered `CHANGES.md` section; no numeric claim is written from anything but a completed run.

**Phases.**
1. **Backbone integration spikes.** Register Moirai-2, MOMENT, TimesFM 2.5, TTM in `src/models/`. Per backbone: confirm clean representation extraction (`.embed()` or encoder/penultimate hidden states), per-channel loc/scale capture, and the pooling contract. Ship the documented fallback if a model won't expose usable embeddings. *(Gate: each backbone reproduces a sane full-data number on FD001 before it joins the campaign.)*
2. **Real-dataset loaders.** MetroPT-3, UCI Hydraulic, Backblaze loaders into the canonical frame (intervention-reset RUL; censoring flags for Backblaze/MetroPT; adjust-vs-replace secondary labels). Each with a synthetic fixture + CPU smoke test, mirroring the XJTU/N-CMAPSS loader contract.
3. **Core comparison (Tier 1).** Full `run_campaign` over all datasets × 5 TSFMs + foils × data-volume × {MSE, CORN} × 5 seeds, at each dataset's ablation-winner. The cross-model / cross-dataset map + the RQ-M fairness ablation.
4. **Factor probes (Tier 2).** RQ-A/C/D/E/G/H on their anchors with the reduced roster; the RQ-F few-shot adjust-vs-replace probe; the RQ-Z zero-shot arm.
5. **Censoring chapter.** Backblaze + MetroPT under the censoring-aware protocol and the lead-time metric.
6. **Synthesis & write-up.** The playbook: per factor, the recommendation + the evidence; the success map (win/tie/loss by condition); the "do you even need a TSFM / which one" verdict.

**Compute posture.** Cost is dominated by one-time **Stage A embedding passes** (one per dataset per TSFM), all cached; head training is cheap. Colab Pro+ (async/background runtime + credits) is the right fit for the long unattended embedding passes. The extra breadth is deliberate and justified by the repositioned deliverable (§0).

---

## 10. What is already built vs. new

**Built (reused as-is):** Chronos-2 embed → fp16 disk cache → 2-layer MLP head (MSE/CORN/quantile); variable-length TSFM contexts; loc/scale fusion; head-feature ablation; context×pooling ablation with a recorded FD001 winner (`tsfm_context_length=256`, `head_features=emb+locscale`, `pooling=mean`; full-data clipped RMSE ≈ 10.7, 5 seeds); by-unit data-scaling sweep; both-protocol NASA/RMSE/MAE evaluation; condition-wise normalization; horizon-stratified eval (per-RUL-bin RMSE/MAE/bias/NASA); cold-start transfer; fairness arms (cycle-count floor, GBM-with-age); XJTU-SY, N-CMAPSS (DS01–DS08c) + DSALL loaders; restartable `run_campaign` with per-dataset overrides.

**New (this plan):**
- **Models:** Moirai-2, MOMENT, TimesFM 2.5, TTM embedders (Chronos-2 only, today) + the RQ-M common-representation fairness ablation.
- **Datasets:** MetroPT-3, UCI Hydraulic, Backblaze loaders (+ intervention-reset RUL, censoring, adjust-vs-replace labels); drop Azure.
- **Targets/metrics:** censoring-aware protocol + lead-time metric; earliness cost-curve + histograms; the formal per-cell win/tie/loss rule + floor guard.
- **Interventions:** channel-drop, downsample/aggregation-coarsening, sim-only noise/drift injection; the RQ-D raw-vs-indicators harness; the RQ-F few-shot classification probe; the RQ-Z zero-shot health-index arm.

---

## 11. Risks & mitigations

- **Embedding extraction not turnkey for Moirai-2 / TimesFM 2.5 / TTM** → Phase-1 integration spikes with a documented fallback (encoder/penultimate hidden states); a backbone that can't expose usable representations is reported as such, not forced.
- **Backblaze scale & censoring** → restrict to a few high-volume drive models; censoring-aware protocol + chapter-specific metric; never compared to sub-cycle-window published numbers.
- **Real-label ambiguity (MetroPT/Hydraulic)** → pre-register the intervention-reset RUL and adjust-vs-replace labeling protocol before running models; document every judgment as a `# DECISION (uncited):` tag + `CHANGES.md` section.
- **Combinatorial explosion** → the two-tier design (full grid only in Tier 1; reduced roster + anchors in Tier 2) bounds it; cached embeddings keep the loss/seed axes cheap.
- **Adjustment-vs-replacement labels are scarce in public data** → operationalized via Hydraulic's native graded severity + MetroPT fault types as a bounded few-shot probe, with the scarcity stated honestly rather than over-claimed.
- **Cross-dataset non-comparability** → same-protocol comparison only; simulated-noise numbers fenced from real-data headlines; published-number comparisons only where the protocol matches.

---

## 12. Key references

- Abdouni, Voisin, Cerisara (2026). *Leveraging TSFM Embeddings for RUL Prediction.* PHM Europe. [10.36001/phme.2026.v9i1.4906](https://doi.org/10.36001/phme.2026.v9i1.4906)
- *Time-Series Foundation Model Embeddings for RUL Estimation* (2026). [arXiv:2606.11990](https://arxiv.org/abs/2606.11990)
- Ansari et al. *Chronos-2: From Univariate to Universal Forecasting.* [arXiv:2510.15821](https://arxiv.org/abs/2510.15821) · [HF model card](https://huggingface.co/amazon/chronos-2)
- Woo et al. *Moirai / Unified Training of Universal Time-Series Forecasting Transformers.* (Salesforce, uni2ts)
- Goswami et al. (2024). *MOMENT: Open Time-Series Foundation Models.* arXiv:2402.03885
- Das et al. *TimesFM: A decoder-only foundation model for time-series forecasting.* (Google)
- Ekambaram et al. (2024). *Tiny Time Mixers.* [arXiv:2401.03955](https://arxiv.org/pdf/2401.03955)
- Dintén & Zorrilla (2025). *Few-shot RUL prediction of aircraft engines with TSFMs.* CMES 144(1).
- Vishnu et al. (2019). *Deep Ordinal Regression for RUL from Censored Data.* [arXiv:1903.09795](https://arxiv.org/abs/1903.09795)
- Shi, Cao & Raschka (2021). *CORN: Rank-consistent ordinal regression.* [arXiv:2111.08851](https://arxiv.org/abs/2111.08851)
- Dempster et al. (2021). *MiniRocket.* KDD. · Lubba et al. (2019). *catch22.*
- Saxena et al. (2008). *Damage Propagation Modeling* (C-MAPSS). · Arias Chao et al. (2021). *N-CMAPSS.* NASA PCoE.
- Wang et al. (2020). *XJTU-SY bearing datasets.*
- Veloso et al. (2022). *The MetroPT dataset for predictive maintenance.* [Nature Sci Data](https://www.nature.com/articles/s41597-022-01877-3) · [MetroPT-3 (UCI 791)](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset)
- Helwig, Pignanelli & Schütze (2015). *Condition Monitoring of Hydraulic Systems* (UCI). 
- Backblaze. *Hard Drive Data and Stats (SMART, 2013–present).*
