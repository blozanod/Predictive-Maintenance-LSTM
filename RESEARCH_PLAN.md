# Research Plan: Foundation Models vs. Specialized Models for RUL Prediction

**Author:** Bernardo Lozano · **Date:** July 2026 · **Status:** Draft v1

---

## 1. Viability assessment — read this first

**Your core idea is viable and validated — but it is no longer novel on its own.** Two papers published in June–July 2026 do almost exactly what you propose:

1. **Abdouni, Voisin & Cerisara (PHM Europe 2026)** — frozen Chronos-2 embeddings + a Wide & Deep adapter (embeddings fused with raw normalized sensors) for RUL regression on the full C-MAPSS benchmark. Average RMSE **10.32**, state-of-the-art, no backbone fine-tuning. [DOI 10.36001/phme.2026.v9i1.4906](https://doi.org/10.36001/phme.2026.v9i1.4906)
2. **arXiv 2606.11990 (June 2026)** — frozen Chronos-2 + 2-layer MLP head on industrial sensor data. Beats LSTM, GRU, TCN, Transformer, and gradient-boosting baselines; even a *linear* head on Chronos-2 embeddings beat the best from-scratch baseline (MAE 60 vs 88). Trained with MSE loss. [arXiv:2606.11990](https://arxiv.org/abs/2606.11990)
3. Also relevant: **Dintén & Zorrilla (CMES 2025)** — few-shot RUL on aircraft engines with TSFMs, and Yu et al. (IEEE CASE 2025) on multi-task PHM with TSFMs.

**Consequence:** "Does Chronos-2 + MLP work for RUL?" is answered (yes). Your project stays valuable if you reframe it around what those papers *don't* answer:

- **Data efficiency curves (your stated primary goal).** Neither paper systematically sweeps training-set size. "How many run-to-failure trajectories before a from-scratch model catches up to a frozen TSFM?" is an open, practically decisive question — it's exactly what determines viability for real industrial deployments with sparse failure data. This is your strongest angle. Keep the `data_fractions` sweep in your notebook and make it the centerpiece.
- **Ordinal loss.** Both papers use MSE. Ordinal regression for RUL exists (LSTM-OR, Vishnu et al. 2019, arXiv:1903.09795 — motivated by censored data), but nobody has combined ordinal heads with frozen TSFM embeddings. Legitimate gap.
- **Cross-TSFM comparison.** Both papers only test Chronos-2. Comparing embedding quality across TSFM families (see §4) is unexplored for RUL.
- **Cross-domain breadth.** Testing whether conclusions hold from turbofans to bearings to your real-world sensor data.

So: don't abandon the pipeline — it's the right pipeline, now with published evidence it works. Reposition the *question* from "does it work" to "**when** does it win, **which** TSFM wins, and **with what loss**."

### Technical notes on your proposed pipeline

- `Chronos2Pipeline.embed()` exists (added in chronos-forecasting 2.x) and returns last-layer encoder embeddings — your plan is directly implementable. Chronos-2 is multivariate-native (group attention across series), so pass all sensor channels as one group rather than embedding channels independently — this is a real advantage over univariate TSFMs.
- **Pooling decision:** `embed()` returns one embedding per patch. Decide and ablate: last-patch vs. mean-pool vs. flatten-all (the PHM paper flattens). Document it — it materially affects results.
- **Cache embeddings to disk.** The backbone is frozen, so embed each window once and train MLPs on cached tensors. Makes the data-fraction × seed × loss grid cheap.
- Your CONFIG (max_rul=125 clipping, window 30) matches community convention for C-MAPSS — good, keeps you comparable to published numbers.
- **One caution on "ordinal, not entropy":** modern ordinal losses (CORAL/CORN) *are* built from binary cross-entropies over cumulative thresholds — that's not a flaw, it's the standard construction. If you mean avoiding naive multiclass cross-entropy over RUL bins (which ignores order), correct instinct. See §5.

---

## 2. Research questions

- **RQ1 (primary):** How does RUL accuracy scale with training data volume for frozen-TSFM+head vs. from-scratch models? Where is the crossover point (if any)?
- **RQ2:** Do ordinal losses beat MSE/quantile regression for RUL heads, and does the answer change in low-data regimes?
- **RQ3:** Which TSFM family produces the best embeddings for prognostics, and does forecasting benchmark rank predict prognostics rank?
- **RQ4:** Do conclusions transfer across domains (turbofan sim → bearing vibration → maintenance-log-style telemetry)?
- **RQ5 (stretch):** Does TSFM zero-shot *forecasting* of a health index (no training at all) provide usable RUL in the 0-data regime?

## 3. Datasets

| Dataset | Role | Notes |
|---|---|---|
| **C-MAPSS FD001–FD004** (in repo) | Primary benchmark | Start FD001, but FD002/FD004 (6 operating conditions, 2 fault modes) are where models actually separate. Published SOTA ~RMSE 11–12 (FD001) to ~16–18 (FD004); Chronos-2 adapter avg 10.32. Use all four. |
| **N-CMAPSS** (NASA, 2021) | Primary #2 — strongly recommended | Successor with realistic flight profiles, 47 channels, DS01–DS08. Much larger; ideal for the *high*-data end of your sweep, which C-MAPSS (100 units) can't provide. **Download** (the old PCoE links rot): official zip `https://phm-datasets.s3.amazonaws.com/NASA/17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip` (~20 GB, h5 per DS); index page [NASA PCoE repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/); mirrors: [PHM Society](https://data.phmsociety.org/nasa/), Kaggle (search "N-CMAPSS"). |
| **Azure Predictive Maintenance** | Keep, but demote | Closest to your real-life telemetry+errors+maintenance-log setting — good for RQ4. Caveats: it's a synthetic *tutorial* dataset, labels suit failure-within-window classification more than RUL regression, and no published RUL leaderboard exists. Treat as a realism probe, not a benchmark. |
| **PRONOSTIA/FEMTO or XJTU-SY bearings** | Add one | Run-to-failure vibration; different physics (high-frequency) and truly scarce units (15–16 bearings) — the natural low-data domain for RQ1/RQ4. |
| **MetroPT-3** (Porto metro compressors) | Optional | Real industrial system, but failure labels are sparse/log-derived; use only if RQ4 needs a second realistic dataset. |

Skip AI4I 2020 (tabular, no temporal degradation — wrong task shape).

## 4. Models

### Foundation models (frozen backbone → head)
- **Chronos-2** (amazon/chronos-2, 120M) — your anchor; native multivariate + covariates, official `.embed()`.
- **MOMENT** (CMU, 385M) — the *only* one below explicitly pretrained for general representations (masked reconstruction) rather than pure forecasting; the most interesting head-to-head vs Chronos-2.
- **TimesFM 2.x** (Google) — decoder-only, univariate; embed per channel and concatenate. Used in the CMES few-shot RUL paper — comparability point.
- **Moirai-2 / Moirai-MoE** (Salesforce) — masked encoder, multivariate via flattening.
- **TTM** (IBM Tiny Time Mixers, ~1–5M) — near-zero-cost comparison point; tests whether *scale* matters or just pretraining.

That's 5; if you cut, keep Chronos-2 + MOMENT + TimesFM. Also run **Chronos-2 zero-shot health-index forecasting** for RQ5.

### Specialized / from-scratch baselines
- **MiniRocket** (multivariate) + ridge/linear head — fast, shockingly strong, and itself a "generic frozen features" method: the perfect foil for TSFM embeddings. Add **catch22+GBM** optionally.
- **GBM (LightGBM/XGBoost)** on per-window statistical features (mean/std/min/max/quantiles/slope/last) — the industrial default; the arXiv paper's GBM baseline gives you a published reference point.
- **From-scratch NNs:** MLP (flattened window), **1D-CNN** (Li et al. 2018-style — canonical C-MAPSS baseline), **LSTM** (you have this), and a small **Transformer or TCN** (strongest baselines in arXiv 2606.11990).
- **Floor baselines:** predict-training-mean, and linear regression on cycle count. Cheap, and they catch bugs.

### Heads (identical across all TSFMs for fairness)
2-layer MLP, hidden ~256, dropout — mirror arXiv 2606.11990 (their ablation: linear < 2-layer ≈ 4-layer, so 2 layers is the right default). Ablate linear vs 2-layer on Chronos-2 only.

## 5. Loss functions

- **MSE** — the literature default; your comparability anchor. Run it always.
- **Ordinal (CORN or CORAL)** — bin RUL into K ordered bins (e.g. K=25, width 5 cycles, after clipping at 125). CORN ([Shi, Cao & Raschka, arXiv:2111.08851](https://arxiv.org/abs/2111.08851), PyTorch impl: [coral-pytorch](https://github.com/Raschka-research-group/coral-pytorch)) avoids CORAL's shared-bias constraint; predict RUL as expected value over bin probabilities. Cite LSTM-OR (arXiv:1903.09795) as prior art; your novelty = ordinal + TSFM embeddings + data-efficiency interaction.
- **Optional third arm:** quantile/pinball loss (gives calibrated intervals nearly free, and maintenance decisions care about lower quantiles more than the mean).

Hypothesis worth stating up front: ordinal's inductive bias (order without metric assumptions) should help most in *low*-data regimes — this ties RQ1 and RQ2 together nicely.

## 6. Experimental protocol

**Data-efficiency sweep (the core experiment).** Subsample **by engine unit, not by row** — fractions of rows leak trajectory information and don't simulate "few failures observed." Grid for C-MAPSS FD001 (100 train units): {2, 5, 10, 25, 50, 100} units. Your current `[0.1 … 1.0]` fractions are fine but express them as unit counts in the paper/report.

**Repetitions.** ≥5 seeds per cell; in low-data cells, resample *which* units too (unit choice dominates variance there). Report mean ± std, plot learning curves with error bands.

**Splits & leakage checklist:**
- Fit scalers (and any sensor selection) on training split only, per data fraction.
- Split by unit everywhere; never let windows from one unit cross splits.
- C-MAPSS test protocol: predict RUL at each test unit's final cycle (standard), using the provided RUL_FDxxx.txt.
- For FD002/FD004: condition-wise normalization (cluster on the 3 settings, normalize per cluster) — standard practice, big effect.

**Metrics.** RMSE (comparability), MAE, and the **NASA scoring function** (asymmetric, punishes late predictions — the metric that matters for maintenance). For ordinal/quantile arms add calibration (coverage of prediction intervals).

**Tracking (your stated goal).** Two distinct curves — track both:
1. **Learning curves**: validation loss + RMSE vs. gradient steps / samples seen, per data-fraction cell (use W&B or a CSV logger).
2. **Data-scaling curves**: final test RMSE / NASA score vs. number of training units — the headline figure.

**Fairness rules.** Same windows, same preprocessing, same split, same budget for hyperparameter tuning per model family (e.g., 20-trial random search on the validation split of the *full-data* cell, then reuse). Re-tune from-scratch models per data fraction (they're the ones that overfit small data); frozen-TSFM heads keep fixed hyperparameters. Log wall-clock and parameter counts — "TTM within X% of Chronos-2 at 1/100 the size" is a result.

## 7. Phases

1. **Weeks 1–2 — Finish FD001 pipeline.** Chronos-2 embed → cache → MLP(MSE) at full data; reproduce ballpark of published numbers (sanity target: RMSE ≤ ~14 on FD001; the Wide&Deep paper hits ~10–11 with raw-feature fusion). Stand up floor baselines + GBM + MiniRocket.
2. **Weeks 3–4 — Core sweep on FD001.** Data-fraction grid × {Chronos-2, MiniRocket, GBM, CNN, LSTM} × {MSE, CORN}, 5 seeds. This alone is a publishable-quality result.
3. **Weeks 5–6 — Cross-TSFM.** Add MOMENT, TimesFM, TTM (Moirai if time). Chronos-2 zero-shot forecasting arm (RQ5).
4. **Weeks 7–8 — Generalize.** FD002–FD004 + N-CMAPSS (or bearings) for the best 2 TSFMs + best 2 baselines. Azure PdM as the realism probe (reframe as RUL-style regression on time-to-failure where labelable; note limitations).
5. **Week 9+ — Write-up.** Headline figure: test error vs. training units, one line per model family, per dataset.

## 8. Risks & mitigations

- **Scooped on the base result** → already happened (§1); mitigated by the data-efficiency + ordinal + cross-TSFM framing. Cite both 2026 papers prominently.
- **Compute** — embedding caching makes head training trivial; the expensive part is one embedding pass per dataset per TSFM. Colab-feasible.
- **C-MAPSS is synthetic & saturated** (SOTA differences are ~1 RMSE point) → the data-efficiency axis, not absolute SOTA, is your contribution; N-CMAPSS/bearings add credibility.
- **Azure PdM label ambiguity** → pre-register how you derive RUL labels from failure records before running models.
- **Ordinal head underperforms** → still a result; report it. Include the expected-value decoding ablation (argmax vs expectation) before concluding.

## 9. Key references

- Abdouni, Voisin, Cerisara (2026). *Leveraging TSFM Embeddings for RUL Prediction.* PHM Europe. [10.36001/phme.2026.v9i1.4906](https://doi.org/10.36001/phme.2026.v9i1.4906)
- *Time-Series Foundation Model Embeddings for RUL Estimation* (2026). [arXiv:2606.11990](https://arxiv.org/abs/2606.11990)
- Ansari et al. *Chronos-2: From Univariate to Universal Forecasting.* [arXiv:2510.15821](https://arxiv.org/abs/2510.15821) · [HF model card](https://huggingface.co/amazon/chronos-2) · [embed() discussion](https://github.com/amazon-science/chronos-forecasting/discussions/354)
- Dintén & Zorrilla (2025). *Few-shot RUL prediction of aircraft engines with TSFMs.* CMES 144(1).
- Vishnu et al. (2019). *Deep Ordinal Regression for RUL from Censored Data.* [arXiv:1903.09795](https://arxiv.org/abs/1903.09795)
- Goswami et al. (2024). *MOMENT: Open time-series foundation models.* arXiv:2402.03885
- Ekambaram et al. (2024). *Tiny Time Mixers.* [arXiv:2401.03955](https://arxiv.org/pdf/2401.03955)
- Saxena et al. (2008). *Damage Propagation Modeling* (C-MAPSS — PDF already in repo)
- Arias Chao et al. (2021). *N-CMAPSS dataset.* NASA Prognostics Data Repository
- Veloso et al. (2022). *MetroPT dataset.* [Nature Sci Data](https://www.nature.com/articles/s41597-022-01877-3)
- Dempster et al. (2021). *MiniRocket.* KDD.
