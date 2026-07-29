"""The RQ-F few-shot probe: does a frozen TSFM embedding separate *adjustment* from
*replacement* faults with few labels -- and does it beat hand-crafted indicators?

RESEARCH_PLAN §2 (RQ-F) / §4 asks whether the embedding a TSFM already produces carries
the FAULT-TYPE distinction a maintenance planner actually acts on: a minor,
self-correcting fault that needs an *adjustment* versus a terminal one that needs a
*part replacement*. It is deliberately scoped as a **few-shot classification probe on
frozen embeddings** (IMPLEMENTATION_PLAN §10 non-goal: no multi-task model), because
real adjust-vs-replace labels are scarce -- the honest question is "how far do a handful
of labels get you?", not "can a big model fit thousands of them".

The probe reuses the Stage-A embedding cache, so it costs no new backbone work:

  1. Load the prepared frames (``data.load_prepared`` -- the ONE loading path) and
     window the SECONDARY LABEL column with the same ``make_windows`` the cache was
     built from, so the labels align 1:1 with the cached embeddings. The alignment is
     ASSERTED against the cached unit ids, never assumed (repo invariant §7).
  2. For each ``shots`` value k, draw k labelled examples PER CLASS from the training
     rows (seeded), fit a light linear probe, and score it on the unit-disjoint test
     rows. Sampling by row -- not by unit -- is correct here: the question is how many
     labelled EVENTS an organization must annotate, and the train/test split is already
     unit-disjoint so no unit's rows appear on both sides.
  3. Do that for each FEATURE SOURCE -- the TSFM embedding vs the catch22 indicator
     bank vs the window statistics -- which is the actual RQ-F comparison.

Writes ``taxonomy.csv``; restartable per (dataset, model, label, feature_source, shots,
seed). CPU-testable through the same ``embedder_factory`` seam as every other runner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .config import Config
from . import data as data_mod
from .evaluate import append_result_row, completed_cells

# One probe cell (a taxonomy.csv row's identity).
TAXONOMY_KEYS = ["dataset", "model", "label", "feature_source", "shots", "seed"]

# Feature sources compared as the classifier's input (the RQ-F contrast).
#   * ``embedding``    -- the frozen TSFM's pooled window embedding (the thing on trial);
#   * ``catch22``      -- the 22 canonical hand-crafted indicators per channel, the
#     "are indicators enough?" foil (RESEARCH_PLAN §6, the same bank catch22_gbm uses);
#   * ``window_stats`` -- the cheap mean/std/slope/... summary the GBM baseline eats.
FEATURE_SOURCES = ("embedding", "catch22", "window_stats")


def _features(source: str, cache: dict, split: str) -> np.ndarray:
    """Feature matrix for one split (``train``/``test``) from one source."""
    if source not in FEATURE_SOURCES:
        raise ValueError(
            f"unknown feature_source {source!r}; choices: {list(FEATURE_SOURCES)}")
    if source == "embedding":
        return np.asarray(cache[f"{split}_emb"], np.float64)
    from .baselines import catch22_features, window_statistics
    windows = np.asarray(cache[f"{split}_windows"], np.float32)
    if source == "catch22":
        return np.asarray(catch22_features(windows), np.float64)
    return np.asarray(window_statistics(windows), np.float64)   # "window_stats"


def secondary_label_windows(config: Config, label_column: str
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-window secondary labels aligned 1:1 to the Stage-A cache.

    Re-windows ``label_column`` out of the prepared frames with the SAME functions
    ``build_embedding_cache`` used (``make_windows`` for train, ``make_test_last_windows``
    for test), so window i here is window i there. Returns
    ``(train_labels, train_units, test_labels, test_units)``; the caller asserts the unit
    arrays against the cache."""
    df_train, df_test = data_mod.load_prepared(config)
    for frame, which in ((df_train, "train"), (df_test, "test")):
        if label_column not in frame.columns:
            raise KeyError(
                f"secondary label {label_column!r} is not in the {which} frame; the "
                f"loader for dataset {config.dataset!r} emits "
                f"{sorted(c for c in frame.columns if c not in config.sensor_columns)}. "
                f"RQ-F needs a dataset whose loader carries a fault-type/severity label "
                f"(UCI Hydraulic, MetroPT).")
    cols = config.sensor_columns
    ws = config.window_size
    _tw, tr_y, tr_u = data_mod.make_windows(df_train, cols, ws, target_col=label_column)
    _ew, te_y, te_u = data_mod.make_test_last_windows(
        df_test, cols, ws, target_col=label_column,
        pad_short=config.pad_short_test_units)
    return tr_y, tr_u, te_y, te_u


def sample_few_shot(labels: np.ndarray, shots: int, seed: int) -> np.ndarray:
    """Row indices of ``shots`` examples PER CLASS, seeded and sorted.

    A class with fewer than ``shots`` examples contributes all of them (and no more) --
    the realistic case for a rare terminal fault, and far more useful than raising. The
    returned index array is what the probe actually trained on, so the caller records
    the true per-class counts rather than the requested ones."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    chosen: list[np.ndarray] = []
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        take = min(int(shots), idx.size)
        chosen.append(rng.choice(idx, size=take, replace=False))
    return np.sort(np.concatenate(chosen)) if chosen else np.empty(0, np.int64)


def probe_scores(y_true: np.ndarray, y_pred: np.ndarray,
                 y_score: Optional[np.ndarray] = None) -> dict:
    """Separability of a fitted probe: accuracy, macro-F1 and (when scores are given and
    both/all classes are present) AUROC -- binary for two classes, macro one-vs-rest
    otherwise. Degenerate cases report ``nan`` rather than raising, so a cell where the
    few-shot draw saw one class still produces an honest row."""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "n_test": int(y_true.size),
        "n_classes_test": int(np.unique(y_true).size),
    }
    auroc = float("nan")
    if y_score is not None and np.unique(y_true).size > 1:
        try:
            if y_score.ndim == 1:
                auroc = float(roc_auc_score(y_true, y_score))
            else:
                auroc = float(roc_auc_score(y_true, y_score, multi_class="ovr",
                                            average="macro"))
        except ValueError:
            # e.g. the probe never saw a class that appears in the test rows, so its
            # score matrix has no column for it -- report nan, do not crash the sweep.
            auroc = float("nan")
    out["auroc"] = auroc
    return out


def _fit_probe(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, seed: int):
    """Fit the light linear probe (standardize -> multinomial logistic regression) and
    return ``(predictions, scores)``. Standardization statistics come from the FEW
    LABELLED TRAINING ROWS ONLY -- the probe must not peek at the unlabelled pool, which
    is the whole premise of a few-shot deployment (leakage rule, Task 2.4).

    A draw containing a single class cannot fit a classifier; the probe then predicts
    that class constantly, which is the honest degenerate answer.

    NaNs are imputed before scaling because the catch22 bank legitimately emits them for
    degenerate (e.g. constant) channels -- ``baselines.catch22_features`` leaves them as
    NaN since LightGBM consumes them natively, but a linear probe cannot. Imputing with
    the median of the LABELLED rows (``keep_empty_features`` so an all-NaN feature stays
    a column rather than silently changing the feature width) keeps the indicator foil
    on the same footing as the embedding instead of crashing the RQ-F comparison."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    classes = np.unique(ytr)
    if classes.size < 2:
        constant = classes[0] if classes.size else 0
        return np.full(len(Xte), constant), None
    clf = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=seed),
    )
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    proba = clf.predict_proba(Xte)
    score = proba[:, 1] if proba.shape[1] == 2 else proba
    return pred, score


def run_taxonomy_probe(
    config: Config,
    label_column: str,
    models: Optional[list[str]] = None,
    feature_sources: Optional[list[str]] = None,
    shots: Optional[list[int]] = None,
    seeds: Optional[list[int]] = None,
    device: str = "cpu",
    embedder_factory: Optional[Callable[[Config], object]] = None,
    out_csv: Optional[str | Path] = None,
) -> Path:
    """Run the RQ-F few-shot adjust-vs-replace probe and append rows to ``taxonomy.csv``.

    For each model x feature source x ``shots`` x seed: draw k labelled examples per
    class from the training windows, fit the linear probe, score it on the test windows,
    and record accuracy / macro-F1 / AUROC alongside the labelled-example count actually
    used. ``embedder_factory`` injects a CPU mock exactly as elsewhere; every cell is
    restartable.

    ``device`` is accepted for interface symmetry with the other runners -- the probe is
    a small sklearn fit on cached features and always runs on CPU."""
    from .embeddings import build_embedding_cache, load_embedding_cache

    models = models if models is not None else [config.model_name]
    feature_sources = (feature_sources if feature_sources is not None
                       else list(FEATURE_SOURCES))
    shots = shots if shots is not None else [1, 2, 5, 10, 25]
    seeds = seeds if seeds is not None else list(config.sweep_seeds)
    out_csv = Path(out_csv) if out_csv else config.results_path("taxonomy.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    done = completed_cells(out_csv, TAXONOMY_KEYS)

    tr_y, tr_u, te_y, te_u = secondary_label_windows(config, label_column)

    for model_name in models:
        cfg = config.replace(model_name=model_name)
        emb = embedder_factory(cfg) if embedder_factory is not None else None
        build_embedding_cache(cfg, embedder=emb)          # idempotent
        cache = load_embedding_cache(cfg)
        # Alignment is the probe's load-bearing assumption: window i of the cache must
        # be window i of the re-derived labels. Assert it (fail loud, invariant §7).
        if not (np.array_equal(np.asarray(cache["train_units"]), tr_u)
                and np.array_equal(np.asarray(cache["test_units"]), te_u)):
            raise ValueError(
                f"secondary-label windows are not aligned to the Stage-A cache for "
                f"{cfg.dataset}/{model_name}: got {tr_u.shape}/{te_u.shape} label rows "
                f"vs {np.asarray(cache['train_units']).shape}/"
                f"{np.asarray(cache['test_units']).shape} cached rows. The cache was "
                f"built from a different window_size/sensor_columns/label protocol -- "
                f"rebuild it before probing.")
        y_tr = tr_y.astype(np.int64)
        y_te = te_y.astype(np.int64)
        model_tag = model_name.split("/")[-1]

        for source in feature_sources:
            key_source = source if source == "embedding" else source
            # The cheap foils do not depend on the backbone; still recorded per model so
            # every model's block is self-contained for plotting (rows dedupe on the key).
            Xtr_all = _features(source, cache, "train")
            Xte_all = _features(source, cache, "test")
            tag = f"{model_tag}_probe" if source == "embedding" else key_source
            for k in shots:
                for seed in seeds:
                    cell = (cfg.dataset, tag, label_column, source, str(k), str(seed))
                    if cell in done:
                        continue
                    idx = sample_few_shot(y_tr, k, seed)
                    pred, score = _fit_probe(Xtr_all[idx], y_tr[idx], Xte_all, seed)
                    row = {
                        "schema_version": 1,
                        "dataset": cfg.dataset, "model": tag, "label": label_column,
                        "feature_source": source, "shots": int(k), "seed": int(seed),
                        "n_labelled": int(idx.size),
                        "n_classes_train": int(np.unique(y_tr[idx]).size),
                        "feature_dim": int(Xtr_all.shape[1]),
                        **probe_scores(y_te, pred, score),
                    }
                    append_result_row(out_csv, row)
                    done.add(cell)
    return out_csv
