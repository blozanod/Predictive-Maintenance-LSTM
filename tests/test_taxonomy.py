"""CPU tests for the RQ-F few-shot adjust-vs-replace probe (src/taxonomy.py, §55).

The probe's claim is that a FROZEN TSFM embedding separates fault TYPES with few labels,
and that this can be compared like-for-like against hand-crafted indicators. The
properties that must hold:

  * the secondary labels are aligned 1:1 to the Stage-A cache (and the probe FAILS LOUD
    rather than scoring a misalignment);
  * ``shots`` means "per class", a class with fewer examples contributes what it has, and
    the number of labels ACTUALLY used is recorded (not the number requested);
  * the probe standardizes on the few labelled rows only -- it must not peek at the
    unlabelled pool, which is the whole premise of a few-shot deployment;
  * a separable signal is recovered, and a degenerate draw yields an honest row instead
    of a crash.

The probe is exercised against synthetic C-MAPSS carrying an injected secondary label, so
it is independent of whether the Hydraulic download is present.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from src.config import Config
from src import data as D
from src import taxonomy as TX
from tests.synthetic import write_synthetic_cmapss, MockEmbedder


LABEL = "action_valve"


@pytest.fixture()
def labelled_cfg(tmp_path, monkeypatch):
    """A synthetic C-MAPSS config whose loading path also emits a 3-class secondary
    label, patched at ``data.load_prepared`` so BOTH the Stage-A cache and the probe see
    the same frames (that shared view is exactly what the alignment assert protects)."""
    data_dir = tmp_path / "CMAPSSData"
    write_synthetic_cmapss(data_dir, dataset="FD001", n_train_units=9, n_test_units=6,
                           min_cycles=26, max_cycles=40, seed=7)
    cfg = Config(dataset="FD001", data_dir=str(data_dir),
                 cache_dir=str(tmp_path / "cache"), results_dir=str(tmp_path / "results"),
                 window_size=6, max_rul=60, sensor_columns=["s_2", "s_3", "s_4"],
                 sweep_seeds=[0, 1])
    real = D.load_prepared

    def _with_label(config):
        df_train, df_test = real(config)
        for frame in (df_train, df_test):
            # class = unit id mod 3, and channel s_2 is shifted per class so the label is
            # genuinely LEARNABLE from the window content (0=none, 1=adjust, 2=replace).
            cls = frame["unit_number"].to_numpy() % 3
            frame[LABEL] = cls.astype(float)
            frame["s_2"] = frame["s_2"].to_numpy() + cls * 250.0
        return df_train, df_test

    monkeypatch.setattr(D, "load_prepared", _with_label)
    return cfg


def _embedder(_cfg):
    return MockEmbedder(feature_dim=12)


# ---------------------------------------------------------------------------
# label windows + alignment
# ---------------------------------------------------------------------------
def test_secondary_label_windows_align_to_the_cache(labelled_cfg):
    from src.embeddings import build_embedding_cache, load_embedding_cache
    build_embedding_cache(labelled_cfg, embedder=_embedder(labelled_cfg), verbose=False)
    cache = load_embedding_cache(labelled_cfg)
    tr_y, tr_u, te_y, te_u = TX.secondary_label_windows(labelled_cfg, LABEL)
    assert np.array_equal(tr_u, np.asarray(cache["train_units"]))
    assert np.array_equal(te_u, np.asarray(cache["test_units"]))
    assert tr_y.shape[0] == cache["train_windows"].shape[0]
    assert set(np.unique(tr_y)) == {0.0, 1.0, 2.0}


def test_missing_label_column_fails_loud(labelled_cfg):
    with pytest.raises(KeyError, match="secondary label"):
        TX.secondary_label_windows(labelled_cfg, "action_nonexistent")


def test_misaligned_cache_fails_loud(labelled_cfg, monkeypatch):
    """If the cache was built under a different windowing protocol the probe must refuse
    to score, not silently pair the wrong rows."""
    from src.embeddings import build_embedding_cache
    build_embedding_cache(labelled_cfg, embedder=_embedder(labelled_cfg), verbose=False)
    original = TX.secondary_label_windows      # capture BEFORE patching (no recursion)

    def _short(config, label_column):
        tr_y, tr_u, te_y, te_u = original(config, label_column)
        return tr_y[:-1], tr_u[:-1], te_y, te_u

    monkeypatch.setattr(TX, "secondary_label_windows", _short)
    with pytest.raises(ValueError, match="not aligned to the Stage-A cache"):
        TX.run_taxonomy_probe(labelled_cfg, LABEL, shots=[2], seeds=[0],
                              feature_sources=["embedding"],
                              embedder_factory=_embedder)


# ---------------------------------------------------------------------------
# few-shot sampling
# ---------------------------------------------------------------------------
def test_sample_few_shot_takes_k_per_class_and_is_seeded():
    labels = np.array([0] * 10 + [1] * 10 + [2] * 3)
    idx = TX.sample_few_shot(labels, 4, seed=0)
    counts = np.bincount(labels[idx], minlength=3)
    assert counts.tolist() == [4, 4, 3], "a scarce class contributes all it has"
    assert list(idx) == sorted(idx)
    assert np.array_equal(idx, TX.sample_few_shot(labels, 4, seed=0))
    assert not np.array_equal(idx, TX.sample_few_shot(labels, 4, seed=1))


def test_sample_few_shot_on_empty_labels():
    assert TX.sample_few_shot(np.array([]), 3, seed=0).size == 0


# ---------------------------------------------------------------------------
# scoring helpers
# ---------------------------------------------------------------------------
def test_probe_scores_perfect_and_chance():
    perfect = TX.probe_scores(np.array([0, 1, 0, 1]), np.array([0, 1, 0, 1]),
                              np.array([0.1, 0.9, 0.2, 0.8]))
    assert perfect["accuracy"] == 1.0 and perfect["macro_f1"] == 1.0
    assert perfect["auroc"] == 1.0 and perfect["n_classes_test"] == 2
    # single-class truth -> AUROC undefined, reported as nan (not 0, not a crash)
    one = TX.probe_scores(np.array([1, 1]), np.array([1, 1]), np.array([0.9, 0.8]))
    assert np.isnan(one["auroc"])
    # no scores supplied at all
    none = TX.probe_scores(np.array([0, 1]), np.array([0, 1]))
    assert np.isnan(none["auroc"])


def test_probe_scores_multiclass_auroc():
    y = np.array([0, 1, 2, 0, 1, 2])
    proba = np.eye(3)[y] * 0.7 + 0.1      # rows sum to 1.0, as roc_auc_score requires
    out = TX.probe_scores(y, y, proba)
    assert out["auroc"] == pytest.approx(1.0)
    assert out["n_classes_test"] == 3


def test_probe_scores_handles_a_class_the_probe_never_saw():
    """The probe's score matrix has no column for a class absent from its few labels;
    that must be reported as nan, not raise."""
    y_true = np.array([0, 1, 2, 2])
    y_pred = np.array([0, 1, 0, 1])
    proba = np.tile([0.6, 0.4], (4, 1))     # only 2 columns for 3 true classes
    out = TX.probe_scores(y_true, y_pred, proba)
    assert np.isnan(out["auroc"])
    assert 0.0 <= out["accuracy"] <= 1.0


def test_fit_probe_single_class_draw_predicts_that_class():
    Xtr = np.zeros((3, 4))
    pred, score = TX._fit_probe(Xtr, np.array([1, 1, 1]), np.zeros((5, 4)), seed=0)
    assert pred.tolist() == [1] * 5 and score is None


def test_features_unknown_source_fails_loud():
    with pytest.raises(ValueError, match="unknown feature_source"):
        TX._features("magic", {"train_emb": np.zeros((2, 3))}, "train")


# ---------------------------------------------------------------------------
# the runner
# ---------------------------------------------------------------------------
def test_run_taxonomy_probe_end_to_end(labelled_cfg):
    pytest.importorskip("pycatch22")
    out = TX.run_taxonomy_probe(
        labelled_cfg, LABEL, shots=[1, 4], seeds=[0, 1],
        feature_sources=["embedding", "catch22", "window_stats"],
        embedder_factory=_embedder)
    rows = list(csv.DictReader(open(out)))
    assert out.name == "taxonomy.csv"
    assert {r["feature_source"] for r in rows} == {"embedding", "catch22", "window_stats"}
    assert {r["shots"] for r in rows} == {"1", "4"}
    assert {r["label"] for r in rows} == {LABEL}
    # the embedding rows are tagged as a TSFM row under test; the foils are not
    from src import scoring as S
    tags = {r["model"] for r in rows}
    assert any(S.is_tsfm_model(t) for t in tags)
    assert {"catch22", "window_stats"} <= tags
    for r in rows:
        assert 0.0 <= float(r["accuracy"]) <= 1.0
        assert int(r["n_labelled"]) <= 3 * int(r["shots"])   # 3 classes
        assert int(r["feature_dim"]) > 0


def test_run_taxonomy_probe_records_labels_used_not_requested(labelled_cfg):
    """``n_labelled`` must be what the probe actually trained on, so a scarce class is
    visible in the record rather than hidden behind the requested k."""
    out = TX.run_taxonomy_probe(labelled_cfg, LABEL, shots=[10_000], seeds=[0],
                                feature_sources=["embedding"],
                                embedder_factory=_embedder)
    row = list(csv.DictReader(open(out)))[0]
    assert int(row["shots"]) == 10_000
    assert 0 < int(row["n_labelled"]) < 10_000
    assert int(row["n_classes_train"]) == 3


def test_run_taxonomy_probe_recovers_a_separable_signal(labelled_cfg):
    """With enough labels the probe must do clearly better than chance (1/3 here) --
    otherwise the whole RQ-F comparison would be measuring noise."""
    out = TX.run_taxonomy_probe(labelled_cfg, LABEL, shots=[25], seeds=[0, 1],
                                feature_sources=["window_stats"],
                                embedder_factory=_embedder)
    accs = [float(r["accuracy"]) for r in csv.DictReader(open(out))]
    assert np.mean(accs) > 0.6


def test_run_taxonomy_probe_is_restartable(labelled_cfg):
    kw = dict(shots=[2], seeds=[0], feature_sources=["embedding"],
              embedder_factory=_embedder)
    out = TX.run_taxonomy_probe(labelled_cfg, LABEL, **kw)
    n = len(list(csv.DictReader(open(out))))
    TX.run_taxonomy_probe(labelled_cfg, LABEL, **kw)
    assert len(list(csv.DictReader(open(out)))) == n


def test_run_taxonomy_probe_defaults(labelled_cfg):
    """Exercise the default shots/seeds/feature-source resolution paths."""
    out = TX.run_taxonomy_probe(labelled_cfg, LABEL, shots=[2], seeds=None,
                                feature_sources=None, models=None,
                                embedder_factory=_embedder,
                                out_csv=labelled_cfg.results_path("tx_defaults.csv"))
    rows = list(csv.DictReader(open(out)))
    assert {r["seed"] for r in rows} == {"0", "1"}          # config.sweep_seeds
    assert {r["feature_source"] for r in rows} == set(TX.FEATURE_SOURCES)


def test_plot_taxonomy_renders(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from src import plots as P
    path = tmp_path / "taxonomy.csv"
    rows = []
    for source, base in (("embedding", 0.8), ("catch22", 0.6)):
        for shots in (1, 5, 25):
            for seed in range(2):
                rows.append({"dataset": "Hydraulic", "model": "m", "label": LABEL,
                             "feature_source": source, "shots": shots, "seed": seed,
                             "accuracy": base, "macro_f1": base + 0.01 * seed,
                             "auroc": base})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    saved = P.plot_taxonomy(path, tmp_path / "figs", show=False)
    assert saved and all(p.exists() for p in saved)
    assert any(LABEL in p.stem for p in saved)


def test_plot_taxonomy_skips_an_all_nan_metric(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from src import plots as P
    path = tmp_path / "taxonomy.csv"
    rows = [{"dataset": "H", "model": "m", "label": LABEL, "feature_source": "embedding",
             "shots": 1, "seed": 0, "macro_f1": float("nan")}]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    assert P.plot_taxonomy(path, tmp_path / "figs", show=False) == []


def test_plot_taxonomy_skips_rows_of_other_labels(tmp_path):
    """A taxonomy CSV may hold several components; each gets its own figure and a row
    for a different label must not leak into it."""
    import matplotlib
    matplotlib.use("Agg")
    from src import plots as P
    path = tmp_path / "taxonomy.csv"
    rows = []
    for label, base in (("action_valve", 0.8), ("action_pump", 0.5)):
        for shots in (1, 5):
            rows.append({"dataset": "H", "model": "m", "label": label,
                         "feature_source": "embedding", "shots": shots, "seed": 0,
                         "macro_f1": base})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    saved = P.plot_taxonomy(path, tmp_path / "figs", show=False)
    stems = {p.stem for p in saved}
    assert any("action_valve" in s for s in stems)
    assert any("action_pump" in s for s in stems)


def test_plot_alarm_scaling_ignores_rows_of_another_dataset(tmp_path):
    """Alarm CSVs can hold several datasets; a figure must be faceted per dataset and
    never pool them (the §24 bug, in the alarm arm's renderer)."""
    import matplotlib
    matplotlib.use("Agg")
    from src import plots as P
    path = tmp_path / "alarm_results.csv"
    rows = []
    for ds, ap in (("MetroPT-3", 0.7), ("Backblaze", 0.3)):
        for seed in range(2):
            rows.append({"dataset": ds, "n_units": 4, "seed": seed,
                         "model": "chronos-2_mlp", "loss": "alarm",
                         "alarm_ap": ap, "alarm_auroc": ap, "alarm_recall": ap,
                         "alarm_mean_lead_time": 5.0})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    saved = P.plot_alarm_scaling(path, tmp_path / "figs", show=False)
    stems = {p.stem for p in saved}
    assert any("MetroPT-3" in s for s in stems) and any("Backblaze" in s for s in stems)
