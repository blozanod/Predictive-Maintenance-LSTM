"""Run-all campaign: every registered dataset x every registered TSFM, one call.

``run_campaign(base_config)`` sweeps the full cross product of
``datasets.all_dataset_names()`` x ``models.EMBEDDERS`` (CHANGES.md §24):

  * Datasets whose raw files are NOT on disk are SKIPPED with a printed notice
    (``datasets.is_available``) -- a missing download never kills the run-all.
  * Each (dataset, model) combo gets its own experiment namespace
    ``<dataset>_<model-tag>`` (e.g. ``FD002_chronos-2``), so every result CSV,
    per-run dir, and figure filename says exactly which dataset/TSFM produced it:
    ``results/FD002_chronos-2_results_v2.csv``,
    ``figures/FD002_chronos-2_data_scaling_FD002_rmse_clipped.png``, ...
    A non-empty ``base_config.experiment_name`` is prepended to that.
  * Per-combo stages (all restartable, so rerunning the campaign resumes):
    ``cache`` (Stage A) -> ``sweep`` -> ``fairness`` (cycle_reg + gbm_age) ->
    ``horizon`` (Stage A-H + horizon eval, at all-units) -> ``figures``.
  * ``dataset_overrides`` maps a dataset name to config overrides applied to its
    combos -- REQUIRED reading for XJTU-SY, whose "cycles" are minutes (pick
    ``max_rul``/``window_size`` deliberately; src/datasets/xjtu.py docstring).
  * ``sensor_columns`` is reset to each dataset's default (config
    ``DEFAULT_SENSOR_COLUMNS``); a custom channel list belongs in
    ``dataset_overrides``, not the base config (DECISION: a base-config list
    would silently be wrong for every other dataset).
  * A combo that fails does not stop the campaign: the error is printed with
    its traceback and collected into the returned summary; the campaign raises
    only if EVERY combo failed (so a red run-all is never mistaken for green).

Returns a list of per-combo summary dicts (dataset, model, status, artifacts).
"""

from __future__ import annotations

import traceback
from typing import Callable, Optional

from .config import Config
from . import datasets as datasets_mod
from .models import EMBEDDERS

CAMPAIGN_STAGES = ("cache", "sweep", "fairness", "horizon", "figures")

# Stages that only make sense for a RUL (run-to-failure) dataset. On a CENSORED fleet
# (MetroPT, Backblaze) the sweep becomes the binary alarm sweep and these two are
# SKIPPED with a notice rather than producing meaningless numbers (CHANGES.md §54):
#   * ``fairness`` -- ``cycle_reg``/``gbm_age`` regress RUL on elapsed cycles, and a
#     right-censored survivor has no RUL to regress;
#   * ``horizon``  -- its bins are RUL bands, which the alarm arm does not predict.
RUL_ONLY_STAGES = ("fairness", "horizon")

# Stages that need a time-to-event target at all. A CLASSIFICATION dataset (UCI
# Hydraulic, §55) has no failure events, so every one of these is skipped for it and the
# ``sweep`` stage runs the RQ-F taxonomy probe instead -- the dataset's real deliverable.
# Emitting a RUL curve for it would table a constant-target number (its blocks are all
# the same length, so the predict-the-mean floor scores a perfect 0.0) beside C-MAPSS's.
TIME_TO_EVENT_STAGES = ("sweep", "fairness", "horizon")

# Per-dataset protocol defaults recorded ONCE here (CHANGES.md §30) instead of every
# notebook re-deciding them. A user-supplied ``dataset_overrides`` is merged OVER these
# per dataset, per key (the user always wins). Pass ``dataset_overrides={}`` to opt out.
DEFAULT_DATASET_OVERRIDES = {
    # XJTU-SY "cycles" are MINUTES (CHANGES.md §22). DECISION (uncited): max_rul=125 min
    # keeps the piecewise-target convention uniform with C-MAPSS (the unit differs;
    # bearings degrade over ~42 min-42 h); window_size=30 minutes; tsfm_context_length=
    # 256 mirrors the recorded §12 winner shape.
    "XJTU-SY": {"max_rul": 125, "window_size": 30, "tsfm_context_length": 256},
    # DSALL: pin the member list so its cache key is deterministic once every file is
    # downloaded (§28). Absent members raise, so this is safe to leave set -- a partial
    # download surfaces loudly rather than silently unioning a different fleet.
    "DSALL": {"dsall_datasets": ["DS01", "DS02", "DS03", "DS04", "DS05", "DS06",
                                 "DS07", "DS08a", "DS08c"]},
    # ---- Phase-B real datasets (CHANGES.md §54-§56) ----
    # MetroPT-3: one "cycle" = one hour (metropt_cycle_minutes=60), so window_size=30 is
    # 30 h of history and tsfm_context_length=256 is ~11 days -- the §12 winner shape in
    # this dataset's units. DECISION (uncited): alarm_horizon=24 asks "will this APU need
    # an intervention within a DAY?", a lead time a metro depot can actually schedule
    # against (the dataset's own stated requirement, ">= 2 h before non-operational", is
    # met with a wide margin); max_rul=168 (one week) keeps the RUL spine meaningful for
    # the runs that DO end in an observed event while satisfying alarm_horizon < max_rul.
    "MetroPT-3": {"max_rul": 168, "window_size": 30, "tsfm_context_length": 256,
                  "alarm_horizon": 24},
    # UCI Hydraulic: one "cycle" = one 60 s rig cycle and a "unit" = a contiguous
    # constant-fault BLOCK, most of which are only ~10-11 cycles long (the valve factor
    # is varied innermost). window_size MUST therefore be small or most blocks yield no
    # windows at all -- and a TEST block additionally needs window_size+1 cycles to
    # survive truncation. DECISION (uncited): window_size=6 leaves a 10-cycle block 5
    # training windows and still supports the truncated test protocol with margin;
    # max_rul=60 is inactive at this block length (its RUL arm is not the point -- this
    # is the RQ-F anchor, see src/datasets/hydraulic.py).
    "Hydraulic": {"max_rul": 60, "window_size": 6, "tsfm_context_length": 6},
    # Backblaze: one "cycle" = one drive-day. DECISION (uncited): alarm_horizon=30 is
    # the standard "will this drive fail within 30 days?" framing in the SMART-prediction
    # literature; max_rul=180 keeps the horizon well inside the clip point. window_size=30
    # (a month of history) with the §12 winner's 256-day context.
    "Backblaze": {"max_rul": 180, "window_size": 30, "tsfm_context_length": 256,
                  "alarm_horizon": 30},
}


def merge_dataset_overrides(user: Optional[dict]) -> dict:
    """DEFAULT_DATASET_OVERRIDES with ``user`` merged over it, per dataset per key
    (user wins). ``user=None`` -> the defaults; ``user={}`` -> also the defaults (there
    is nothing to override); a dataset present only in ``user`` is kept verbatim."""
    merged = {ds: dict(over) for ds, over in DEFAULT_DATASET_OVERRIDES.items()}
    for ds, over in (user or {}).items():
        merged[ds] = {**merged.get(ds, {}), **over}
    return merged


def campaign_experiment_name(base: Config, dataset: str, model_name: str) -> str:
    """``[<base.experiment_name>_]<dataset>_<model-tag>`` -- the per-combo
    namespace every saved filename carries."""
    tag = model_name.split("/")[-1]
    prefix = f"{base.experiment_name}_" if base.experiment_name else ""
    return f"{prefix}{dataset}_{tag}"


def _combo_config(base: Config, dataset: str, model_name: str,
                  dataset_overrides: dict) -> Config:
    over = dict(dataset_overrides.get(dataset, {}))
    over.setdefault("sensor_columns", None)  # dataset default unless overridden
    return base.replace(
        dataset=dataset, model_name=model_name,
        experiment_name=campaign_experiment_name(base, dataset, model_name),
        **over)


def _run_stages(cfg: Config, stages, device: str,
                embedder_factory: Optional[Callable[[Config], object]],
                baseline_names: Optional[list[str]]) -> dict:
    from .embeddings import build_embedding_cache
    from .sweep import run_sweep, run_alarm_sweep, run_fairness_baselines
    from .horizon import build_horizon_cache, run_horizon_eval

    emb = embedder_factory(cfg) if embedder_factory is not None else None
    censored = cfg.is_censored_dataset()
    classification = cfg.is_classification_dataset()
    artifacts: dict = {}
    if "cache" in stages:
        artifacts["cache"] = str(build_embedding_cache(cfg, embedder=emb))
    if "sweep" in stages:
        if classification:
            # No failure events at all -> the RQ-F taxonomy probe IS the deliverable.
            from .taxonomy import run_taxonomy_probe
            from .datasets.hydraulic import action_column
            artifacts["taxonomy_csv"] = str(run_taxonomy_probe(
                cfg, action_column(cfg.hydraulic_taxonomy_component),
                device=device, embedder_factory=embedder_factory))
        elif censored:
            # A mostly-healthy fleet is scored on the alarm question, into its OWN CSV
            # (the metric columns are disjoint from the RUL ones -- §54). ``baseline_names``
            # is a RUL roster, so the alarm sweep's own default roster is used instead.
            artifacts["alarm_csv"] = str(run_alarm_sweep(cfg, device=device))
        else:
            artifacts["results_csv"] = str(run_sweep(cfg, device=device,
                                                     baseline_names=baseline_names))
    if classification:
        skipped = [s for s in TIME_TO_EVENT_STAGES if s in stages and s != "sweep"]
        if skipped:
            print(f"[campaign] {cfg.dataset}: skipping time-to-event stage(s) {skipped} "
                  f"-- this dataset has NO failure events (its units end because the "
                  f"experiment moved on), so RUL is degenerate; the RQ-F taxonomy probe "
                  f"is its deliverable (§55).")
    elif censored:
        skipped = [s for s in RUL_ONLY_STAGES if s in stages]
        if skipped:
            print(f"[campaign] {cfg.dataset}: skipping RUL-only stage(s) {skipped} -- "
                  f"this is a CENSORED fleet, scored on the alarm/lead-time metric (§54).")
    time_to_event = not (censored or classification)
    if "fairness" in stages and time_to_event:
        artifacts["fairness_csv"] = str(run_fairness_baselines(cfg))
    if "horizon" in stages and time_to_event:
        build_horizon_cache(cfg, embedder=emb)
        # n_units_list=None => all units of THIS dataset (XJTU has 9, FD001 100)
        artifacts["horizon_csv"] = str(run_horizon_eval(cfg, device=device))
    if "figures" in stages:
        from .plots import (plot_alarm_scaling, plot_data_scaling, plot_horizon,
                            plot_taxonomy)
        figs = []
        if "results_csv" in artifacts:
            figs += plot_data_scaling(artifacts["results_csv"], cfg.figures_dir(),
                                      prefix=cfg.result_prefix(), show=False)
        if "alarm_csv" in artifacts:
            figs += plot_alarm_scaling(artifacts["alarm_csv"], cfg.figures_dir(),
                                       prefix=cfg.result_prefix(), show=False)
        if "taxonomy_csv" in artifacts:
            figs += plot_taxonomy(artifacts["taxonomy_csv"], cfg.figures_dir(),
                                  prefix=cfg.result_prefix(), show=False)
        if "horizon_csv" in artifacts:
            figs += plot_horizon(artifacts["horizon_csv"], cfg.figures_dir(),
                                 prefix=cfg.result_prefix(), show=False)
        artifacts["figures"] = [str(f) for f in figs]
    return artifacts


def run_campaign(
    base_config: Config,
    datasets: Optional[list[str]] = None,
    models: Optional[list[str]] = None,
    stages=CAMPAIGN_STAGES,
    dataset_overrides: Optional[dict] = None,
    device: str = "cpu",
    embedder_factory: Optional[Callable[[Config], object]] = None,
    baseline_names: Optional[list[str]] = None,
) -> list[dict]:
    """The run-all entry point (see module docstring). ``embedder_factory`` is
    the CPU-test injection hook, exactly as in ``run_transfer_eval``;
    ``baseline_names`` passes through to ``run_sweep`` (None => its default set).

    ``dataset_overrides`` semantics (CHANGES.md §30): ``None`` (default) uses
    ``DEFAULT_DATASET_OVERRIDES``; a non-empty dict is merged OVER those defaults
    (user wins per dataset per key); an explicit ``{}`` opts out of all overrides."""
    datasets = datasets if datasets is not None else datasets_mod.all_dataset_names()
    models = models if models is not None else sorted(EMBEDDERS)
    if dataset_overrides is None:
        dataset_overrides = merge_dataset_overrides(None)       # the recorded defaults
    elif dataset_overrides:                                     # non-empty -> merge
        dataset_overrides = merge_dataset_overrides(dataset_overrides)
    # else: explicit {} -> no overrides at all (opt-out)

    summary: list[dict] = []
    for ds in datasets:
        probe = base_config.replace(dataset=ds, sensor_columns=None,
                                    **{k: v for k, v in dataset_overrides.get(ds, {}).items()
                                       if k != "sensor_columns"})
        if not datasets_mod.is_available(probe):
            print(f"[campaign] SKIP {ds}: raw data not found under "
                  f"{probe.data_dir or probe.data_root} (download it to include "
                  f"this dataset in the sweep).")
            summary.append({"dataset": ds, "model": None, "status": "skipped_no_data"})
            continue
        for model_name in models:
            cfg = _combo_config(base_config, ds, model_name, dataset_overrides)
            over = dataset_overrides.get(ds, {})
            print(f"[campaign] {ds} x {model_name} -> experiment "
                  f"'{cfg.experiment_name}' (stages: {', '.join(stages)}"
                  f"{'; overrides: ' + str(over) if over else ''})")
            try:
                artifacts = _run_stages(cfg, stages, device, embedder_factory,
                                        baseline_names)
                summary.append({"dataset": ds, "model": model_name,
                                "status": "ok", **artifacts})
            except Exception as e:  # keep the run-all alive; report at the end
                traceback.print_exc()
                print(f"[campaign] FAILED {ds} x {model_name}: {e}")
                summary.append({"dataset": ds, "model": model_name,
                                "status": "failed", "error": f"{type(e).__name__}: {e}"})

    ran = [s for s in summary if s["status"] in ("ok", "failed")]
    failed = [s for s in summary if s["status"] == "failed"]
    print(f"[campaign] done: {len(ran) - len(failed)} ok, {len(failed)} failed, "
          f"{len(summary) - len(ran)} skipped (no data).")
    for s in failed:
        print(f"  FAILED {s['dataset']} x {s['model']}: {s['error']}")
    if ran and len(failed) == len(ran):
        raise RuntimeError("every campaign combo failed -- see tracebacks above")
    return summary
