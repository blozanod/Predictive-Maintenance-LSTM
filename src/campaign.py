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
    from .sweep import run_sweep, run_fairness_baselines
    from .horizon import build_horizon_cache, run_horizon_eval

    emb = embedder_factory(cfg) if embedder_factory is not None else None
    artifacts: dict = {}
    if "cache" in stages:
        artifacts["cache"] = str(build_embedding_cache(cfg, embedder=emb))
    if "sweep" in stages:
        artifacts["results_csv"] = str(run_sweep(cfg, device=device,
                                                 baseline_names=baseline_names))
    if "fairness" in stages:
        artifacts["fairness_csv"] = str(run_fairness_baselines(cfg))
    if "horizon" in stages:
        build_horizon_cache(cfg, embedder=emb)
        # n_units_list=None => all units of THIS dataset (XJTU has 9, FD001 100)
        artifacts["horizon_csv"] = str(run_horizon_eval(cfg, device=device))
    if "figures" in stages:
        from .plots import plot_data_scaling, plot_horizon
        figs = []
        if "results_csv" in artifacts:
            figs += plot_data_scaling(artifacts["results_csv"], cfg.figures_dir(),
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
    ``baseline_names`` passes through to ``run_sweep`` (None => its default set)."""
    datasets = datasets if datasets is not None else datasets_mod.all_dataset_names()
    models = models if models is not None else sorted(EMBEDDERS)
    dataset_overrides = dataset_overrides or {}

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
            print(f"[campaign] {ds} x {model_name} -> experiment "
                  f"'{cfg.experiment_name}' (stages: {', '.join(stages)})")
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
