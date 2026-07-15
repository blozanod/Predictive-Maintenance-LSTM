"""Dataset loaders, one module per dataset family, behind a small registry.

Each loader returns the raw ``(df_train, df_test, rul_truth)`` triple in the
canonical C-MAPSS-shaped frame; labels, condition-wise normalization, windowing
and splits are applied downstream by ``src/data.py``. Adding a dataset (e.g.
N-CMAPSS) is one new module + one ``DATASET_LOADERS`` entry -- the pipeline picks
it up through ``config.dataset_kind()`` with no other change (RESEARCH_PLAN sec.3).
"""

from __future__ import annotations

from ..config import Config
from .base import resolve_data_dir
from . import cmapss, xjtu, ncmapss

# dataset kind (config.dataset_kind()) -> (loader, subdir under config.data_root).
DATASET_LOADERS = {
    "cmapss": (cmapss.load_cmapss, cmapss.CMAPSS_SUBDIR),
    "xjtu": (xjtu.load_xjtu, xjtu.XJTU_SUBDIR),
    "ncmapss": (ncmapss.load_ncmapss, ncmapss.NCMAPSS_SUBDIR),
}

# dataset kind -> family module (DATASETS names + is_available live there).
DATASET_FAMILIES = {"cmapss": cmapss, "xjtu": xjtu, "ncmapss": ncmapss}


def load_raw(config: Config):
    """Dispatch to the loader for ``config.dataset`` and return its raw
    ``(df_train, df_test, rul_truth)`` triple."""
    kind = config.dataset_kind()
    loader, _ = DATASET_LOADERS[kind]
    return loader(config)


def all_dataset_names() -> list[str]:
    """Every dataset name any registered family serves, in registry order --
    the campaign's default sweep list (CHANGES.md §24)."""
    return [name for fam in DATASET_FAMILIES.values() for name in fam.DATASETS]


def is_available(config: Config) -> bool:
    """Cheap on-disk availability check for ``config.dataset`` (no full load)."""
    return DATASET_FAMILIES[config.dataset_kind()].is_available(config)


__all__ = ["DATASET_LOADERS", "DATASET_FAMILIES", "load_raw", "resolve_data_dir",
           "all_dataset_names", "is_available", "cmapss", "xjtu", "ncmapss"]
