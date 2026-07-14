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
from . import cmapss, xjtu

# dataset kind (config.dataset_kind()) -> (loader, subdir under config.data_root).
DATASET_LOADERS = {
    "cmapss": (cmapss.load_cmapss, cmapss.CMAPSS_SUBDIR),
    "xjtu": (xjtu.load_xjtu, xjtu.XJTU_SUBDIR),
}


def load_raw(config: Config):
    """Dispatch to the loader for ``config.dataset`` and return its raw
    ``(df_train, df_test, rul_truth)`` triple."""
    kind = config.dataset_kind()
    loader, _ = DATASET_LOADERS[kind]
    return loader(config)


__all__ = ["DATASET_LOADERS", "load_raw", "resolve_data_dir",
           "cmapss", "xjtu"]
