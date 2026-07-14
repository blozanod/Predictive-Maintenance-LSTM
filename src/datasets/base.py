"""Shared helpers for the dataset loaders (src/datasets/).

Each dataset family lives in its own module (``cmapss.py``, ``xjtu.py``) and
declares the subdirectory its raw files occupy under the single ``config.data_root``
folder. ``resolve_data_dir`` turns (root, subdir) into the concrete path, honouring
an explicit ``config.data_dir`` override when one is set (the tests point it straight
at a synthetic folder). No dataset-specific logic lives here.
"""

from __future__ import annotations

from pathlib import Path

from ..config import Config


def resolve_data_dir(config: Config, subdir: str) -> Path:
    """Resolve where a dataset family's raw files live.

    * ``config.data_dir`` set  -> use it verbatim (explicit, dataset-specific path).
    * otherwise                -> ``config.data_root / subdir`` (the one-Data-folder
      layout, e.g. ``Data/CMAPSSData`` or ``Data/XJTU-SY``).
    """
    if config.data_dir:
        return Path(config.data_dir)
    return Path(config.data_root) / subdir
