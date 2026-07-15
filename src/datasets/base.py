"""Shared helpers for the dataset loaders (src/datasets/).

Each dataset family lives in its own module (``cmapss.py``, ``xjtu.py``) and
declares the subdirectory (or accepted subdirectory names) its raw files occupy
under the single ``config.data_root`` folder. ``resolve_data_dir`` turns
(root, subdir) into the concrete path, honouring an explicit ``config.data_dir``
override when one is set (the tests point it straight at a synthetic folder). No
dataset-specific logic lives here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from ..config import Config


def resolve_data_dir(config: Config, subdir: Union[str, tuple]) -> Path:
    """Resolve where a dataset family's raw files live.

    * ``config.data_dir`` set  -> use it verbatim (explicit, dataset-specific path).
    * ``subdir`` a str         -> ``config.data_root / subdir``.
    * ``subdir`` a tuple       -> the first ``config.data_root / candidate`` that
      exists, else ``config.data_root / candidates[0]`` (so a "not found" error names
      the documented layout). Accepts real-world variants like the zip's own name
      ``XJTU-SY_Bearing_Datasets`` without the user renaming anything (CHANGES.md §26).

    Paths are NOT part of any cache key (embeddings are location-independent, §23).
    """
    if config.data_dir:
        return Path(config.data_dir)
    root = Path(config.data_root)
    if isinstance(subdir, str):
        return root / subdir
    for candidate in subdir:
        if (root / candidate).is_dir():
            return root / candidate
    return root / subdir[0]
