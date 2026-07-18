"""Shared helpers for the dataset loaders (src/datasets/).

Each dataset family lives in its own module (``cmapss.py``, ``xjtu.py``) and
declares the subdirectory (or accepted subdirectory names) its raw files occupy
under the single ``config.data_root`` folder. ``resolve_data_dir`` turns
(root, subdir) into the concrete path, honouring an explicit ``config.data_dir``
override when one is set (the tests point it straight at a synthetic folder). No
dataset-specific logic lives here.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Union

import numpy as np

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


def resolve_truncation_fraction(config: Config, member_dataset: str, unit_id: int,
                                fixed_fraction: float) -> float:
    """The life fraction at which one test unit's trajectory is truncated.

    * ``test_truncation_mode == "fixed"`` -> ``fixed_fraction`` verbatim (the caller
      passes the dataset's own field, ``xjtu_test_truncation`` / ``ncmapss_test_truncation``),
      so the recorded v1 protocol is byte-identical.
    * ``"random"`` (protocol v2, §32) -> a fraction drawn from ``config.test_truncation_range``
      by a generator seeded on ``(member_dataset, unit_id, test_truncation_seed)``. This is
      deterministic and reproducible, varied across units, and -- with those fields in the
      window cache key -- a pure function of config. The seed is keyed on the MEMBER dataset
      name (e.g. "DS02"), NOT ``config.dataset``, so DSALL's per-file reuse of the same raw
      unit ids never makes two different engines share a fraction.

    Independent of the per-cell training seed on purpose: the test set is a fixed benchmark
    for a given config, evaluated identically across every model and sweep seed.

    DECISION (uncited): drawing the fraction from a SHA256(member|unit|seed)-seeded
    generator (rather than one shared RNG stream) makes each unit's fraction order-
    independent and DSALL-safe; no community standard prescribes it (CHANGES.md §32).
    """
    if config.test_truncation_mode == "fixed":
        return float(fixed_fraction)
    lo, hi = config.test_truncation_range
    key = f"{member_dataset}|{int(unit_id)}|{int(config.test_truncation_seed)}"
    digest = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
    return float(np.random.default_rng(digest).uniform(float(lo), float(hi)))
