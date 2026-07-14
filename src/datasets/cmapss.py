"""C-MAPSS (FD001-FD004) raw loader.

Turbofan degradation simulation (Saxena et al. 2008; see the dataset's readme.txt):
26 whitespace-separated columns per row -- unit, cycle, 3 operating settings, 21
sensors -- with a separate ``RUL_FDxxx.txt`` giving each test unit's remaining
cycles at its last observed cycle. All four FD00x variants share one directory
(``data_root/CMAPSSData``); ``config.dataset`` selects the train/test/RUL triple.

This module only READS the raw files into the canonical frame shape. RUL labels,
condition-wise normalization, windowing and splits are applied downstream by
``src/data.py`` (via ``data.load_prepared``).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import Config, ALL_COLUMNS
from .base import resolve_data_dir

# Subdirectory of ``config.data_root`` holding the C-MAPSS text files.
CMAPSS_SUBDIR = "CMAPSSData"


def load_cmapss(config: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load train, test, and ground-truth test-RUL for ``config.dataset``.

    Returns (df_train, df_test, rul_truth) where rul_truth is indexed by
    unit_number and holds the provided remaining-cycles-at-last-observed-cycle.
    """
    data_dir = resolve_data_dir(config, CMAPSS_SUBDIR)
    ds = config.dataset
    df_train = _read_cmapss_txt(data_dir / f"train_{ds}.txt")
    df_test = _read_cmapss_txt(data_dir / f"test_{ds}.txt")

    rul_path = data_dir / f"RUL_{ds}.txt"
    rul_values = pd.read_csv(rul_path, header=None).iloc[:, 0].to_numpy()
    # RUL_FDxxx.txt is ordered by unit number 1..N (Saxena et al. 2008).
    rul_truth = pd.Series(
        rul_values, index=pd.RangeIndex(1, len(rul_values) + 1, name="unit_number"),
        name="rul_truth",
    )
    return df_train, df_test, rul_truth


def _read_cmapss_txt(path: Path) -> pd.DataFrame:
    # 26 whitespace-separated columns, trailing whitespace tolerated.
    df = pd.read_csv(path, sep=r"\s+", header=None, names=ALL_COLUMNS)
    return df
