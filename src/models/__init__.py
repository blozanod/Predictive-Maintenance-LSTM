"""Frozen-TSFM embedders, one module per model, behind a small registry.

Each embedder exposes ``embed_windows(contexts) -> (embeddings, loc_scale)`` and
``describe() -> dict`` (the ``embeddings.Embedder`` protocol). ``make_embedder``
picks the class for ``config.model_name`` so Stage A stays model-agnostic; adding
MOMENT/TimesFM/TTM/Moirai is one new module + one registry entry (RESEARCH_PLAN
sec.4; the ``model_name`` string is the documented slot-in point, CHANGES.md).

The generic embedding cache, pooling, and loc/scale handling live in
``src/embeddings.py`` -- these modules are only the concrete backbones.
"""

from __future__ import annotations

from typing import Optional

from ..config import Config
from .chronos import ChronosEmbedder
from .moirai import MoiraiEmbedder
from .moment import MomentEmbedder
from .timesfm import TimesFMEmbedder
from .ttm import TTMEmbedder

# model_name (config.model_name) -> embedder class. Five TSFMs across three families
# (RESEARCH_PLAN §6): multivariate-native (Chronos-2, Moirai-2), univariate (MOMENT,
# TimesFM), tiny channel-mixing (TTM). The four v2 backbones share the plain-patch
# base (models/base.py); only their backbone load/call differ (CHANGES.md §34).
EMBEDDERS = {
    "amazon/chronos-2": ChronosEmbedder,
    "Salesforce/moirai-2": MoiraiEmbedder,
    "AutonLab/MOMENT-1-large": MomentEmbedder,
    "google/timesfm-2.5": TimesFMEmbedder,
    "ibm-granite/granite-timeseries-ttm-r2": TTMEmbedder,
}


def make_embedder(config: Config, device: Optional[str] = None):
    """Instantiate the embedder registered for ``config.model_name``."""
    name = config.model_name
    if name not in EMBEDDERS:
        raise KeyError(
            f"no embedder registered for model_name={name!r}; "
            f"choices: {sorted(EMBEDDERS)}. Add one in src/models/ and register it."
        )
    return EMBEDDERS[name](config, device=device)


__all__ = ["EMBEDDERS", "make_embedder", "ChronosEmbedder", "MoiraiEmbedder",
           "MomentEmbedder", "TimesFMEmbedder", "TTMEmbedder"]
